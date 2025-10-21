import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal

IC = Literal["aic", "bic", "hqic"]

@dataclass
class ARFit:
    p: int
    phi: Optional[np.ndarray]     # shape (p,) or None if p=0
    sigma2: float                 # residual variance
    n_eff: int                    # number of used observations
    ic_values: dict               # {"aic":..., "bic":..., "hqic":...}

@dataclass
class ARDiagnostics:
    p: int
    phi: Optional[np.ndarray]
    eigenvalues: Optional[np.ndarray]
    spectral_radius: float
    stationary_by_companion: bool
    charpoly_roots: Optional[np.ndarray]
    stationary_by_roots: bool
    ic_used: str

def _lagged_matrix(y: np.ndarray, p: int):
    """
    Create (y_p, X_p) for AR(p): y_t = sum_{i=1}^p phi_i y_{t-i} + e_t
    Returns y_p (n_eff,), X_p (n_eff, p)
    """
    if p == 0:
        return y.copy(), np.empty((y.shape[0], 0))
    T = len(y)
    y_p = y[p:].copy()
    X_p = np.column_stack([y[p - i:T - i] for i in range(1, p + 1)])
    return y_p, X_p

def _ols(y: np.ndarray, X: np.ndarray):
    """
    OLS solution: beta, residuals, sigma2, n_eff
    If X has zero columns (p=0), beta is empty and residuals = y.
    """
    n_eff = len(y)
    if X.size == 0:
        resid = y
        sigma2 = float(np.dot(resid, resid) / n_eff)
        return np.empty((0,)), resid, sigma2, n_eff
    # Solve (X'X) beta = X'y with robust numerics
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float(np.dot(resid, resid) / n_eff)  # ML variance (not n_eff-p)
    return beta, resid, sigma2, n_eff

def _gaussian_ic(sigma2: float, n_eff: int, k: int):
    """
    Gaussian loglik (up to additive constant across models):
      logL = -n_eff/2*(log(2π) + 1) - n_eff/2 * log(sigma2)
    AIC = -2 logL + 2k; BIC = -2 logL + k log(n_eff); HQIC = -2 logL + 2k log(log n_eff)
    k = number of free parameters (here, p AR coefficients; we de-mean instead of fitting intercept)
    """
    if sigma2 <= 0:
        sigma2 = 1e-300
    logL = -0.5 * n_eff * (np.log(2 * np.pi) + 1.0 + np.log(sigma2))
    aic = -2.0 * logL + 2.0 * k
    bic = -2.0 * logL + k * np.log(max(n_eff, 2))
    hq = -2.0 * logL + 2.0 * k * np.log(np.log(max(n_eff, 3)))
    return {"aic": float(aic), "bic": float(bic), "hqic": float(hq)}

def _companion_and_eigs(phi: np.ndarray):
    """
    Build companion matrix and compute eigenvalues.
    AR(p): y_t = φ1 y_{t-1} + ... + φp y_{t-p} + e_t
    Companion matrix Φ has top row [φ1,...,φp], subdiagonal identity.
    """
    p = len(phi)
    if p == 0:
        return None, np.array([])  # no eigenvalues
    Phi = np.zeros((p, p), dtype=float)
    Phi[0, :] = phi
    if p > 1:
        Phi[1:, :-1] = np.eye(p - 1)
    eigs = np.linalg.eigvals(Phi)
    return Phi, eigs

def _charpoly_roots(phi: np.ndarray):
    """
    Roots of: 1 - φ1 z - φ2 z^2 - ... - φp z^p = 0
    Coeffs (descending powers): [-φ_p, ..., -φ_1, 1]
    """
    p = len(phi)
    if p == 0:
        return np.array([])
    coefs = np.r_[ -phi[::-1], 1.0 ]
    return np.roots(coefs)

def select_ar_and_check_stationarity(
    y,
    max_p: int = 12,
    ic: IC = "bic",
    demean: bool = True,
    eps: float = 1e-12,
) -> ARDiagnostics:
    """
    Pure-NumPy AR order selection + stationarity check.

    Parameters
    ----------
    y : array-like (1D)
        Time series.
    max_p : int
        Consider p in {0,...,max_p}.
    ic : {"aic","bic","hqic"}
        Information criterion for order selection.
    demean : bool
        If True, subtract sample mean before fitting (no intercept in regression).
    eps : float
        Tolerance for strict stationarity inequalities.

    Returns
    -------
    ARDiagnostics
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    if y.size < 5:
        raise ValueError("Need at least 5 observations after dropping NaNs.")
    y_fit = y - y.mean() if demean else y.copy()

    fits: list[ARFit] = []
    T = len(y_fit)
    max_p = min(max_p, T - 2)  # ensure at least some rows in design matrix

    for p in range(max_p + 1):
        y_p, X_p = _lagged_matrix(y_fit, p)
        phi, resid, sigma2, n_eff = _ols(y_p, X_p)
        ic_vals = _gaussian_ic(sigma2, n_eff, k=p)
        fits.append(ARFit(p=p, phi=None if p == 0 else phi, sigma2=sigma2, n_eff=n_eff, ic_values=ic_vals))

    # pick best p by chosen IC
    ic_name = ic.lower()
    best = min(fits, key=lambda f: f.ic_values[ic_name])

    # stationarity checks
    if best.p == 0:
        return ARDiagnostics(
            p=0,
            phi=None,
            eigenvalues=None,
            spectral_radius=0.0,
            stationary_by_companion=True,   # white noise is stationary
            charpoly_roots=None,
            stationary_by_roots=True,
            ic_used=ic_name,
        )

    phi = best.phi
    _, eigs = _companion_and_eigs(phi)
    spectral_radius = float(np.max(np.abs(eigs)))
    stationary_companion = bool(spectral_radius < 1.0 - eps)

    roots = _charpoly_roots(phi)
    stationary_roots = bool(np.all(np.abs(roots) > 1.0 + eps))

    return ARDiagnostics(
        p=best.p,
        phi=phi,
        eigenvalues=eigs,
        spectral_radius=spectral_radius,
        stationary_by_companion=stationary_companion,
        charpoly_roots=roots,
        stationary_by_roots=stationary_roots,
        ic_used=ic_name,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    T = 800
    e = rng.normal(size=T)
    x = np.zeros(T)
    # AR(3): 0.7 y_{t-1} - 0.25 y_{t-2} + 0.1 y_{t-3} + e_t
    for t in range(3, T):
        x[t] = 0.7*x[t-1] - 0.25*x[t-2] + 0.1*x[t-3] + e[t]

    diag = select_ar_and_check_stationarity(x, max_p=10, ic="bic", demean=True)
    print("Selected p:", diag.p)
    print("phi:", np.round(diag.phi, 4))
    print("Stationary (companion):", diag.stationary_by_companion)
    print("Min |char poly root|:", round(np.min(np.abs(diag.charpoly_roots)), 6))
    print("Stationary (roots

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yfinance as yf

@dataclass
class ARDiagnostics:
    p: int
    phi: Optional[np.ndarray]           # length p (None if p=0)
    eigenvalues: Optional[np.ndarray]   # eigenvalues of companion matrix
    max_abs_eig: float                  # spectral radius (0 if p=0)
    stationary_by_companion: bool
    charpoly_roots: Optional[np.ndarray]   # roots of 1 - phi1 z - ... - phip z^p = 0
    stationary_by_roots: bool
    info_criterion_used: str

def ar_order_and_stationarity(
    y,
    max_p: int = 12,
    ic: str = "bic", # the information criterion we want to minimise (like AIC or BIC)
    eps: float = 1e-12,
) -> ARDiagnostics:
    """
    Estimate AR(p) order by IC (manual OLS) and test stationarity via:
      (i) companion-matrix eigenvalues and (ii) characteristic polynomial roots.

    Parameters
    ----------
    y : array-like (1D). Time series values. NaNs dropped.
    max_p : int. Max AR order to consider.
    ic : {"aic","bic","hqic"}. Information criterion for order selection.
    eps : float. Numerical buffer for strict inequalities.

    Returns: ARDiagnostics
    """
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    if y.size < 5:
        raise ValueError("Need at least 5 observations after dropping NaNs.")
    y_fit = y.copy()

    def lagged(y_arr: np.ndarray, p_: int):
        if p_ == 0:
            return y_arr.copy(), np.empty((y_arr.shape[0], 0))
        T = len(y_arr)
        y_p = y_arr[p_:].copy()
        X_p = np.column_stack([y_arr[p_ - i:T - i] for i in range(1, p_ + 1)])
        return y_p, X_p

    def ols(y_arr: np.ndarray, X_arr: np.ndarray):
        n_eff = len(y_arr)
        if X_arr.size == 0:
            resid = y_arr
            sigma2 = float(np.dot(resid, resid) / n_eff)
            return np.empty(0), resid, sigma2, n_eff
        beta, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        resid = y_arr - X_arr @ beta
        sigma2 = float(np.dot(resid, resid) / n_eff)  # ML variance
        return beta, resid, sigma2, n_eff

    def gaussian_ic(s2: float, n_eff: int, k: int):
        # logL up to additive constant: -n/2 * log(sigma^2)
        s2 = max(s2, 1e-300)
        logL = -0.5 * n_eff * (np.log(2*np.pi) + 1.0 + np.log(s2))
        aic = -2.0*logL + 2.0*k
        bic = -2.0*logL + k * np.log(max(n_eff, 2))
        hq  = -2.0*logL + 2.0*k * np.log(np.log(max(n_eff, 3)))
        return {"aic": float(aic), "bic": float(bic), "hqic": float(hq)}

    T = len(y_fit)
    max_p = int(min(max_p, T - 2))
    ic_name = ic.lower()

    best = None
    for p_try in range(max_p + 1):
        y_p, X_p = lagged(y_fit, p_try)
        phi_hat, resid, sigma2, n_eff = ols(y_p, X_p)
        ics = gaussian_ic(sigma2, n_eff, k=p_try)  # k=p (no intercept)
        rec = {
            "p": p_try,
            "phi": None if p_try == 0 else phi_hat,
            "sigma2": sigma2,
            "ic": ics[ic_name]
        }
        if (best is None) or (rec["ic"] < best["ic"]):
            best = rec

    p = best["p"]
    if p == 0:
        return ARDiagnostics(
            p=0,
            phi=None,
            eigenvalues=None,
            max_abs_eig=0.0,
            stationary_by_companion=True,  # white noise is stationary
            charpoly_roots=None,
            stationary_by_roots=True,
            info_criterion_used=ic_name,
        )

    phi = best["phi"]
    Phi = np.zeros((p, p), dtype=float)
    Phi[0, :] = phi
    if p > 1:
        Phi[1:, :-1] = np.eye(p - 1, dtype=float)
    eigvals = np.linalg.eigvals(Phi)
    max_abs_eig = float(np.max(np.abs(eigvals)))
    stationary_companion = bool(max_abs_eig < 1.0 - eps)

    # ---------- characteristic polynomial roots ----------
    # Roots of: 1 - phi1 z - ... - phip z^p = 0  (coeffs descending: [-phi_p, ..., -phi_1, 1])
    char_coefs = np.r_[ -phi[::-1], 1.0 ]
    roots = np.roots(char_coefs)
    stationary_roots = bool(np.all(np.abs(roots) > 1.0 + eps))

    return ARDiagnostics(
        p=p,
        phi=phi,
        eigenvalues=eigvals,
        max_abs_eig=max_abs_eig,
        stationary_by_companion=stationary_companion,
        charpoly_roots=roots,
        stationary_by_roots=stationary_roots,
        info_criterion_used=ic_name,
    )


if __name__ == "__main__":
    ticker = "BZ=F"
    df = yf.download(tickers=ticker, period="30d", interval="5m", auto_adjust=False, progress=False
    )
    print(df.head(10))
    df_out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    ts = df_out[['Close']]

    res = ar_order_and_stationarity(ts['Close'].values)
    print(0)

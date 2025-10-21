import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yfinance as yf
from check_order_stationarity import _ols, _gaussian_ic, _lagged_matrix, _companion_and_eigs

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

    T = len(y_fit)
    max_p = int(min(max_p, T - 2))
    ic_name = ic.lower()

    best = None
    for p_try in range(max_p + 1):
        y_p, X_p = _lagged_matrix(y_fit, p_try)
        phi_hat, resid, sigma2, n_eff = _ols(y_p, X_p)
        ics = _gaussian_ic(sigma2, n_eff, k=p_try)  # k=p (no intercept)
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

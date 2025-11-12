from gibbs import GibbsConfig, is_stationary_from_phi, GibbsResult, gibbs_ar, _lagged_matrix
import yfinance as yf
import pandas as pd
from check_order_stationarity import select_ar_and_check_stationarity
from partial_autocorrelation import phi_to_pacf, pacf_to_phi
import numpy as np
from scipy.stats import beta as beta_dist
from typing import Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass




def companion_matrix(phi: np.ndarray) -> np.ndarray:
    p = len(phi)
    Phi = np.zeros((p, p))
    Phi[0, :] = phi
    if p > 1:
        Phi[1:, :-1] = np.eye(p - 1)
    return Phi


@dataclass
class GibbsPACFConfig:
    p: int
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 1
    alpha_beta_prior: tuple = (2.0, 2.0)
    prop_sd: float = 0.05
    a0: float = 2.0
    b0_ig: float = 1.0
    rng_seed: Optional[int] = 42


def gibbs_ar_pacf(y: np.ndarray, cfg: GibbsPACFConfig):
    rng = np.random.default_rng(cfg.rng_seed)
    y = np.asarray(y, float).ravel()
    y_p, X = _lagged_matrix(y, cfg.p)
    n, p = X.shape

    # Initialize parameters
    kappa = np.zeros(p)
    phi = pacf_to_phi(kappa)
    sigma2 = np.var(y_p)

    # Priors
    alpha, beta = cfg.alpha_beta_prior

    n_save = (cfg.n_iter - cfg.burn) // cfg.thin
    kappa_samps = np.empty((n_save, p))
    sigma2_samps = np.empty(n_save)

    save_i = 0

    for it in range(cfg.n_iter):

        # Step 1: MH update for kappa
        kappa_prop = kappa + rng.normal(0, cfg.prop_sd, size=p)
        kappa_prop = np.clip(kappa_prop, -0.9999, 0.9999)
        phi_prop = pacf_to_phi(kappa_prop)

        # Log-likelihoods
        resid_curr = y_p - X @ phi
        resid_prop = y_p - X @ phi_prop
        ll_curr = -0.5 * np.sum(resid_curr**2) / sigma2
        ll_prop = -0.5 * np.sum(resid_prop**2) / sigma2

        def log_prior_kappa(k):
            x = 0.5 * (k + 1)  # map to (0,1)
            return np.sum(beta_dist.logpdf(x, alpha, beta))

        lp_curr = log_prior_kappa(kappa)
        lp_prop = log_prior_kappa(kappa_prop)

        log_acc_ratio = (ll_prop + lp_prop) - (ll_curr + lp_curr)
        if np.log(rng.uniform()) < log_acc_ratio:
            kappa = kappa_prop
            phi = phi_prop  # accept move

        # ---- Step 2: sample sigma2 | y, phi
        resid = y_p - X @ phi
        an = cfg.a0 + 0.5 * n
        bn = cfg.b0_ig + 0.5 * np.sum(resid**2)
        sigma2 = 1.0 / rng.gamma(an, 1.0 / bn)

        # save smple
        if it >= cfg.burn and ((it - cfg.burn) % cfg.thin == 0):
            kappa_samps[save_i] = kappa
            sigma2_samps[save_i] = sigma2
            save_i += 1

    return dict(kappa_samples=kappa_samps,
                sigma2_samples=sigma2_samps)



def plot_gibbs_pacf_traces(
    res: dict,
    true_kappa: np.ndarray = None,
    true_phi: np.ndarray = None,
    true_sigma2: float = None,
    figsize: tuple = (10, 6)
):

    kappa_samps = res["kappa_samples"]
    sigma2_samps = res["sigma2_samples"]
    p = kappa_samps.shape[1]

    fig, axes = plt.subplots(p + 1, 1, figsize=figsize, sharex=False)

    # Plot each kappa trace
    for i in range(p):
        axes[i].plot(kappa_samps[:, i], lw=0.7, color="tab:blue")
        if true_kappa is not None:
            axes[i].axhline(true_kappa[i], ls="--", color="red", lw=1)
        axes[i].set_title(f"Trace: κ{i+1}")
        axes[i].set_ylabel("value")

    # Plot sigma² trace
    axes[-1].plot(sigma2_samps, lw=0.7, color="tab:orange")
    if true_sigma2 is not None:
        axes[-1].axhline(true_sigma2, ls="--", color="red", lw=1)
    axes[-1].set_title("Trace: σ²")
    axes[-1].set_ylabel("value")
    axes[-1].set_xlabel("iteration")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    vix = yf.download("^VIX", period="10y", interval="1d", auto_adjust=False, progress=False)
    vix = vix[["Close"]].rename(columns={"Close": "VIX"})


    res = select_ar_and_check_stationarity(y=vix, max_p=15, ic='bic', demean=False)

    pacf_coeff = phi_to_pacf(res.phi)
    # hmm first coefficient of pacf is greater than 1. that doesnt make sense but i am sure that the function works
    print(np.all(np.abs(np.linalg.eigvals(companion_matrix(res.phi))) < 1))
    phi_back = pacf_to_phi(phi_to_pacf(res.phi))
    print(np.max(np.abs(phi_back - res.phi)))
    # I am guessing a numerical error then and i am going to carry on regardless and clip results to -1,1
    pacf_coeff = np.clip(pacf_coeff, -0.999999, 0.999999)

    # CHECK NEW SAMPLER ON SIMULATED DATA
    rng = np.random.default_rng(123)
    true_kappa = np.array([0.6, -0.3])  # true PACF values
    true_phi = pacf_to_phi(true_kappa)
    true_sigma2 = 0.5
    true_sigma = np.sqrt(true_sigma2)

    n = 2000
    burn_in = 200
    x = np.zeros(n + burn_in)
    eps = rng.normal(0, true_sigma, size=n + burn_in)
    for t in range(2, n + burn_in):
        x[t] = np.dot(true_phi, x[t - 2:t][::-1]) + eps[t]
    x = x[burn_in:]

    # ---- run sampler ----
    cfg = GibbsPACFConfig(p=2, n_iter=8000, burn=2000, thin=5)
    res = gibbs_ar_pacf(x, cfg)

    # Plot traces and overlay true values
    plot_gibbs_pacf_traces(
        res,
        true_kappa=true_kappa,
        true_phi=true_phi,
        true_sigma2=true_sigma2
    )



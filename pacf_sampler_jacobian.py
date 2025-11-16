from gibbs import _lagged_matrix
from partial_autocorrelation import phi_to_pacf, pacf_to_phi
import numpy as np
from scipy.stats import beta as beta_dist
from typing import Optional
from dataclasses import dataclass
from gibbs_pacf import plot_gibbs_pacf_traces
import yfinance as yf


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


def kappa_to_z(kappa: np.ndarray) -> np.ndarray:
    # g(kappa) = kappa / sqrt(1 - kappa^2)
    return kappa / np.sqrt(1.0 - kappa**2)


def z_to_kappa(z: np.ndarray) -> np.ndarray:
    # g^{-1}(z) = z / sqrt(1 + z^2)
    return z / np.sqrt(1.0 + z**2)


def gibbs_ar_pacf(y: np.ndarray, cfg: GibbsPACFConfig):
    rng = np.random.default_rng(cfg.rng_seed)
    y = np.asarray(y, float).ravel()
    y_p, X = _lagged_matrix(y, cfg.p)
    n, p = X.shape

    # initialise
    kappa = np.zeros(p)
    z = kappa_to_z(kappa)          # this is just zeros, but keeps the logic consistent
    phi = pacf_to_phi(kappa)
    sigma2 = np.var(y_p)

    alpha_beta = cfg.alpha_beta_prior
    alpha, beta = alpha_beta

    n_save = (cfg.n_iter - cfg.burn) // cfg.thin
    kappa_samps = np.empty((n_save, p))
    sigma2_samps = np.empty(n_save)

    accept_count = 0
    total_proposals = 0
    save_i = 0

    def log_prior_kappa(k):
        # scaled Beta prior on each component
        x = 0.5 * (k + 1.0)  # map (-1,1) to (0,1)
        return np.sum(beta_dist.logpdf(x, alpha, beta))

    for it in range(cfg.n_iter):

        # --- Step 1: MH for kappa using z proposal ---
        total_proposals += 1

        # current log likelihood and prior
        resid_curr = y_p - X @ phi
        ll_curr = -0.5 * np.sum(resid_curr**2) / sigma2
        lp_curr = log_prior_kappa(kappa)

        # propose z'
        z_prop = z + rng.normal(0.0, cfg.prop_sd, size=p)
        kappa_prop = z_to_kappa(z_prop)
        phi_prop = pacf_to_phi(kappa_prop)

        # log likelihood at proposal
        resid_prop = y_p - X @ phi_prop
        ll_prop = -0.5 * np.sum(resid_prop**2) / sigma2
        lp_prop = log_prior_kappa(kappa_prop)

        # proposal correction term (Jacobian of z = g(kappa))
        # log |dz/dkappa| = -1.5 * log(1 - kappa^2)
        log_jac_curr = -1.5 * np.sum(np.log(1.0 - kappa**2))
        log_jac_prop = -1.5 * np.sum(np.log(1.0 - kappa_prop**2))
        log_q_ratio = log_jac_curr - log_jac_prop   # log q(kappa|kappa') - log q(kappa'|kappa)

        log_acc_ratio = (ll_prop + lp_prop) - (ll_curr + lp_curr) + log_q_ratio

        if np.log(rng.uniform()) < log_acc_ratio:
            # accept
            kappa = kappa_prop
            z = z_prop
            phi = phi_prop
            accept_count += 1
            ll_curr = ll_prop
            lp_curr = lp_prop

        # --- Step 2: sigma2 | y, phi ---
        resid = y_p - X @ phi
        an = cfg.a0 + 0.5 * n
        bn = cfg.b0_ig + 0.5 * np.sum(resid**2)
        sigma2 = 1.0 / rng.gamma(an, 1.0 / bn)

        # save
        if it >= cfg.burn and ((it - cfg.burn) % cfg.thin == 0):
            kappa_samps[save_i] = kappa
            sigma2_samps[save_i] = sigma2
            save_i += 1

    accept_rate = accept_count / max(total_proposals, 1)

    return dict(
        kappa_samples=kappa_samps,
        sigma2_samples=sigma2_samps,
        accept_rate=accept_rate,
    )

if __name__ == "__main__":
    true_kappa = np.array([0.6, -0.3, 0.2])  # PACF parameters in (-1,1)
    true_phi = pacf_to_phi(true_kappa)  # convert PACF -> AR coeffs

    true_sigma2 = 0.7
    true_sigma = np.sqrt(true_sigma2)

    print("TRUE kappa:", true_kappa)
    print("TRUE phi:  ", true_phi)
    print("TRUE sigma2:", true_sigma2)

    rng = np.random.default_rng(123)
    n = 4000
    burn = 500
    x = np.zeros(n + burn)
    eps = rng.normal(0, true_sigma, size=n + burn)
    p = 3
    for t in range(p, n + burn):
        x[t] = np.dot(true_phi, x[t - p:t][::-1]) + eps[t]

    x = x[burn:]
    ts = x.copy()

    cfg = GibbsPACFConfig(
        p=3,
        n_iter=10000,
        burn=3000,
        thin=5,
        prop_sd=0.05,
        rng_seed=42
    )

    res = gibbs_ar_pacf(ts, cfg)

    print("MH acceptance rate:", res["accept_rate"])
    plot_gibbs_pacf_traces(
        res,
        true_kappa=true_kappa,
        true_phi=true_phi,
        true_sigma2=true_sigma2,
        figsize=(10, 8)
    )
    plot_gibbs_pacf_traces(res, true_kappa=true_kappa, true_phi=true_phi, true_sigma2=true_sigma2)

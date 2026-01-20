from gibbs_pacf2 import kappa_to_z, z_to_kappa
from gibbs import _lagged_matrix
from partial_autocorrelation import pacf_to_phi
import numpy as np
from scipy.stats import beta as beta_dist
from typing import Optional
from dataclasses import dataclass
from gibbs_pacf import plot_gibbs_pacf_traces
from matplotlib import pyplot as plt

@dataclass
class GibbsPACFConfigMean:
    p: int
    n_iter: int = 5000
    burn: int = 1000
    thin: int = 1
    alpha_beta_prior: tuple = (2.0, 2.0)
    prop_sd: float = 0.01
    a0: float = 2.0
    b0_ig: float = 1.0
    rng_seed: Optional[int] = 42

    # mean prior: mu - N(mu0, c2)
    mu0: float = 0.0
    c2: float = 1e6


def gibbs_ar_pacf_with_mean(y: np.ndarray, cfg: GibbsPACFConfigMean):
    rng = np.random.default_rng(cfg.rng_seed)
    y = np.asarray(y, float).ravel()
    y_p, X = _lagged_matrix(y, cfg.p)  # y_p shape (n*,), X shape (n*, p)
    n_star, p = X.shape

    # --- initialise
    kappa = np.zeros(p)
    z = kappa_to_z(kappa)
    phi = pacf_to_phi(kappa)

    mu = float(np.mean(y_p))  # reasonable init
    sigma2 = float(np.var(y_p))

    alpha, beta = cfg.alpha_beta_prior

    n_save = (cfg.n_iter - cfg.burn) // cfg.thin
    kappa_samps = np.empty((n_save, p))
    mu_samps = np.empty(n_save)
    sigma2_samps = np.empty(n_save)

    accept_count = 0
    total_proposals = 0
    save_i = 0

    ones = np.ones(n_star)

    def log_prior_kappa(k: np.ndarray) -> float:
        # scaled Beta prior on each component: x=(k+1)/2 in (0,1)
        x = 0.5 * (k + 1.0)
        return float(np.sum(beta_dist.logpdf(x, alpha, beta)))

    def c_from_phi(ph: np.ndarray) -> float:
        # c(phi) = 1 - sum_j phi_j
        return float(1.0 - np.sum(ph))

    for it in range(cfg.n_iter):

        # ---------- Step 1: MH for kappa (via z), conditional on (mu, sigma2)
        total_proposals += 1

        c_curr = c_from_phi(phi)
        # e = (y - X phi) - c(phi)*mu*1
        resid_curr = (y_p - X @ phi) - c_curr * mu * ones
        ll_curr = -0.5 * np.sum(resid_curr**2) / sigma2
        lp_curr = log_prior_kappa(kappa)

        # propose in z-space
        z_prop = z + rng.normal(0.0, cfg.prop_sd, size=p)
        kappa_prop = z_to_kappa(z_prop)
        phi_prop = pacf_to_phi(kappa_prop)

        c_prop = c_from_phi(phi_prop)
        resid_prop = (y_p - X @ phi_prop) - c_prop * mu * ones
        ll_prop = -0.5 * np.sum(resid_prop**2) / sigma2
        lp_prop = log_prior_kappa(kappa_prop)

        # Jacobian term for transformation kappa -> z
        log_jac_curr = -1.5 * np.sum(np.log(1.0 - kappa**2))
        log_jac_prop = -1.5 * np.sum(np.log(1.0 - kappa_prop**2))
        log_q_ratio = log_jac_curr - log_jac_prop

        log_acc_ratio = (ll_prop + lp_prop) - (ll_curr + lp_curr) + log_q_ratio

        if np.log(rng.uniform()) < log_acc_ratio:
            kappa = kappa_prop
            z = z_prop
            phi = phi_prop
            accept_count += 1
            c_curr = c_prop  # keep in sync

        # ---------- Step 2: Gibbs for mu | (kappa, sigma2, y)
        # Using r = y - X phi, and model r - N(c(phi)*mu*1, sigma2 I)
        c_curr = c_from_phi(phi)
        r = y_p - X @ phi
        S = float(np.sum(r))

        # posterior variance and mean:
        # V_mu = (1/c2 + n* c^2 / sigma2)^(-1)
        # m_mu = V_mu * (mu0/c2 + c*S/sigma2)
        prec = (1.0 / cfg.c2) + (n_star * (c_curr**2) / sigma2)
        V_mu = 1.0 / prec
        m_mu = V_mu * ((cfg.mu0 / cfg.c2) + (c_curr * S / sigma2))

        mu = float(rng.normal(m_mu, np.sqrt(V_mu)))

        # ---------- Step 3: Gibbs for sigma2 | (kappa, mu, y)
        resid = r - c_curr * mu * ones
        an = cfg.a0 + 0.5 * n_star
        bn = cfg.b0_ig + 0.5 * float(np.sum(resid**2))
        sigma2 = 1.0 / rng.gamma(an, 1.0 / bn)

        # ---------- save
        if it >= cfg.burn and ((it - cfg.burn) % cfg.thin == 0):
            kappa_samps[save_i] = kappa
            mu_samps[save_i] = mu
            sigma2_samps[save_i] = sigma2
            save_i += 1

    accept_rate = accept_count / max(total_proposals, 1)

    return dict(
        kappa_samples=kappa_samps,
        mu_samples=mu_samps,
        sigma2_samples=sigma2_samps,
        accept_rate=accept_rate,
    )



def plot_gibbs_pacf_traces(
    res: dict,
    true_kappa: np.ndarray = None,
    true_mu: float = None,
    true_phi: np.ndarray = None,
    true_sigma2: float = None,
    figsize: tuple = (10, 6),
    savepath: str = '/Users/FreddieLewin/Desktop/dissertation/plots/jacobian_pacf_traces3.png'
):

    kappa_samps = res["kappa_samples"]
    sigma2_samps = res["sigma2_samples"]
    mu_samps = res.get("mu_samples", None)

    p = kappa_samps.shape[1]
    has_mu = mu_samps is not None
    n_axes = p + (2 if has_mu else 1)

    fig, axes = plt.subplots(n_axes, 1, figsize=figsize, sharex=False)

    # If only one axis, make it indexable
    if n_axes == 1:
        axes = [axes]

    # Plot each kappa trace
    for i in range(p):
        axes[i].plot(kappa_samps[:, i], lw=0.7, color="tab:blue")
        if true_kappa is not None:
            axes[i].axhline(true_kappa[i], ls="--", color="red", lw=1)
        axes[i].set_title(f"Trace: κ{i+1}")
        axes[i].set_ylabel("value")

    next_row = p

    # Plot mu trace if present
    if has_mu:
        axes[next_row].plot(mu_samps, lw=0.7, color="tab:green")
        if true_mu is not None:
            axes[next_row].axhline(true_mu, ls="--", color="red", lw=1)
        axes[next_row].set_title("Trace: μ")
        axes[next_row].set_ylabel("value")
        next_row += 1

    # Plot sigma² trace
    axes[next_row].plot(sigma2_samps, lw=0.7, color="tab:orange")
    if true_sigma2 is not None:
        axes[next_row].axhline(true_sigma2, ls="--", color="red", lw=1)
    axes[next_row].set_title("Trace: σ2")
    axes[next_row].set_ylabel("value")
    axes[next_row].set_xlabel("iteration")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()


if __name__ == "__main__":
    true_kappa = np.array([0.6, -0.3, 0.2])
    true_phi = pacf_to_phi(true_kappa)

    true_mu = 0.8
    true_sigma2 = 0.7
    true_sigma = np.sqrt(true_sigma2)

    print("TRUE kappa:", true_kappa)
    print("TRUE phi:  ", true_phi)
    print("TRUE mu:   ", true_mu)
    print("TRUE sigma2:", true_sigma2)

    rng = np.random.default_rng(123)
    n = 10000
    burn = 1000
    x = np.zeros(n + burn)
    eps = rng.normal(0, true_sigma, size=n + burn)
    p = 3

    # simulate using mean form: x_t = mu + sum phi_j (x_{t-j}-mu) + eps_t
    for t in range(p, n + burn):
        x[t] = true_mu + np.dot(true_phi, (x[t - p:t][::-1] - true_mu)) + eps[t]

    x = x[burn:]
    ts = x.copy()

    cfg = GibbsPACFConfigMean(
        p=3,
        n_iter=10000,
        burn=1000,
        thin=10,
        prop_sd=0.01,
        rng_seed=42,
        mu0=0.0,
        c2=1e6
    )

    res = gibbs_ar_pacf_with_mean(ts, cfg)

    print("MH acceptance rate:", res["accept_rate"])
    print("Posterior mean(mu):", np.mean(res["mu_samples"]))
    print("Posterior mean(sigma2):", np.mean(res["sigma2_samples"]))

    plot_gibbs_pacf_traces(
        res,
        true_kappa=true_kappa,
        true_mu=true_mu,
        true_sigma2=true_sigma2,
        figsize=(10, 10),
        savepath='/Users/FreddieLewin/Desktop/dissertation/plots/jacobian_pacf_traces_with_mu.png'
    )







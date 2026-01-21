import numpy as np
import matplotlib.pyplot as plt


def _autocorr_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Fast sample autocorrelation for lags 0..max_lag.
    Uses an unbiased-ish normalisation (divide by lag-0 variance).
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x - np.mean(x)
    n = len(x)
    if n < 2:
        return np.full(max_lag + 1, np.nan)

    # variance at lag 0
    var0 = np.dot(x, x) / n
    if var0 <= 0:
        return np.full(max_lag + 1, np.nan)

    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        acf[lag] = (np.dot(x[:-lag], x[lag:]) / (n - lag)) / var0
    return acf


def plot_mcmc_autocorr(
    res: dict,
    p: int,
    max_lag: int = 100,
    thin: int = 1,
    params=("mu", "sigma2", "kappa"),
    combine_chains: bool = True,
    figsize=(10, 8),
    savepath: str | None = None,
):
    """
    Plot autocorrelation functions (ACF) for MCMC draws in your `res` dict.

    Expected keys in `res`:
      - res["mu_samples"]      shape (S,) or (C,S) or (S*C,)
      - res["sigma2_samples"]  shape (S,) or (C,S) or (S*C,)
      - res["kappa_samples"]   shape (S,p) or (C,S,p)

    Notes
    -----
    - If combine_chains=True and samples are shaped (C,S,...) we concatenate chains.
    - If combine_chains=False we plot one ACF per chain (light) and their mean (bold).
    - `thin` is applied before computing ACF to reduce computation/plot clutter.
    """
    def _as_chain_list(arr, is_kappa=False):
        arr = np.asarray(arr)
        if not is_kappa:
            if arr.ndim == 1:
                return [arr]
            if arr.ndim == 2:
                # assume (C,S)
                return [arr[c, :] for c in range(arr.shape[0])]
            raise ValueError(f"Unexpected shape for scalar draws: {arr.shape}")
        else:
            if arr.ndim == 2:
                # assume (S,p)
                return [arr]
            if arr.ndim == 3:
                # assume (C,S,p)
                return [arr[c, :, :] for c in range(arr.shape[0])]
            raise ValueError(f"Unexpected shape for kappa draws: {arr.shape}")

    # prepare figure grid
    n_rows = 0
    if "mu" in params:
        n_rows += 1
    if "sigma2" in params:
        n_rows += 1
    if "kappa" in params:
        n_rows += p

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    if n_rows == 1:
        axes = [axes]

    row = 0
    lags = np.arange(max_lag + 1)

    # ---- mu ----
    if "mu" in params:
        chains = _as_chain_list(res["mu_samples"], is_kappa=False)
        chains = [c[::thin] for c in chains]
        ax = axes[row]
        if combine_chains or len(chains) == 1:
            x = np.concatenate(chains)
            ax.plot(lags, _autocorr_1d(x, max_lag))
        else:
            acfs = []
            for c in chains:
                a = _autocorr_1d(c, max_lag)
                acfs.append(a)
                ax.plot(lags, a, alpha=0.35)
            ax.plot(lags, np.mean(acfs, axis=0), linewidth=2.5)
        ax.set_title("ACF: mu")
        ax.axhline(0.0, linewidth=1)
        row += 1

    # ---- sigma2 ----
    if "sigma2" in params:
        chains = _as_chain_list(res["sigma2_samples"], is_kappa=False)
        chains = [c[::thin] for c in chains]
        ax = axes[row]
        if combine_chains or len(chains) == 1:
            x = np.concatenate(chains)
            ax.plot(lags, _autocorr_1d(x, max_lag))
        else:
            acfs = []
            for c in chains:
                a = _autocorr_1d(c, max_lag)
                acfs.append(a)
                ax.plot(lags, a, alpha=0.35)
            ax.plot(lags, np.mean(acfs, axis=0), linewidth=2.5)
        ax.set_title("ACF: sigma^2")
        ax.axhline(0.0, linewidth=1)
        row += 1

    # ---- kappa ----
    if "kappa" in params:
        kappa_chains = _as_chain_list(res["kappa_samples"], is_kappa=True)
        kappa_chains = [kc[::thin, :] for kc in kappa_chains]  # each is (S,p)

        for j in range(p):
            ax = axes[row]
            if combine_chains or len(kappa_chains) == 1:
                x = np.concatenate([kc[:, j] for kc in kappa_chains])
                ax.plot(lags, _autocorr_1d(x, max_lag))
            else:
                acfs = []
                for kc in kappa_chains:
                    a = _autocorr_1d(kc[:, j], max_lag)
                    acfs.append(a)
                    ax.plot(lags, a, alpha=0.35)
                ax.plot(lags, np.mean(acfs, axis=0), linewidth=2.5)
            ax.set_title(f"ACF: kappa[{j+1}]")
            ax.axhline(0.0, linewidth=1)
            row += 1

    axes[-1].set_xlabel("Lag")
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=200)
    return fig, axes

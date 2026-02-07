"""
Diagnostic plots for MS-AR Gibbs sampler results.

Includes:
- Trace plots for parameters
- ACF plots for mixing diagnostics
- Posterior histograms vs true values
- State path visualization
- State occupancy analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_trace_and_acf(samples, param_name, true_value=None, max_lag=100):
    """
    Plot trace and ACF for a single parameter.

    Args:
        samples: 1D array of MCMC samples
        param_name: string, name for plot title
        true_value: optional true value to overlay
        max_lag: maximum lag for ACF
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    # Trace plot
    ax1.plot(samples, alpha=0.7, linewidth=0.5)
    if true_value is not None:
        ax1.axhline(true_value, color='red', linestyle='--', label='True', linewidth=2)
        ax1.legend()
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(param_name)
    ax1.set_title(f'Trace: {param_name}')
    ax1.grid(alpha=0.3)

    # ACF plot
    n = len(samples)
    mean = np.mean(samples)
    c0 = np.sum((samples - mean) ** 2) / n

    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            acf[lag] = 1.0
        else:
            ck = np.sum((samples[:-lag] - mean) * (samples[lag:] - mean)) / n
            acf[lag] = ck / c0

    ax2.bar(range(max_lag), acf, width=1.0, alpha=0.7)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axhline(1.96 / np.sqrt(n), color='red', linestyle='--', linewidth=1, label='95% CI')
    ax2.axhline(-1.96 / np.sqrt(n), color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('ACF')
    ax2.set_title(f'Autocorrelation: {param_name}')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_msar_traces(results, true_params=None):
    """
    Create trace plots for all MS-AR parameters.

    Args:
        results: dict from gibbs_msar
        true_params: dict with keys 'kappa', 'mu', 'sigma2' (optional)
    """
    K, p = results['kappa_samples'].shape[1:3]
    n_params = K * (p + 2)  # κ, μ, σ² for each state

    # Calculate grid size
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    idx = 0

    for k in range(K):
        # Plot κ parameters
        for j in range(p):
            ax = axes[idx]
            ax.plot(results['kappa_samples'][:, k, j], alpha=0.7, linewidth=0.5)
            if true_params is not None and 'kappa' in true_params:
                ax.axhline(true_params['kappa'][k, j], color='red',
                           linestyle='--', linewidth=2, label='True')
                ax.legend(fontsize=8)
            ax.set_title(f'$\\kappa_{{{k + 1},{j + 1}}}$')
            ax.set_xlabel('Iteration')
            ax.grid(alpha=0.3)
            idx += 1

        # Plot μ
        ax = axes[idx]
        ax.plot(results['mu_samples'][:, k], alpha=0.7, linewidth=0.5)
        if true_params is not None and 'mu' in true_params:
            ax.axhline(true_params['mu'][k], color='red',
                       linestyle='--', linewidth=2, label='True')
            ax.legend(fontsize=8)
        ax.set_title(f'$\\mu_{{{k + 1}}}$')
        ax.set_xlabel('Iteration')
        ax.grid(alpha=0.3)
        idx += 1

        # Plot σ²
        ax = axes[idx]
        ax.plot(results['sigma2_samples'][:, k], alpha=0.7, linewidth=0.5)
        if true_params is not None and 'sigma2' in true_params:
            ax.axhline(true_params['sigma2'][k], color='red',
                       linestyle='--', linewidth=2, label='True')
            ax.legend(fontsize=8)
        ax.set_title(f'$\\sigma^2_{{{k + 1}}}$')
        ax.set_xlabel('Iteration')
        ax.grid(alpha=0.3)
        idx += 1

    # Hide unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')

    plt.suptitle('MCMC Trace Plots', fontsize=16, y=1.00)
    plt.tight_layout()
    return fig


def plot_posterior_histograms(results, true_params=None):
    """
    Plot posterior histograms for all parameters with true values.
    """
    K, p = results['kappa_samples'].shape[1:3]
    n_params = K * (p + 2)

    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten()

    idx = 0

    for k in range(K):
        # κ parameters
        for j in range(p):
            ax = axes[idx]
            ax.hist(results['kappa_samples'][:, k, j], bins=30,
                    density=True, alpha=0.6, edgecolor='black')
            if true_params is not None and 'kappa' in true_params:
                ax.axvline(true_params['kappa'][k, j], color='red',
                           linestyle='--', linewidth=2, label='True')
                ax.legend(fontsize=8)
            ax.set_title(f'$\\kappa_{{{k + 1},{j + 1}}}$')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.grid(alpha=0.3)
            idx += 1

        # μ
        ax = axes[idx]
        ax.hist(results['mu_samples'][:, k], bins=30,
                density=True, alpha=0.6, edgecolor='black')
        if true_params is not None and 'mu' in true_params:
            ax.axvline(true_params['mu'][k], color='red',
                       linestyle='--', linewidth=2, label='True')
            ax.legend(fontsize=8)
        ax.set_title(f'$\\mu_{{{k + 1}}}$')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3)
        idx += 1

        # σ²
        ax = axes[idx]
        ax.hist(results['sigma2_samples'][:, k], bins=30,
                density=True, alpha=0.6, edgecolor='black')
        if true_params is not None and 'sigma2' in true_params:
            ax.axvline(true_params['sigma2'][k], color='red',
                       linestyle='--', linewidth=2, label='True')
            ax.legend(fontsize=8)
        ax.set_title(f'$\\sigma^2_{{{k + 1}}}$')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3)
        idx += 1

    # Hide unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Posterior Distributions', fontsize=16, y=1.00)
    plt.tight_layout()
    return fig


def plot_state_paths(results, true_states=None, n_paths=5):
    """
    Plot sample state paths and compare to true states.

    Args:
        results: dict from gibbs_msar
        true_states: optional true state sequence
        n_paths: number of sampled paths to show
    """
    state_samples = results['state_samples']
    n_save, T = state_samples.shape
    K = results['mu_samples'].shape[1]

    fig, axes = plt.subplots(n_paths + 2, 1, figsize=(15, 2 * (n_paths + 2)))

    # Plot true states if available
    if true_states is not None:
        axes[0].plot(true_states, color='red', linewidth=2, alpha=0.8)
        axes[0].set_ylabel('State')
        axes[0].set_title('True State Sequence')
        axes[0].set_ylim(-0.5, K - 0.5)
        axes[0].grid(alpha=0.3)
        axes[0].set_yticks(range(K))
    else:
        axes[0].axis('off')

    # Plot sample paths
    indices = np.linspace(0, n_save - 1, n_paths, dtype=int)
    for i, idx in enumerate(indices):
        axes[i + 1].plot(state_samples[idx, :], linewidth=1, alpha=0.8)
        axes[i + 1].set_ylabel('State')
        axes[i + 1].set_title(f'Sampled Path {idx}')
        axes[i + 1].set_ylim(-0.5, K - 0.5)
        axes[i + 1].grid(alpha=0.3)
        axes[i + 1].set_yticks(range(K))

    # Plot posterior mean state probabilities
    state_probs = np.zeros((T, K))
    for k in range(K):
        state_probs[:, k] = np.mean(state_samples == k, axis=0)

    ax = axes[-1]
    for k in range(K):
        ax.plot(state_probs[:, k], label=f'P(S_t={k})', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.set_title('Posterior State Probabilities')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    return fig


def plot_state_occupancy(results, true_states=None):
    """
    Analyze state occupancy over time.
    """
    state_samples = results['state_samples']
    n_save, T = state_samples.shape
    K = results['mu_samples'].shape[1]

    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 2, figure=fig)

    # 1. Histogram of state occupancies
    ax1 = fig.add_subplot(gs[0, 0])
    for k in range(K):
        occupancies = np.sum(state_samples == k, axis=1) / T
        ax1.hist(occupancies, bins=30, alpha=0.6, label=f'State {k}', edgecolor='black')
    if true_states is not None:
        for k in range(K):
            true_occ = np.sum(true_states == k) / len(true_states)
            ax1.axvline(true_occ, color=f'C{k}', linestyle='--', linewidth=2)
    ax1.set_xlabel('Occupancy Proportion')
    ax1.set_ylabel('Frequency')
    ax1.set_title('State Occupancy Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Time series of state probabilities
    ax2 = fig.add_subplot(gs[0, 1])
    for k in range(K):
        state_probs = np.mean(state_samples == k, axis=0)
        ax2.plot(state_probs, label=f'State {k}', linewidth=1.5, alpha=0.8)
    if true_states is not None:
        for k in range(K):
            mask = (true_states == k)
            ax2.fill_between(range(T), 0, 1, where=mask, alpha=0.2, color=f'C{k}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('P(S_t = k | data)')
    ax2.set_title('Posterior State Probabilities Over Time')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. State duration analysis
    ax3 = fig.add_subplot(gs[1, 0])
    for k in range(K):
        durations = []
        for sample_idx in range(n_save):
            states = state_samples[sample_idx, :]
            in_state = (states == k)
            changes = np.diff(np.concatenate([[False], in_state, [False]]).astype(int))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            durations.extend(ends - starts)

        if len(durations) > 0:
            ax3.hist(durations, bins=range(1, min(max(durations) + 1, 51)),
                     alpha=0.6, label=f'State {k}', edgecolor='black', density=True)

    ax3.set_xlabel('Duration (time steps)')
    ax3.set_ylabel('Density')
    ax3.set_title('State Duration Distribution')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_xlim(0, 50)

    # 4. Transition frequency matrix
    ax4 = fig.add_subplot(gs[1, 1])
    trans_counts = np.zeros((K, K))
    for sample_idx in range(n_save):
        states = state_samples[sample_idx, :]
        for t in range(len(states) - 1):
            if states[t] >= 0 and states[t + 1] >= 0:
                trans_counts[states[t], states[t + 1]] += 1

    trans_freq = trans_counts / (trans_counts.sum(axis=1, keepdims=True) + 1e-10)

    im = ax4.imshow(trans_freq, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax4.set_xlabel('To State')
    ax4.set_ylabel('From State')
    ax4.set_title('Empirical Transition Frequencies')
    ax4.set_xticks(range(K))
    ax4.set_yticks(range(K))

    for i in range(K):
        for j in range(K):
            ax4.text(j, i, f'{trans_freq[i, j]:.2f}',
                     ha="center", va="center", color="black", fontsize=10)

    plt.colorbar(im, ax=ax4)
    plt.tight_layout()
    return fig


def plot_data_with_states(y, results, true_states=None):
    """
    Plot the time series data colored by most likely state.
    """
    state_samples = results['state_samples']
    T = len(y)
    K = results['mu_samples'].shape[1]

    mode_states = np.zeros(T, dtype=int)
    for t in range(T):
        if t >= len(state_samples[0]):
            continue
        counts = np.bincount(state_samples[:, t], minlength=K)
        mode_states[t] = np.argmax(counts)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    ax = axes[0]
    for k in range(K):
        mask = (mode_states == k)
        ax.scatter(np.where(mask)[0], y[mask], s=2, alpha=0.6, label=f'State {k}')
    ax.set_xlabel('Time')
    ax.set_ylabel('y_t')
    ax.set_title('Data Colored by Posterior Mode State')
    ax.legend()
    ax.grid(alpha=0.3)

    if true_states is not None:
        ax = axes[1]
        for k in range(K):
            mask = (true_states == k)
            ax.scatter(np.where(mask)[0], y[mask], s=2, alpha=0.6, label=f'State {k}')
        ax.set_xlabel('Time')
        ax.set_ylabel('y_t')
        ax.set_title('Data Colored by True State')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        axes[1].axis('off')

    plt.tight_layout()
    return fig


def plot_regime_means(y, results, true_params=None):
    """
    Plot data with regime-specific mean lines.
    """
    mu_samples = results['mu_samples']
    state_samples = results['state_samples']
    T = len(y)
    K = mu_samples.shape[1]

    mu_post_mean = np.mean(mu_samples, axis=0)

    mode_states = np.zeros(T, dtype=int)
    for t in range(T):
        if t < state_samples.shape[1]:
            counts = np.bincount(state_samples[:, t], minlength=K)
            mode_states[t] = np.argmax(counts)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(y, color='black', alpha=0.3, linewidth=0.5, label='Data')

    for k in range(K):
        in_state = (mode_states == k)
        changes = np.diff(np.concatenate([[False], in_state, [False]]).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for start, end in zip(starts, ends):
            ax.hlines(mu_post_mean[k], start, end,
                      colors=f'C{k}', linewidth=2, alpha=0.8,
                      label=f'μ_{k}' if start == starts[0] else '')

            if true_params is not None and 'mu' in true_params:
                ax.hlines(true_params['mu'][k], start, end,
                          colors=f'C{k}', linewidth=2, linestyle='--', alpha=0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Data with Regime-Specific Means')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def create_all_diagnostic_plots(y, results, true_params=None, true_states=None, save_dir=None):
    """
    Generate all diagnostic plots and optionally save them.

    Args:
        y: observed data
        results: dict from gibbs_msar
        true_params: dict with 'kappa', 'mu', 'sigma2' (optional)
        true_states: true state sequence (optional)
        save_dir: directory to save plots (optional)

    Returns:
        dict of matplotlib figures
    """
    print("Generating diagnostic plots...")

    figs = {}

    print("  1. Trace plots...")
    figs['traces'] = plot_msar_traces(results, true_params)

    print("  2. Posterior histograms...")
    figs['histograms'] = plot_posterior_histograms(results, true_params)

    print("  3. State paths...")
    figs['state_paths'] = plot_state_paths(results, true_states, n_paths=5)

    print("  4. State occupancy...")
    figs['occupancy'] = plot_state_occupancy(results, true_states)

    print("  5. Data with states...")
    figs['data_states'] = plot_data_with_states(y, results, true_states)

    print("  6. Regime means...")
    figs['regime_means'] = plot_regime_means(y, results, true_params)

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            filepath = os.path.join(save_dir, f'msar_{name}.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")

    print("Done!")
    return figs


if __name__ == "__main__":
    print("This module provides plotting functions for MS-AR diagnostics.")
    print("Import and use with your MS-AR results:")
    print()
    print("  from msar_diagnostic_plots import create_all_diagnostic_plots")
    print("  figs = create_all_diagnostic_plots(y, results, true_params, true_states)")
    print("  plt.show()")

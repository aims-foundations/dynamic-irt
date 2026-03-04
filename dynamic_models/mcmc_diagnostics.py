"""GPIRT diagnostics and visualization.

Single-chain diagnostics (R-hat, ESS, trace plots, autocorrelation),
multi-chain diagnostics, and posterior visualization.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tueplots import bundles, figsizes

plt.rcParams.update(bundles.aaai2024())

# Standardized color palette (Paul Tol qualitative)
COLORS = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44"]


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

def _autocorr(x, max_lag=None):
    """Compute autocorrelation for a 1-D chain using FFT."""
    n = len(x)
    if max_lag is None:
        max_lag = n
    x = x - x.mean()
    # FFT-based autocorrelation
    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real
    acf /= acf[0] if acf[0] != 0 else 1.0
    return acf[:max_lag]


def effective_sample_size(chain):
    """Estimate ESS from a 1-D chain using initial monotone sequence estimator."""
    n = len(chain)
    acf = _autocorr(chain)
    # Sum pairs of autocorrelations until they become negative
    # (Geyer's initial positive sequence estimator)
    tau = 1.0
    for lag in range(1, n // 2):
        pair_sum = acf[2 * lag - 1] + acf[2 * lag] if 2 * lag < n else 0
        if pair_sum < 0:
            break
        tau += 2 * pair_sum
    return n / tau


def _rhat_from_chains(chains):
    """Compute R-hat from a list of 1-D chains (Gelman-Rubin diagnostic).

    Args:
        chains: list of 1-D numpy arrays (one per chain), all same length.

    Returns:
        R-hat statistic. < 1.01 is excellent, < 1.1 is acceptable.
    """
    m = len(chains)
    n = len(chains[0])
    chain_means = np.array([c.mean() for c in chains])
    chain_vars = np.array([c.var(ddof=1) for c in chains])

    # Within-chain variance
    W = chain_vars.mean()
    # Between-chain variance
    grand_mean = chain_means.mean()
    B = n * np.var(chain_means, ddof=1)
    # Marginal posterior variance estimate
    var_hat = ((n - 1) / n) * W + (1.0 / n) * B

    if W == 0:
        return float('nan')
    return np.sqrt(var_hat / W)


def split_rhat(chain):
    """Compute split-R-hat from a single chain (split in half)."""
    n = len(chain)
    mid = n // 2
    return _rhat_from_chains([chain[:mid], chain[mid:2 * mid]])


def multi_chain_rhat(chains):
    """Compute R-hat from multiple independent chains.

    Each chain is split in half, giving 2*M sub-chains for M chains.
    This is the recommended approach from BDA3 (Gelman et al.).
    """
    sub_chains = []
    for chain in chains:
        mid = len(chain) // 2
        sub_chains.append(chain[:mid])
        sub_chains.append(chain[mid:2 * mid])
    return _rhat_from_chains(sub_chains)


def compute_diagnostics(posterior_samples, param_names=None):
    """Compute R-hat, ESS, mean, std for each parameter column.

    Args:
        posterior_samples: (n_samples, n_params) numpy array
        param_names: optional list of names

    Returns:
        List of dicts with diagnostics per parameter.
    """
    n_samples, n_params = posterior_samples.shape
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]

    results = []
    for j in range(n_params):
        chain = posterior_samples[:, j]
        results.append({
            "name": param_names[j],
            "mean": chain.mean(),
            "std": chain.std(ddof=1),
            "rhat": split_rhat(chain),
            "ess": effective_sample_size(chain),
        })
    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_trace(chains_dict, result_dir, warmup=0):
    """Plot trace + posterior density for selected parameters.

    Args:
        chains_dict: dict mapping param_name -> 1-D numpy array (full chain incl. warmup)
        result_dir: output directory
        warmup: number of warmup samples to shade
    """
    n_params = len(chains_dict)
    fig, axes = plt.subplots(n_params, 2, figsize=(
        figsizes.aaai2024_full()["figure.figsize"][0],
        1.5 * n_params,
    ))
    if n_params == 1:
        axes = axes[np.newaxis, :]

    for i, (name, chain) in enumerate(chains_dict.items()):
        # Trace plot
        ax_trace = axes[i, 0]
        ax_trace.plot(chain, alpha=0.6, linewidth=0.5, color=COLORS[0])
        if warmup > 0:
            ax_trace.axvspan(0, warmup, alpha=0.15, color=COLORS[1], label="Warmup")
        ax_trace.set_ylabel(name)
        if i == 0:
            ax_trace.set_title("Trace")
        if i == n_params - 1:
            ax_trace.set_xlabel("Sample")

        # Posterior density (post-warmup only)
        ax_hist = axes[i, 1]
        posterior = chain[warmup:]
        ax_hist.hist(posterior, bins=40, density=True, alpha=0.3, color=COLORS[0])
        sns.kdeplot(posterior, color=COLORS[0], linewidth=1.5, ax=ax_hist)
        if i == 0:
            ax_hist.set_title("Posterior")

    fig.tight_layout()
    save_path = os.path.join(result_dir, "trace_plots.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_autocorr(chains_dict, result_dir, max_lag=100):
    """Plot autocorrelation functions for selected parameters."""
    n_params = len(chains_dict)
    fig, axes = plt.subplots(1, n_params, figsize=(
        figsizes.aaai2024_full()["figure.figsize"][0],
        figsizes.aaai2024_half()["figure.figsize"][1],
    ))
    if n_params == 1:
        axes = [axes]

    for i, (name, chain) in enumerate(chains_dict.items()):
        acf = _autocorr(chain, max_lag=max_lag)
        axes[i].bar(range(len(acf)), acf, width=1.0, alpha=0.5, color=COLORS[0])
        axes[i].axhline(0, color="black", linewidth=0.5)
        axes[i].set_xlabel("Lag")
        axes[i].set_title(name)
        if i == 0:
            axes[i].set_ylabel("ACF")

    fig.tight_layout()
    save_path = os.path.join(result_dir, "autocorrelation.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_param_hist(values, param_name, filename, result_dir, xlabel,
                    bins=30, pct_clip=(1, 99)):
    """Plot histogram + KDE, clipping x-axis to the given percentile range."""
    lo, hi = np.percentile(values, pct_clip)
    clipped = values[(values >= lo) & (values <= hi)]

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_half()["figure.figsize"])
    ax.hist(clipped, bins=bins, density=True, alpha=0.3, color=COLORS[0])
    sns.kdeplot(clipped, color=COLORS[0], linewidth=1.5, bw_adjust=0.5, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"{param_name} Distribution")
    save_path = os.path.join(result_dir, f"{filename}.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_trajectories(ability_means, segment_sizes, ability_prior_dists,
                      result_dir, n_top=5):
    """Plot posterior mean ability trajectories for top students by final ability."""
    # Compute final ability for each student
    final_abilities = []
    offset = 0
    for seg_size in segment_sizes:
        student_traj = ability_means[offset:offset + seg_size]
        final_abilities.append(student_traj[-1])
        offset += seg_size
    final_abilities = np.array(final_abilities)

    # Pick top students by final ability
    top_idx = np.argsort(final_abilities)[::-1][:n_top]

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
    offset = 0
    student_i = 0
    color_i = 0
    for seg_size in segment_sizes:
        if student_i in top_idx:
            traj = ability_means[offset:offset + seg_size]
            # Use the GP prior's time points as x-axis
            time_points = ability_prior_dists[student_i].loc.cpu().numpy()
            t_axis = np.arange(len(traj)) if len(time_points) != len(traj) else np.arange(len(traj))
            ax.plot(t_axis, traj, color=COLORS[color_i % len(COLORS)],
                    label=f"Student {student_i}")
            color_i += 1
        offset += seg_size
        student_i += 1

    ax.set_xlabel("Time index")
    ax.set_ylabel(r"$\theta(t)$")
    ax.set_title(r"Ability Trajectories: Top Students (Posterior Mean)")
    ax.legend()
    save_path = os.path.join(result_dir, "ability_trajectories.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_log_likelihood_chain(all_abilities, all_difficulties, model, result_dir):
    """Plot log-likelihood over the full MCMC chain (warmup + posterior)."""
    n_total = all_abilities.shape[0]
    llhs = []
    for i in range(n_total):
        with torch.no_grad():
            ll = model.log_likelihood(
                all_abilities[i].to(model.device),
                all_difficulties[i].to(model.device),
            )
        llhs.append(ll.item())

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
    ax.plot(llhs, alpha=0.7, color=COLORS[0])
    ax.set_xlabel("Sample")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("GPIRT ESS: Log-likelihood Chain")
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(result_dir, "log_likelihood_chain.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


# ---------------------------------------------------------------------------
# Single-chain diagnostics
# ---------------------------------------------------------------------------

def run_diagnostics(ability_chain, difficulty_chain, segment_sizes,
                    result_dir, warmup=100, gamma_chain=None):
    """Run full posterior diagnostics and save results.

    Args:
        ability_chain: (n_total, total_ability_dim) full chain including warmup
        difficulty_chain: (n_total, n_questions) full chain including warmup
        segment_sizes: list of per-student ability dimensions
        result_dir: output directory
        warmup: number of warmup samples
    """
    import pandas as pd

    posterior_ability = ability_chain[warmup:]    # (n_post, total_ability_dim)
    posterior_diff = difficulty_chain[warmup:]    # (n_post, n_questions)
    n_post = posterior_ability.shape[0]
    n_questions = posterior_diff.shape[1]
    n_students = len(segment_sizes)

    print(f"\n{'='*60}")
    print(f"MCMC Diagnostics ({n_post} posterior samples)")
    print(f"{'='*60}")

    # --- Difficulty diagnostics ---
    diff_names = [f"beta_{j}" for j in range(n_questions)]
    diff_diag = compute_diagnostics(posterior_diff, diff_names)

    rhats_diff = np.array([d["rhat"] for d in diff_diag])
    ess_diff = np.array([d["ess"] for d in diff_diag])

    print(f"\nDifficulty parameters (n={n_questions}):")
    print(f"  R-hat: median={np.nanmedian(rhats_diff):.4f}, "
          f"max={np.nanmax(rhats_diff):.4f}, "
          f"pct>1.1={100*np.nanmean(rhats_diff > 1.1):.1f}%")
    print(f"  ESS:   median={np.nanmedian(ess_diff):.1f}, "
          f"min={np.nanmin(ess_diff):.1f}, "
          f"pct<100={100*np.nanmean(ess_diff < 100):.1f}%")

    # --- Per-student ability diagnostics (posterior mean per student at each step) ---
    student_mean_chains = []
    offset = 0
    for s, seg_size in enumerate(segment_sizes):
        student_chain = posterior_ability[:, offset:offset + seg_size]
        student_mean_chains.append(student_chain.mean(axis=1))
        offset += seg_size
    student_mean_chains = np.stack(student_mean_chains, axis=1)

    ability_names = [f"theta_mean_{s}" for s in range(n_students)]
    ability_diag = compute_diagnostics(student_mean_chains, ability_names)

    rhats_ab = np.array([d["rhat"] for d in ability_diag])
    ess_ab = np.array([d["ess"] for d in ability_diag])

    print(f"\nAbility parameters (n={n_students} students, mean over time):")
    print(f"  R-hat: median={np.nanmedian(rhats_ab):.4f}, "
          f"max={np.nanmax(rhats_ab):.4f}, "
          f"pct>1.1={100*np.nanmean(rhats_ab > 1.1):.1f}%")
    print(f"  ESS:   median={np.nanmedian(ess_ab):.1f}, "
          f"min={np.nanmin(ess_ab):.1f}, "
          f"pct<100={100*np.nanmean(ess_ab < 100):.1f}%")

    # --- Testlet effect diagnostics ---
    gamma_diag = []
    if gamma_chain is not None:
        posterior_gamma = gamma_chain[warmup:]
        n_students_g, n_q_g = posterior_gamma.shape[1], posterior_gamma.shape[2]
        rng = np.random.default_rng(0)
        n_sample = min(10, n_students_g * n_q_g)
        s_idxs = rng.integers(0, n_students_g, size=n_sample)
        q_idxs = rng.integers(0, n_q_g, size=n_sample)
        gamma_sample = posterior_gamma[:, s_idxs, q_idxs]
        gamma_names = [f"gamma_s{s}_q{q}" for s, q in zip(s_idxs, q_idxs)]
        gamma_diag = compute_diagnostics(gamma_sample, gamma_names)

        rhats_g = np.array([d["rhat"] for d in gamma_diag])
        ess_g = np.array([d["ess"] for d in gamma_diag])
        print(f"\nTestlet effects γ (sample of {n_sample} s×q pairs):")
        print(f"  R-hat: median={np.nanmedian(rhats_g):.4f}, "
              f"max={np.nanmax(rhats_g):.4f}, "
              f"pct>1.1={100*np.nanmean(rhats_g > 1.1):.1f}%")
        print(f"  ESS:   median={np.nanmedian(ess_g):.1f}, "
              f"min={np.nanmin(ess_g):.1f}, "
              f"pct<50={100*np.nanmean(ess_g < 50):.1f}%")

    # --- Save summary CSV ---
    summary_rows = []
    for d in diff_diag:
        summary_rows.append({**d, "type": "difficulty"})
    for d in ability_diag:
        summary_rows.append({**d, "type": "ability_mean"})
    for d in gamma_diag:
        summary_rows.append({**d, "type": "testlet_effect"})
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(result_dir, "diagnostics_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Diagnostics CSV: {csv_path}")

    # --- Trace plots for select parameters ---
    trace_params = {}
    diff_means = posterior_diff.mean(axis=0)
    for tag, idx in [("easiest", np.argmin(diff_means)),
                     ("median", np.argsort(diff_means)[n_questions // 2]),
                     ("hardest", np.argmax(diff_means))]:
        trace_params[f"beta_{tag}"] = difficulty_chain[:, idx]

    ab_stds = student_mean_chains.std(axis=0)

    full_student_means = []
    offset = 0
    for seg_size in segment_sizes:
        student_chain = ability_chain[:, offset:offset + seg_size]
        full_student_means.append(student_chain.mean(axis=1))
        offset += seg_size
    full_student_means = np.stack(full_student_means, axis=1)

    trace_params[r"$\bar{\theta}$ (most variable)"] = full_student_means[:, np.argmax(ab_stds)]
    trace_params[r"$\bar{\theta}$ (least variable)"] = full_student_means[:, np.argmin(ab_stds)]

    plot_trace(trace_params, result_dir, warmup=warmup)

    # --- Autocorrelation plots (post-warmup only) ---
    acf_params = {}
    for tag, idx in [("easiest", np.argmin(diff_means)),
                     ("hardest", np.argmax(diff_means))]:
        acf_params[f"beta_{tag}"] = posterior_diff[:, idx]
    acf_params[r"$\bar{\theta}$ (most var)"] = student_mean_chains[:, np.argmax(ab_stds)]
    acf_params[r"$\bar{\theta}$ (least var)"] = student_mean_chains[:, np.argmin(ab_stds)]

    plot_autocorr(acf_params, result_dir, max_lag=min(100, n_post // 2))

    # --- R-hat distribution plot ---
    fig, axes = plt.subplots(1, 2, figsize=figsizes.aaai2024_full()["figure.figsize"])

    axes[0].hist(rhats_diff, bins=40, alpha=0.5, color=COLORS[0], density=True)
    axes[0].axvline(1.1, color=COLORS[1], linestyle="--", linewidth=1, label=r"$\hat{R}=1.1$")
    axes[0].set_xlabel(r"Split-$\hat{R}$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"$\beta$ (Difficulty)")
    axes[0].legend()

    axes[1].hist(rhats_ab, bins=40, alpha=0.5, color=COLORS[2], density=True)
    axes[1].axvline(1.1, color=COLORS[1], linestyle="--", linewidth=1, label=r"$\hat{R}=1.1$")
    axes[1].set_xlabel(r"Split-$\hat{R}$")
    axes[1].set_title(r"$\bar{\theta}$ (Mean Ability)")
    axes[1].legend()

    fig.tight_layout()
    save_path = os.path.join(result_dir, "rhat_distribution.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")

    # --- ESS distribution plot ---
    fig, axes = plt.subplots(1, 2, figsize=figsizes.aaai2024_full()["figure.figsize"])

    axes[0].hist(ess_diff, bins=40, alpha=0.5, color=COLORS[0], density=True)
    axes[0].axvline(100, color=COLORS[1], linestyle="--", linewidth=1, label="ESS=100")
    axes[0].set_xlabel("ESS")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"$\beta$ (Difficulty)")
    axes[0].legend()

    axes[1].hist(ess_ab, bins=40, alpha=0.5, color=COLORS[2], density=True)
    axes[1].axvline(100, color=COLORS[1], linestyle="--", linewidth=1, label="ESS=100")
    axes[1].set_xlabel("ESS")
    axes[1].set_title(r"$\bar{\theta}$ (Mean Ability)")
    axes[1].legend()

    fig.tight_layout()
    save_path = os.path.join(result_dir, "ess_distribution.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Multi-chain diagnostics
# ---------------------------------------------------------------------------

def run_multichain_diagnostics(chain_folders, segment_sizes, result_dir, warmup=200):
    """Run multi-chain diagnostics from saved chain files.

    Args:
        chain_folders: list of result folder paths, each containing
            ability_chain.pt and difficulty_chain.pt
        segment_sizes: list of per-student ability dimensions
        result_dir: output directory for plots and CSVs
        warmup: number of warmup samples per chain
    """
    import pandas as pd

    n_chains = len(chain_folders)
    print(f"\n{'='*60}")
    print(f"Multi-Chain Diagnostics ({n_chains} chains)")
    print(f"{'='*60}")

    # Load all chains
    diff_chains = []
    ability_mean_chains = []

    for folder in chain_folders:
        diff_chain = torch.load(f"{folder}/difficulty_chain.pt", map_location="cpu")
        ab_chain = torch.load(f"{folder}/ability_chain.pt", map_location="cpu")

        diff_post = diff_chain[warmup:].numpy()
        ab_post = ab_chain[warmup:].numpy()

        diff_chains.append(diff_post)

        student_means = []
        offset = 0
        for seg_size in segment_sizes:
            student_means.append(ab_post[:, offset:offset + seg_size].mean(axis=1))
            offset += seg_size
        ability_mean_chains.append(np.stack(student_means, axis=1))

        print(f"  {folder}: {diff_post.shape[0]} posterior samples")

    n_post = diff_chains[0].shape[0]
    n_questions = diff_chains[0].shape[1]
    n_students = ability_mean_chains[0].shape[1]

    # --- Multi-chain R-hat and ESS ---
    print(f"\nComputing multi-chain R-hat and ESS...")

    rhats_diff = np.zeros(n_questions)
    ess_diff = np.zeros(n_questions)
    for j in range(n_questions):
        param_chains = [dc[:, j] for dc in diff_chains]
        rhats_diff[j] = multi_chain_rhat(param_chains)
        ess_diff[j] = sum(effective_sample_size(c) for c in param_chains)

    print(f"\nDifficulty parameters (n={n_questions}):")
    print(f"  R-hat: median={np.nanmedian(rhats_diff):.4f}, "
          f"max={np.nanmax(rhats_diff):.4f}, "
          f"pct>1.1={100*np.nanmean(rhats_diff > 1.1):.1f}%")
    print(f"  ESS:   median={np.nanmedian(ess_diff):.1f}, "
          f"min={np.nanmin(ess_diff):.1f}, "
          f"pct<100={100*np.nanmean(ess_diff < 100):.1f}%")

    rhats_ab = np.zeros(n_students)
    ess_ab = np.zeros(n_students)
    for s in range(n_students):
        param_chains = [ac[:, s] for ac in ability_mean_chains]
        rhats_ab[s] = multi_chain_rhat(param_chains)
        ess_ab[s] = sum(effective_sample_size(c) for c in param_chains)

    print(f"\nAbility parameters (n={n_students} students, mean over time):")
    print(f"  R-hat: median={np.nanmedian(rhats_ab):.4f}, "
          f"max={np.nanmax(rhats_ab):.4f}, "
          f"pct>1.1={100*np.nanmean(rhats_ab > 1.1):.1f}%")
    print(f"  ESS:   median={np.nanmedian(ess_ab):.1f}, "
          f"min={np.nanmin(ess_ab):.1f}, "
          f"pct<100={100*np.nanmean(ess_ab < 100):.1f}%")

    # --- Save summary CSV ---
    summary_rows = []
    for j in range(n_questions):
        summary_rows.append({
            "name": f"beta_{j}", "type": "difficulty",
            "rhat": rhats_diff[j], "ess": ess_diff[j],
            "mean": np.mean([dc[:, j].mean() for dc in diff_chains]),
            "std": np.mean([dc[:, j].std() for dc in diff_chains]),
        })
    for s in range(n_students):
        summary_rows.append({
            "name": f"theta_mean_{s}", "type": "ability_mean",
            "rhat": rhats_ab[s], "ess": ess_ab[s],
            "mean": np.mean([ac[:, s].mean() for ac in ability_mean_chains]),
            "std": np.mean([ac[:, s].std() for ac in ability_mean_chains]),
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(result_dir, "multichain_diagnostics.csv"), index=False)
    print(f"\n  CSV: {os.path.join(result_dir, 'multichain_diagnostics.csv')}")

    # --- Trace overlay plot: show all chains for select parameters ---
    diff_means_overall = np.mean([dc.mean(axis=0) for dc in diff_chains], axis=0)
    representative_diff = {
        "easiest": np.argmin(diff_means_overall),
        "median": np.argsort(diff_means_overall)[n_questions // 2],
        "hardest": np.argmax(diff_means_overall),
    }

    ab_stds_overall = np.mean([ac.std(axis=0) for ac in ability_mean_chains], axis=0)
    representative_ab = {
        "most_var": np.argmax(ab_stds_overall),
        "least_var": np.argmin(ab_stds_overall),
    }

    n_trace = len(representative_diff) + len(representative_ab)
    fig, axes = plt.subplots(n_trace, 2, figsize=(
        figsizes.aaai2024_full()["figure.figsize"][0],
        1.5 * n_trace,
    ))

    row = 0
    for tag, idx in representative_diff.items():
        ax_trace, ax_hist = axes[row, 0], axes[row, 1]
        for c_i, dc in enumerate(diff_chains):
            full_chain = torch.load(
                f"{chain_folders[c_i]}/difficulty_chain.pt", map_location="cpu"
            ).numpy()[:, idx]
            ax_trace.plot(full_chain, alpha=0.5, linewidth=0.4,
                          color=COLORS[c_i % len(COLORS)],
                          label=f"Chain {c_i+1}")
            ax_hist.hist(dc[:, idx], bins=30, density=True, alpha=0.2,
                         color=COLORS[c_i % len(COLORS)])
        ax_trace.axvspan(0, warmup, alpha=0.1, color="gray")
        ax_trace.set_ylabel(f"beta_{tag}")
        ax_hist.set_title(f"R-hat={rhats_diff[idx]:.3f}" if row == 0 else
                          f"R-hat={rhats_diff[idx]:.3f}")
        if row == 0:
            ax_trace.set_title("Trace (all chains)")
            ax_trace.legend(fontsize=5)
        row += 1

    for tag, idx in representative_ab.items():
        ax_trace, ax_hist = axes[row, 0], axes[row, 1]
        for c_i, ac in enumerate(ability_mean_chains):
            ab_chain = torch.load(
                f"{chain_folders[c_i]}/ability_chain.pt", map_location="cpu"
            ).numpy()
            offset = sum(segment_sizes[:idx])
            full_student = ab_chain[:, offset:offset + segment_sizes[idx]].mean(axis=1)
            ax_trace.plot(full_student, alpha=0.5, linewidth=0.4,
                          color=COLORS[c_i % len(COLORS)])
            ax_hist.hist(ac[:, idx], bins=30, density=True, alpha=0.2,
                         color=COLORS[c_i % len(COLORS)])
        ax_trace.axvspan(0, warmup, alpha=0.1, color="gray")
        ax_trace.set_ylabel(f"theta_{tag}")
        ax_hist.set_title(f"R-hat={rhats_ab[idx]:.3f}")
        row += 1

    axes[-1, 0].set_xlabel("Sample")
    fig.tight_layout()
    save_path = os.path.join(result_dir, "multichain_traces.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")

    # --- R-hat distribution plot ---
    fig, axes = plt.subplots(1, 2, figsize=figsizes.aaai2024_full()["figure.figsize"])

    axes[0].hist(rhats_diff, bins=50, alpha=0.5, color=COLORS[0], density=True)
    axes[0].axvline(1.1, color=COLORS[1], linestyle="--", linewidth=1, label=r"$\hat{R}=1.1$")
    axes[0].set_xlabel(r"$\hat{R}$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"$\beta$ (Difficulty)")
    axes[0].legend()

    axes[1].hist(rhats_ab, bins=50, alpha=0.5, color=COLORS[2], density=True)
    axes[1].axvline(1.1, color=COLORS[1], linestyle="--", linewidth=1, label=r"$\hat{R}=1.1$")
    axes[1].set_xlabel(r"$\hat{R}$")
    axes[1].set_title(r"$\bar{\theta}$ (Mean Ability)")
    axes[1].legend()

    fig.tight_layout()
    save_path = os.path.join(result_dir, "multichain_rhat.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")

    # --- ESS distribution plot ---
    fig, axes = plt.subplots(1, 2, figsize=figsizes.aaai2024_full()["figure.figsize"])

    axes[0].hist(ess_diff, bins=50, alpha=0.5, color=COLORS[0], density=True)
    axes[0].axvline(100, color=COLORS[1], linestyle="--", linewidth=1, label="ESS=100")
    axes[0].set_xlabel("ESS (total across chains)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"$\beta$ (Difficulty)")
    axes[0].legend()

    axes[1].hist(ess_ab, bins=50, alpha=0.5, color=COLORS[2], density=True)
    axes[1].axvline(100, color=COLORS[1], linestyle="--", linewidth=1, label="ESS=100")
    axes[1].set_xlabel("ESS (total across chains)")
    axes[1].set_title(r"$\bar{\theta}$ (Mean Ability)")
    axes[1].legend()

    fig.tight_layout()
    save_path = os.path.join(result_dir, "multichain_ess.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")

    print(f"{'='*60}\n")

    return rhats_diff, rhats_ab, ess_diff, ess_ab


# ---------------------------------------------------------------------------
# Top-level visualization entry point
# ---------------------------------------------------------------------------

def visualize(ability_samples, difficulty_samples, all_abilities, all_difficulties,
              model, result_dir):
    """Generate all GPIRT plots."""
    # Posterior mean difficulty
    diff_mean = difficulty_samples.mean(dim=0).numpy()
    plot_param_hist(diff_mean, r"$\beta$", "difficulty", result_dir,
                    r"$\beta$ (Item Difficulty)", bins=80)

    # Posterior mean ability (concatenated across students)
    ability_mean = ability_samples.mean(dim=0).numpy()
    plot_param_hist(ability_mean, r"$\theta$", "ability", result_dir,
                    r"$\theta$ (Ability)", bins=80, pct_clip=(5, 95))

    # Trajectories
    plot_trajectories(ability_mean, model.segment_sizes,
                      model.ability_prior_dists, result_dir)

    # Log-likelihood chain
    plot_log_likelihood_chain(all_abilities, all_difficulties, model, result_dir)

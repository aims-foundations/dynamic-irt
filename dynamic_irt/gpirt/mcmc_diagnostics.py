"""
Multi-chain MCMC diagnostics for GPIRT-Testlet blocked ESS.

Computes:
  1. R-hat (Gelman-Rubin) convergence diagnostic
  2. Effective Sample Size (ESS) via FFT autocorrelation
  3. Within-chain drift (Q1 vs Q4 posterior mean shift)
  4. Inter-chain posterior mean correlation
  5. Posterior summary statistics

Usage:
    python mcmc_diagnostics.py --seeds 54 55 56 57 --warmup 500
    python mcmc_diagnostics.py --seeds 54 55 56 57 --warmup 500 --course dsa_hk231
"""

import argparse
import os
from itertools import combinations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Core diagnostics
# ---------------------------------------------------------------------------

def compute_rhat(chain_samples_list):
    """Gelman-Rubin R-hat across M chains.

    Args:
        chain_samples_list: list of (n_samples, n_params) arrays.

    Returns:
        (n_params,) array of R-hat values.
    """
    M = len(chain_samples_list)
    N = min(c.shape[0] for c in chain_samples_list)
    half = N // 2
    if half < 10:
        return None
    trimmed = [c[half:N] for c in chain_samples_list]
    n = trimmed[0].shape[0]
    chain_means = np.array([c.mean(axis=0) for c in trimmed])
    B = n * np.var(chain_means, axis=0, ddof=1)
    W = np.mean([np.var(c, axis=0, ddof=1) for c in trimmed], axis=0)
    var_hat = (1 - 1.0 / n) * W + (1.0 / n) * B
    rhat = np.sqrt(var_hat / np.maximum(W, 1e-10))
    return rhat


def ess_1d(x):
    """ESS for a single 1-D chain using Geyer's initial monotone estimator."""
    n = len(x)
    x = x - x.mean()
    fft = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fft * np.conj(fft))[:n].real
    acf /= acf[0]
    tau = 1.0
    for i in range(0, n - 1, 2):
        pair_sum = acf[i] + (acf[i + 1] if i + 1 < n else 0.0)
        if pair_sum < 0:
            break
        tau += 2 * pair_sum
    return max(1.0, n / tau)


def compute_ess(chain_samples_list, n_subsample=2000):
    """Compute ESS across chains, subsampling params for speed.

    Returns:
        (n_subsampled * n_chains,) array of per-parameter-per-chain ESS values.
    """
    n_params = chain_samples_list[0].shape[1]
    np.random.seed(42)
    idx = np.random.choice(n_params, min(n_subsample, n_params), replace=False)
    all_ess = []
    for c in chain_samples_list:
        for j in idx:
            all_ess.append(ess_1d(c[:, j]))
    return np.array(all_ess)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def report_rhat(name, rhat):
    print(f"\n  === {name} R-hat (n_params={rhat.shape[0]}) ===")
    print(f"    mean={rhat.mean():.4f}, median={np.median(rhat):.4f}, "
          f"max={rhat.max():.4f}")
    print(f"    % < 1.01: {(rhat < 1.01).mean()*100:.1f}%")
    print(f"    % < 1.05: {(rhat < 1.05).mean()*100:.1f}%")
    print(f"    % < 1.1:  {(rhat < 1.1).mean()*100:.1f}%")
    print(f"    % > 1.1:  {(rhat > 1.1).mean()*100:.1f}%")


def report_ess(name, ess_vals, n_post):
    print(f"\n  === {name} ESS (post-warmup N={n_post} per chain) ===")
    print(f"    mean={ess_vals.mean():.0f}, median={np.median(ess_vals):.0f}, "
          f"min={ess_vals.min():.0f}, max={ess_vals.max():.0f}")
    print(f"    ESS/N ratio: mean={ess_vals.mean()/n_post:.3f}, "
          f"min={ess_vals.min()/n_post:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MCMC diagnostics")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="Chain seeds to analyse")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--course", type=str, default="dsa_hk231")
    parser.add_argument("--method", type=str,
                        default="blocked_ess_testlet_kernelRBF_ls1.0")
    parser.add_argument("--ess-subsample", type=int, default=2000,
                        help="Number of params to subsample for ESS")
    args = parser.parse_args()

    seeds = args.seeds
    warmup = args.warmup
    base = "results"

    # ------------------------------------------------------------------
    # Load chains
    # ------------------------------------------------------------------
    print("=" * 60)
    print("MCMC DIAGNOSTICS")
    print("=" * 60)

    chains = {}
    for seed in seeds:
        path = os.path.join(
            base, f"{args.course}_s{seed}_{args.method}", "checkpoint.pt"
        )
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            chains[seed] = ckpt
            it = ckpt["iteration"]
            post = max(0, it - warmup)
            print(f"  seed={seed}: iter={it}, post-warmup={post}")
        except Exception as e:
            print(f"  seed={seed}: SKIP ({e})")

    loaded = sorted(chains.keys())
    if len(loaded) < 2:
        print("Need at least 2 chains for diagnostics.")
        return

    n_post = min(chains[s]["difficulty_chain"][warmup:].shape[0]
                 for s in loaded)
    print(f"\nUsing {len(loaded)} chains, {n_post} post-warmup samples each")

    # ------------------------------------------------------------------
    # 1. R-hat
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. R-HAT (Gelman-Rubin)")
    print("=" * 60)

    diff_chains = [chains[s]["difficulty_chain"][warmup:].numpy()
                   for s in loaded]
    rhat_diff = compute_rhat(diff_chains)
    report_rhat("Difficulty", rhat_diff)

    is_testlet = "gamma_chain" in chains[loaded[0]]
    if is_testlet:
        gamma_chains = [
            chains[s]["gamma_chain"][warmup:]
            .reshape(chains[s]["gamma_chain"][warmup:].shape[0], -1)
            .numpy()
            for s in loaded
        ]
        rhat_gamma = compute_rhat(gamma_chains)
        report_rhat("Gamma", rhat_gamma)

    ab_chains = [chains[s]["ability_chain"][warmup:].numpy()
                 for s in loaded]
    n_ab = ab_chains[0].shape[1]
    np.random.seed(0)
    idx = np.random.choice(n_ab, min(5000, n_ab), replace=False)
    ab_sub = [c[:, idx] for c in ab_chains]
    rhat_ab = compute_rhat(ab_sub)
    report_rhat("Ability (5000 subsampled)", rhat_ab)

    # ------------------------------------------------------------------
    # 2. Effective Sample Size
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. EFFECTIVE SAMPLE SIZE")
    print("=" * 60)

    ess_diff = compute_ess(diff_chains, n_subsample=min(2183, args.ess_subsample))
    report_ess("Difficulty", ess_diff, n_post)

    if is_testlet:
        ess_gamma = compute_ess(gamma_chains, n_subsample=args.ess_subsample)
        report_ess("Gamma", ess_gamma, n_post)

    ess_ab = compute_ess(ab_sub, n_subsample=args.ess_subsample)
    report_ess("Ability", ess_ab, n_post)

    # ------------------------------------------------------------------
    # 3. Within-chain drift (Q1 vs Q4 of post-warmup)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. WITHIN-CHAIN DRIFT (Q1 vs Q4 mean shift)")
    print("=" * 60)

    for seed in loaded:
        dc = chains[seed]["difficulty_chain"][warmup:].numpy()
        q1 = dc.shape[0] // 4
        diff_drift = np.abs(dc[-q1:].mean(0) - dc[:q1].mean(0)).mean()
        msg = f"  seed={seed}: diff_drift={diff_drift:.4f}"
        if is_testlet:
            gc = chains[seed]["gamma_chain"][warmup:].numpy()
            gamma_drift = np.abs(gc[-q1:].mean(0) - gc[:q1].mean(0)).mean()
            msg += f", gamma_drift={gamma_drift:.4f}"
        print(msg)

    # ------------------------------------------------------------------
    # 4. Inter-chain posterior mean correlation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. INTER-CHAIN POSTERIOR MEAN CORRELATION")
    print("=" * 60)

    diff_means = np.array([
        chains[s]["difficulty_chain"][warmup:].numpy().mean(0)
        for s in loaded
    ])
    if is_testlet:
        gamma_means = np.array([
            chains[s]["gamma_chain"][warmup:]
            .reshape(-1, chains[s]["gamma_chain"].shape[1]
                     * chains[s]["gamma_chain"].shape[2])
            .numpy().mean(0)
            for s in loaded
        ])

    for i, j in combinations(range(len(loaded)), 2):
        r_diff = np.corrcoef(diff_means[i], diff_means[j])[0, 1]
        msg = f"  chains {loaded[i]}-{loaded[j]}: diff_corr={r_diff:.4f}"
        if is_testlet:
            r_gamma = np.corrcoef(gamma_means[i], gamma_means[j])[0, 1]
            msg += f", gamma_corr={r_gamma:.4f}"
        print(msg)

    # ------------------------------------------------------------------
    # 5. Posterior summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. POSTERIOR SUMMARY")
    print("=" * 60)

    all_diff = np.concatenate([
        chains[s]["difficulty_chain"][warmup:].numpy() for s in loaded
    ])
    post_mean_diff = all_diff.mean(0)
    print(f"  Difficulty: mean={post_mean_diff.mean():.3f}, "
          f"std(across items)={post_mean_diff.std():.3f}, "
          f"range=[{post_mean_diff.min():.2f}, {post_mean_diff.max():.2f}]")

    if is_testlet:
        all_gamma = np.concatenate([
            chains[s]["gamma_chain"][warmup:].numpy() for s in loaded
        ])
        gamma_rms = np.sqrt((all_gamma ** 2).mean())
        gamma_std_per_q = all_gamma.std(axis=(0, 1))
        print(f"  Gamma: rms={gamma_rms:.3f}, "
              f"per-question std range="
              f"[{gamma_std_per_q.min():.3f}, {gamma_std_per_q.max():.3f}]")

        for seed in loaded:
            g = chains[seed]["gamma"]
            print(f"    seed={seed}: gamma_rms="
                  f"{g.pow(2).mean().sqrt().item():.4f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()

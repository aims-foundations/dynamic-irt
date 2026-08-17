"""Summarize LLM predictor runs as mean balanced accuracy bar panels.

Reads JSONL files from the base run and ablation subdirectories, computes
mean balanced accuracy across attempts (with bootstrap CIs) per run, writes
ablation_summary.csv, and plots two bar panels in one figure: model
comparison (top) and Qwen prompt ablations (bottom).

Usage:
    python data_analysis/plot_llm_ablations.py
    python data_analysis/plot_llm_ablations.py --course dsa_hk231
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tueplots.bundles

from data_analysis.llm_eval_common import compute_llm_balanced_accuracy

matplotlib.use("Agg")

# Colors deliberately avoid palettes used in other paper figures
# (e.g. plot_filtered_accuracy.py assigns blue/red/green/purple to IRT/CIRT/BKT/DKT).
ABLATION_CONFIGS = {
    "Opus 4.6": {
        "jsonl": "claude_attempts10.jsonl",
        "color": "#332288",
    },
    "Qwen3-14B": {
        "jsonl": "qwen_server_attempts10.jsonl",
        "color": "#009988",
    },
    "Gemma-4-31B": {
        "jsonl": "gemma_server_attempts10.jsonl",
        "color": "#8c510a",
    },
}

ABLATION_ONLY_CONFIGS = {
    "Qwen: No persona": {
        "jsonl": "ablation_no_persona/qwen_server_attempts10.jsonl",
        "color": "#aa4499",
    },
    "Qwen: No trajectory": {
        "jsonl": "ablation_no_trajectory/qwen_server_attempts10.jsonl",
        "color": "#999933",
    },
    "Qwen: No RAG": {
        "jsonl": "ablation_no_rag/qwen_server_attempts10.jsonl",
        "color": "#ddaa33",
    },
    "Qwen: Recent questions": {
        "jsonl": "ablation_recent_questions/qwen_server_attempts10.jsonl",
        "color": "#666666",
    },
}


def compute_balanced_accuracy(jsonl_path, course, max_attempts=10):
    if not os.path.exists(jsonl_path):
        return None, None, None, None, None
    return compute_llm_balanced_accuracy(jsonl_path, course, max_attempts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--course", default="dsa_hk231")
    parser.add_argument("--max_attempts", type=int, default=10)
    args = parser.parse_args()

    base_dir = f"results/llm_student_eval/{args.course}"

    def _summarize(label, cfg):
        """Mean balanced accuracy across attempts with a bootstrap CI on that mean."""
        jsonl_path = os.path.join(base_dir, cfg["jsonl"])
        if not os.path.exists(jsonl_path):
            raise SystemExit(
                f"missing JSONL for run '{label}': {jsonl_path}; "
                "refusing to build a partial figure")

        accs, ci_los, ci_his, counts, boots = compute_balanced_accuracy(
            jsonl_path, args.course, args.max_attempts,
        )

        mean_acc = np.nanmean(accs)
        boot_stack = [b for b in boots if len(b) > 0]
        if boot_stack:
            n_rep = min(len(b) for b in boot_stack)
            mean_boots = np.mean([b[:n_rep] for b in boot_stack], axis=0)
            ci_lo, ci_hi = np.quantile(mean_boots, [0.025, 0.975])
        else:
            ci_lo = ci_hi = np.nan
        print(f"  {label}: mean balanced acc = {mean_acc:.4f} "
              f"[{ci_lo:.4f}, {ci_hi:.4f}], per-attempt = {accs}")
        return mean_acc, ci_lo, ci_hi

    all_configs = {**ABLATION_CONFIGS, **ABLATION_ONLY_CONFIGS}
    summaries = {label: _summarize(label, cfg) for label, cfg in all_configs.items()}

    csv_path = os.path.join(base_dir, "ablation_summary.csv")
    with open(csv_path, "w") as f:
        f.write("label,mean_balanced_acc,ci_lo,ci_hi\n")
        for label, (mean_acc, ci_lo, ci_hi) in summaries.items():
            f.write(f"{label},{mean_acc:.4f},{ci_lo:.4f},{ci_hi:.4f}\n")
    print(f"Summary saved: {csv_path}")

    with plt.rc_context({**tueplots.bundles.icml2022(), "text.usetex": True}):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 4.4))

        def _bar_on_ax(ax, entries, title):
            labels = [e[0] for e in entries]
            means = [e[1][0] for e in entries]
            err_lo = [e[1][0] - e[1][1] for e in entries]
            err_hi = [e[1][2] - e[1][0] for e in entries]
            colors = [e[2] for e in entries]

            pos = np.arange(len(labels))
            ax.bar(pos, means, yerr=[err_lo, err_hi], color=colors,
                   width=0.6, capsize=3, error_kw={"linewidth": 0.8})
            ax.set_xticks(pos)
            if len(labels) > 4:
                ax.set_xticklabels(labels, rotation=15, ha="right")
            else:
                ax.set_xticklabels(labels)
            ax.set_ylabel("Balanced Accuracy")
            ax.set_title(title)
            tops = [m + e for m, e in zip(means, err_hi) if not np.isnan(m + e)]
            if not tops:
                raise SystemExit(f"no plottable values for panel '{title}'")
            ax.set_ylim(0.5, max(tops) + 0.02)
            ax.grid(True, axis="y", alpha=0.2)

        # Top: model comparison
        model_entries = [(label, summaries[label], ABLATION_CONFIGS[label]["color"])
                         for label in ABLATION_CONFIGS if label in summaries]
        _bar_on_ax(ax1, model_entries, "Model Comparison")

        # Bottom: Qwen ablations ("Full" reuses the Qwen3-14B summary)
        qwen_panel = [
            ("Full", "Qwen3-14B"),
            ("No persona", "Qwen: No persona"),
            ("No trajectory", "Qwen: No trajectory"),
            ("No RAG", "Qwen: No RAG"),
            ("Recent questions", "Qwen: Recent questions"),
        ]
        ablation_entries = [(display, summaries[key], all_configs[key]["color"])
                            for display, key in qwen_panel if key in summaries]
        _bar_on_ax(ax2, ablation_entries, "Prompt Ablation (Qwen3-14B)")

        fig.tight_layout()
        out_path = os.path.join(base_dir, "llm_combined.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()

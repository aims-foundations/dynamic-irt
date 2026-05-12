"""RSSM performance across courses and years: AUC and LL vs horizon."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tueplots import bundles

plt.rcParams.update(bundles.neurips2024())
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

COURSE_GROUPS = {
    "DSA": [("dsa_hk221", "2022"), ("dsa_hk231", "2023")],
    "PF": [("pf_hk222", "2022"), ("pf_hk232", "2023")],
}
METRICS = ["auc", "log_likelihood"]
METRIC_LABELS = {"auc": r"AUC $\uparrow$", "log_likelihood": r"Log-Likelihood $\uparrow$"}
YEAR_STYLES = {"2022": ("#4477aa", "o", "-"), "2023": ("#ee6677", "s", "--")}


def main():
    df = pd.read_csv("results/temporal_eval_full/temporal_eval_all.csv")
    df = df[df["model"] == "RSSM"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))

    for col, (group_name, courses) in enumerate(COURSE_GROUPS.items()):
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            for course_id, year in courses:
                sub = df[(df["course"] == course_id) & (df["metric"] == metric)]
                sub = sub.sort_values("horizon")
                color, marker, ls = YEAR_STYLES[year]
                ax.plot(sub["horizon"], sub["value"], color=color, marker=marker,
                        linestyle=ls, linewidth=1.5, markersize=4,
                        label=f"{year}")

            ax.set_xlabel("Train Cutoff Week")
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric])
            ax.set_xticks(sorted(df[df["course"] == courses[0][0]]["horizon"].unique()))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            if row == 0:
                ax.set_title(group_name)

    fig.tight_layout()
    out = "overleaf/figures/rssm_per_course_metrics.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

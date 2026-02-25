"""
Testlet Response Theory (TRT) analysis — Python port of testlet.R.

Pipeline:
  1) Load + preprocess (from HuggingFace or local CSV)
  2) Train/test split
  3) Fit TRT model (random intercepts: student, item, testlet)
  4) Evaluate (ROC-AUC) + Person fit (infit/outfit)
  5) Extract random effects and generate plots

Two inference backends:
  - "mle"     : Penalised MLE (MAP) via PyTorch Adam — fast, GPU-friendly
  - "bayesian": Full posterior via Pyro NUTS — GPU-accelerated MCMC

Usage:
    cd CodeInsights && python -m dynamic_irt.testlet               # default: MLE
    cd CodeInsights && python -m dynamic_irt.testlet --method bayesian
    cd CodeInsights && python -m dynamic_irt.testlet --device cuda  # GPU
"""

import argparse
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tueplots import bundles, figsizes

matplotlib.use("Agg")
plt.rcParams.update(bundles.aaai2024())

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from dynamic_irt.gpirt.utils import ensure_dir, set_seed

# Standardized color palette (Paul Tol qualitative) — matches CIRT / Elo
COLORS = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44"]

# ---- Default params ----
RANDOM_SEED = 123
TEST_FRACTION = 0.20
N_TESTLET_GROUPS = 3
N_STUDENT_GROUPS = 10
USE_ALL_TESTLETS = False
USE_ALL_STUDENTS = False

# MLE defaults
MLE_LR = 0.01
MLE_EPOCHS = 500

# Bayesian defaults (Pyro NUTS)
BAYES_WARMUP = 1000
BAYES_SAMPLES = 1000
BAYES_CHAINS = 1  # Pyro NUTS runs 1 chain by default; increase for diagnostics
TARGET_ACCEPT = 0.85


# ---------------------------------------------------------------------------
# Helper functions  (data preprocessing — identical to testlet.R)
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load and preprocess CodeInsights data from HuggingFace.

    Loads main_data.csv + question_infos.csv, joins on question_id to get
    topic (= Testlet_ID), computes binary ItemScore and time ordering.

    Returns DataFrame with columns:
        StudentID_SF, ItemID_SF, Testlet_ID, T, ItemScore
    """
    from huggingface_hub import login, snapshot_download

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    path = snapshot_download(
        repo_id="stair-lab/code_insights_csv", repo_type="dataset"
    )
    main_data = pd.read_csv(f"{path}/main_data.csv", low_memory=False)
    question_infos = pd.read_csv(f"{path}/question_infos.csv")

    # Filter to actual submissions
    main_data = main_data[
        main_data["response_type"].isin(["Submit", "Prechecked"])
    ].copy()
    main_data["pass"] = main_data["pass"].astype(str).replace("nan", np.nan)
    main_data = main_data.dropna(subset=["pass"])

    # Compute time ordering per student
    main_data["timestamp"] = pd.to_datetime(
        main_data["timestamp"], format="%d/%m/%y, %H:%M:%S"
    )
    main_data["T"] = main_data.groupby("student_id")["timestamp"].transform(
        lambda x: (x - x.min()).dt.total_seconds()
    )

    # Binary ItemScore: 1 if all test cases pass, else 0
    main_data["ItemScore"] = main_data["pass"].apply(
        lambda s: 1 if all(c == "1" for c in str(s)) else 0
    )

    # Join with question_infos to get topic (= testlet)
    testlet_map = question_infos[["question_id", "topic"]].drop_duplicates()
    main_data = main_data.merge(
        testlet_map,
        left_on="question_unittest_id",
        right_on="question_id",
        how="left",
    )

    data = main_data[
        ["student_id", "question_unittest_id", "topic", "T", "ItemScore"]
    ].rename(columns={
        "student_id": "StudentID_SF",
        "question_unittest_id": "ItemID_SF",
        "topic": "Testlet_ID",
    })
    data = data.dropna(subset=["Testlet_ID"])
    data["ItemID_SF"] = data["ItemID_SF"].astype(str)
    data["Testlet_ID"] = data["Testlet_ID"].astype(str)

    return data.reset_index(drop=True)


def keep_latest_attempt(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the latest attempt per (student, item)."""
    idx = df.groupby(["StudentID_SF", "ItemID_SF"])["T"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def drop_no_variation_items(df_latest: pd.DataFrame) -> list:
    """Return item IDs with >= 2 unique response values."""
    variation = df_latest.groupby("ItemID_SF")["ItemScore"].nunique()
    return variation[variation >= 2].index.tolist()


def subset_by_testlet_student(
    df: pd.DataFrame,
    use_all_testlets: bool = False,
    use_all_students: bool = False,
    n_testlet_groups: int = 3,
    n_student_groups: int = 10,
) -> pd.DataFrame:
    """Optionally subset data to a fraction of testlets / students for speed."""
    out = df.copy()

    if not use_all_testlets:
        testlet_ids = out["Testlet_ID"].unique()
        np.random.shuffle(testlet_ids)
        splits = np.array_split(testlet_ids, n_testlet_groups)
        out = out[out["Testlet_ID"].isin(splits[0])]

    if not use_all_students:
        student_ids = out["StudentID_SF"].unique()
        np.random.shuffle(student_ids)
        splits = np.array_split(student_ids, n_student_groups)
        out = out[out["StudentID_SF"].isin(splits[0])]

    return out.reset_index(drop=True)


def first_attempt_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first attempt per (student, item)."""
    idx = df.groupby(["StudentID_SF", "ItemID_SF"])["T"].idxmin()
    return df.loc[idx].reset_index(drop=True)


def train_test_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """Random row-level train/test split."""
    mask = np.random.rand(len(df)) < test_fraction
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


def encode_ids(train_df):
    """Encode categorical IDs to integer indices. Returns indices + coord maps."""
    student_ids = train_df["StudentID_SF"].unique()
    item_ids = train_df["ItemID_SF"].unique()
    testlet_ids = train_df["Testlet_ID"].unique()

    student_map = {s: i for i, s in enumerate(student_ids)}
    item_map = {it: i for i, it in enumerate(item_ids)}
    testlet_map = {t: i for i, t in enumerate(testlet_ids)}

    coords = {
        "student": student_ids,
        "item": item_ids,
        "testlet": testlet_ids,
        "student_map": student_map,
        "item_map": item_map,
        "testlet_map": testlet_map,
    }

    s_idx = train_df["StudentID_SF"].map(student_map).values
    i_idx = train_df["ItemID_SF"].map(item_map).values
    t_idx = train_df["Testlet_ID"].map(testlet_map).values
    y = train_df["ItemScore"].values.astype(np.float32)

    return s_idx, i_idx, t_idx, y, coords


# ---------------------------------------------------------------------------
# Method 1: Penalised MLE (MAP) via PyTorch
# ---------------------------------------------------------------------------

class TestletMLE(nn.Module):
    """Bernoulli GLMM with three crossed random intercepts (MAP estimation).

    Model:
        logit(P(y=1)) = intercept + re_student[s] + re_item[i] + re_testlet[t]

    Penalised log-likelihood:
        L = log_lik + log_prior(sigma) + log_prior(re | sigma)
    """

    def __init__(self, n_students, n_items, n_testlets):
        super().__init__()
        self.intercept = nn.Parameter(torch.zeros(1))
        self.re_student = nn.Parameter(torch.zeros(n_students))
        self.re_item = nn.Parameter(torch.zeros(n_items))
        self.re_testlet = nn.Parameter(torch.zeros(n_testlets))
        # Log standard deviations (unconstrained parameterisation)
        self.log_sigma_student = nn.Parameter(torch.zeros(1))
        self.log_sigma_item = nn.Parameter(torch.zeros(1))
        self.log_sigma_testlet = nn.Parameter(torch.zeros(1))

    def forward(self, s_idx, i_idx, t_idx):
        eta = (
            self.intercept
            + self.re_student[s_idx]
            + self.re_item[i_idx]
            + self.re_testlet[t_idx]
        )
        return eta

    def penalised_nll(self, s_idx, i_idx, t_idx, y):
        """Negative penalised log-likelihood (to minimise)."""
        eta = self.forward(s_idx, i_idx, t_idx)
        # Binary cross-entropy (log-likelihood)
        nll = nn.functional.binary_cross_entropy_with_logits(eta, y, reduction="sum")

        # Normal priors on random effects: re ~ N(0, sigma^2)
        sigma_s = self.log_sigma_student.exp()
        sigma_i = self.log_sigma_item.exp()
        sigma_t = self.log_sigma_testlet.exp()

        re_penalty = (
            0.5 * (self.re_student ** 2).sum() / sigma_s ** 2
            + 0.5 * (self.re_item ** 2).sum() / sigma_i ** 2
            + 0.5 * (self.re_testlet ** 2).sum() / sigma_t ** 2
        )
        # Log-determinant terms for the normal prior
        n_s = len(self.re_student)
        n_i = len(self.re_item)
        n_t = len(self.re_testlet)
        log_det = (
            n_s * self.log_sigma_student
            + n_i * self.log_sigma_item
            + n_t * self.log_sigma_testlet
        )

        # Half-normal prior on sigmas: sigma ~ HalfNormal(2)
        sigma_prior = (
            0.5 * (sigma_s ** 2) / 4
            + 0.5 * (sigma_i ** 2) / 4
            + 0.5 * (sigma_t ** 2) / 4
        )

        return nll + re_penalty + log_det.sum() + sigma_prior


def fit_trt_mle(train_df, device="cpu", lr=0.01, epochs=500):
    """Fit TRT via penalised MLE (MAP) with PyTorch Adam.

    Returns
    -------
    result : dict
        Point estimates and standard errors for all parameters.
    coords : dict
        ID mappings.
    """
    s_idx, i_idx, t_idx, y, coords = encode_ids(train_df)

    s_idx = torch.tensor(s_idx, dtype=torch.long, device=device)
    i_idx = torch.tensor(i_idx, dtype=torch.long, device=device)
    t_idx = torch.tensor(t_idx, dtype=torch.long, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)

    model = TestletMLE(
        len(coords["student"]), len(coords["item"]), len(coords["testlet"])
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.penalised_nll(s_idx, i_idx, t_idx, y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={loss.item():.1f}  "
                  f"sigma_s={model.log_sigma_student.exp().item():.3f}  "
                  f"sigma_i={model.log_sigma_item.exp().item():.3f}  "
                  f"sigma_t={model.log_sigma_testlet.exp().item():.3f}")

    with torch.no_grad():
        result = {
            "intercept": model.intercept.cpu().item(),
            "re_student": model.re_student.cpu().numpy(),
            "re_item": model.re_item.cpu().numpy(),
            "re_testlet": model.re_testlet.cpu().numpy(),
            "sigma_student": model.log_sigma_student.exp().cpu().item(),
            "sigma_item": model.log_sigma_item.exp().cpu().item(),
            "sigma_testlet": model.log_sigma_testlet.exp().cpu().item(),
        }

    return result, coords


# ---------------------------------------------------------------------------
# Method 2: Full Bayesian via Pyro NUTS
# ---------------------------------------------------------------------------

def fit_trt_bayesian(
    train_df, device="cpu",
    warmup=1000, num_samples=1000, num_chains=1, target_accept=0.85,
):
    """Fit TRT via Pyro NUTS (GPU-accelerated MCMC).

    Model (equivalent to brms formula):
        ItemScore ~ 1 + (1 | StudentID_SF) + (1 | ItemID_SF) + (1 | Testlet_ID)
        family = bernoulli(logit)

    Returns
    -------
    result : dict
        Posterior samples for all parameters.
    coords : dict
        ID mappings.
    """
    import pyro
    import pyro.distributions as dist
    from pyro.infer import MCMC, NUTS

    pyro.clear_param_store()

    s_idx_np, i_idx_np, t_idx_np, y_np, coords = encode_ids(train_df)
    s_idx = torch.tensor(s_idx_np, dtype=torch.long, device=device)
    i_idx = torch.tensor(i_idx_np, dtype=torch.long, device=device)
    t_idx = torch.tensor(t_idx_np, dtype=torch.long, device=device)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device)

    n_students = len(coords["student"])
    n_items = len(coords["item"])
    n_testlets = len(coords["testlet"])

    def model(s_idx, i_idx, t_idx, y=None):
        intercept = pyro.sample("intercept", dist.Normal(0.0, 5.0))

        sigma_student = pyro.sample("sigma_student", dist.HalfNormal(2.0))
        sigma_item = pyro.sample("sigma_item", dist.HalfNormal(2.0))
        sigma_testlet = pyro.sample("sigma_testlet", dist.HalfNormal(2.0))

        with pyro.plate("students", n_students):
            re_student = pyro.sample("re_student", dist.Normal(0.0, sigma_student))
        with pyro.plate("items", n_items):
            re_item = pyro.sample("re_item", dist.Normal(0.0, sigma_item))
        with pyro.plate("testlets", n_testlets):
            re_testlet = pyro.sample("re_testlet", dist.Normal(0.0, sigma_testlet))

        eta = intercept + re_student[s_idx] + re_item[i_idx] + re_testlet[t_idx]

        with pyro.plate("obs", len(s_idx)):
            pyro.sample("y", dist.Bernoulli(logits=eta), obs=y)

    kernel = NUTS(model, target_accept_prob=target_accept, jit_compile=False)
    mcmc = MCMC(
        kernel,
        num_samples=num_samples,
        warmup_steps=warmup,
        num_chains=num_chains,
    )
    mcmc.run(s_idx, i_idx, t_idx, y=y_t)
    mcmc.summary()

    samples = mcmc.get_samples()
    result = {k: v.cpu().numpy() for k, v in samples.items()}

    return result, coords


# ---------------------------------------------------------------------------
# Prediction (works for both MLE and Bayesian results)
# ---------------------------------------------------------------------------

def predict_proba(result, coords, df, method="mle"):
    """Compute predicted P(y=1) for each row in df.

    MLE: point-estimate predictions.
    Bayesian: posterior-mean predictions.

    New levels unseen during training get random effect = 0.
    """
    student_map = coords["student_map"]
    item_map = coords["item_map"]
    testlet_map = coords["testlet_map"]

    s_idx = np.array([student_map.get(s, -1) for s in df["StudentID_SF"]])
    i_idx = np.array([item_map.get(it, -1) for it in df["ItemID_SF"]])
    t_idx = np.array([testlet_map.get(t, -1) for t in df["Testlet_ID"]])

    if method == "mle":
        intercept = result["intercept"]
        re_s = result["re_student"]
        re_i = result["re_item"]
        re_t = result["re_testlet"]

        eta = np.full(len(df), intercept)
        known_s = s_idx >= 0
        known_i = i_idx >= 0
        known_t = t_idx >= 0
        eta[known_s] += re_s[s_idx[known_s]]
        eta[known_i] += re_i[i_idx[known_i]]
        eta[known_t] += re_t[t_idx[known_t]]

        return 1 / (1 + np.exp(-eta))

    else:  # bayesian — average over posterior samples
        intercept = result["intercept"]  # (n_samples,)
        re_s = result["re_student"]  # (n_samples, n_students)
        re_i = result["re_item"]  # (n_samples, n_items)
        re_t = result["re_testlet"]  # (n_samples, n_testlets)

        n_samples = intercept.shape[0]
        n_obs = len(df)

        eta = np.broadcast_to(intercept[:, None], (n_samples, n_obs)).copy()

        known_s = s_idx >= 0
        if known_s.any():
            eta[:, known_s] += re_s[:, s_idx[known_s]]
        known_i = i_idx >= 0
        if known_i.any():
            eta[:, known_i] += re_i[:, i_idx[known_i]]
        known_t = t_idx >= 0
        if known_t.any():
            eta[:, known_t] += re_t[:, t_idx[known_t]]

        prob = 1 / (1 + np.exp(-eta))
        return prob.mean(axis=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_auc(result, coords, test_df, method="mle"):
    """Evaluate ROC-AUC on test data."""
    pred_mean = predict_proba(result, coords, test_df, method=method)
    auc = roc_auc_score(test_df["ItemScore"].values, pred_mean)
    return pred_mean, auc


def compute_person_fit(result, coords, train_df, method="mle") -> pd.DataFrame:
    """Compute infit and outfit statistics per student."""
    p_hat = predict_proba(result, coords, train_df, method=method)
    y = train_df["ItemScore"].values.astype(float)

    p_hat = np.clip(p_hat, 1e-6, 1 - 1e-6)
    res_z = (y - p_hat) / np.sqrt(p_hat * (1 - p_hat))
    w = p_hat * (1 - p_hat)

    fit_df = train_df[["StudentID_SF"]].copy()
    fit_df["res2"] = res_z ** 2
    fit_df["w"] = w
    fit_df["w_res2"] = w * res_z ** 2

    person_fit = (
        fit_df.groupby("StudentID_SF")
        .agg(
            n_items=("res2", "count"),
            outfit=("res2", "mean"),
            sum_w_res2=("w_res2", "sum"),
            sum_w=("w", "sum"),
        )
        .reset_index()
    )
    person_fit["infit"] = person_fit["sum_w_res2"] / person_fit["sum_w"]
    person_fit = person_fit.drop(columns=["sum_w_res2", "sum_w"])
    person_fit = person_fit.sort_values("outfit", ascending=False).reset_index(drop=True)
    return person_fit


# ---------------------------------------------------------------------------
# Random effects extraction
# ---------------------------------------------------------------------------

def extract_random_effects(result, coords, method="mle"):
    """Extract random effect estimates as DataFrames (analogous to brms ranef()).

    MLE: point estimates (no uncertainty intervals).
    Bayesian: posterior mean + 95% CI.
    """
    def _make_df_mle(re_values, id_name, id_values):
        return pd.DataFrame({
            id_name: list(id_values),
            "Estimate": re_values,
        })

    def _make_df_bayesian(samples, id_name, id_values):
        # samples shape: (n_posterior, n_levels)
        return pd.DataFrame({
            id_name: list(id_values),
            "Estimate": samples.mean(axis=0),
            "Est.Error": samples.std(axis=0),
            "Q2.5": np.percentile(samples, 2.5, axis=0),
            "Q97.5": np.percentile(samples, 97.5, axis=0),
        })

    if method == "mle":
        ability_df = _make_df_mle(result["re_student"], "StudentID_SF", coords["student"])
        item_df = _make_df_mle(result["re_item"], "ItemID_SF", coords["item"])
        testlet_df = _make_df_mle(result["re_testlet"], "Testlet_ID", coords["testlet"])
    else:
        ability_df = _make_df_bayesian(result["re_student"], "StudentID_SF", coords["student"])
        item_df = _make_df_bayesian(result["re_item"], "ItemID_SF", coords["item"])
        testlet_df = _make_df_bayesian(result["re_testlet"], "Testlet_ID", coords["testlet"])

    return ability_df, item_df, testlet_df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

import seaborn as sns


def plot_param_hist(values, param_name, filename, result_dir, xlabel,
                    bins=30, pct_clip=(1, 99)):
    """Plot histogram + KDE, clipping x-axis to the given percentile range.

    Matches the standard style used by cirt.py and elo.py.
    """
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


def plot_roc(test_df, pred_mean, auc_value, save_path):
    """Plot ROC curve."""
    from sklearn.metrics import RocCurveDisplay
    fig, ax = plt.subplots(figsize=figsizes.aaai2024_half()["figure.figsize"])
    RocCurveDisplay.from_predictions(
        test_df["ItemScore"].values, pred_mean, ax=ax, color=COLORS[0]
    )
    ax.set_title(f"ROC Curve (AUC = {auc_value:.3f})")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_testlet_vs_correctness(testlet_df, first_data, save_path):
    """Scatter: testlet random effect vs. average correctness."""
    correctness = (
        first_data.groupby("Testlet_ID")["ItemScore"]
        .mean()
        .reset_index()
        .rename(columns={"ItemScore": "correctness"})
    )
    merged = testlet_df.merge(correctness, on="Testlet_ID", how="left")

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_half()["figure.figsize"])
    ax.scatter(merged["Estimate"], merged["correctness"], color=COLORS[0], alpha=0.7)
    mask = merged[["Estimate", "correctness"]].dropna()
    if len(mask) >= 2:
        z = np.polyfit(mask["Estimate"], mask["correctness"], 1)
        xs = np.linspace(mask["Estimate"].min(), mask["Estimate"].max(), 100)
        ax.plot(xs, np.polyval(z, xs), color=COLORS[1], linewidth=1.5)
    ax.set_xlabel("Testlet RE (logit)")
    ax.set_ylabel("Mean correctness")
    ax.set_title("Testlet random effect vs. average correctness")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_item_difficulties_by_testlet(item_difficulty_by_testlet, save_path):
    """Dot plot of item difficulties grouped by testlet."""
    means = (
        item_difficulty_by_testlet.groupby("Testlet_ID")["Estimate"]
        .mean()
        .reset_index()
        .rename(columns={"Estimate": "mean_difficulty"})
    )
    plot_df = item_difficulty_by_testlet.merge(means, on="Testlet_ID", how="left")
    order = means.sort_values("mean_difficulty")["Testlet_ID"].tolist()

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
    y_positions = {tid: i for i, tid in enumerate(order)}
    ys = plot_df["Testlet_ID"].map(y_positions)
    ax.scatter(plot_df["Estimate"], ys, s=20, alpha=0.7, color=COLORS[0])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6)
    ax.set_xlabel("Item difficulty (logit)")
    ax.set_ylabel("Testlet ID")
    ax.set_title("Item difficulties by testlet")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def summarise_testlet_ranges(item_difficulty_by_testlet, top_n=5):
    """Return testlets with the largest and smallest difficulty ranges."""
    ranges = (
        item_difficulty_by_testlet.groupby("Testlet_ID")
        .agg(
            range_difficulty=("Estimate", lambda x: x.max() - x.min()),
            n_items=("Estimate", "count"),
        )
        .reset_index()
    )
    ranges = ranges[ranges["n_items"] > 1]
    top = ranges.nlargest(top_n, "range_difficulty")
    bottom = ranges.nsmallest(top_n, "range_difficulty")
    return top, bottom


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Testlet Response Theory analysis")
    p.add_argument("--method", choices=["mle", "bayesian"], default="mle",
                   help="Inference method (default: mle)")
    p.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    p.add_argument("--lr", type=float, default=MLE_LR, help="MLE learning rate")
    p.add_argument("--epochs", type=int, default=MLE_EPOCHS, help="MLE epochs")
    p.add_argument("--warmup", type=int, default=BAYES_WARMUP, help="NUTS warmup steps")
    p.add_argument("--samples", type=int, default=BAYES_SAMPLES, help="NUTS posterior samples")
    p.add_argument("--chains", type=int, default=BAYES_CHAINS, help="NUTS chains")
    p.add_argument("--target_accept", type=float, default=TARGET_ACCEPT)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--use_all_testlets", action="store_true")
    p.add_argument("--use_all_students", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    output_dir = os.path.join(REPO_ROOT, "results", "testlet")
    ensure_dir(output_dir)

    # 1) Load + Basic Preprocessing
    print("Loading data from HuggingFace...")
    data_raw = load_data()
    data_last = keep_latest_attempt(data_raw)

    kept_item_ids = drop_no_variation_items(data_last)
    print(f"  Kept {len(kept_item_ids)} items with variation "
          f"(dropped {data_last['ItemID_SF'].nunique() - len(kept_item_ids)})")

    # 2) Optional subsetting for a fast run
    first_data = subset_by_testlet_student(
        data_raw,
        use_all_testlets=args.use_all_testlets,
        use_all_students=args.use_all_students,
        n_testlet_groups=N_TESTLET_GROUPS,
        n_student_groups=N_STUDENT_GROUPS,
    )
    print(f"  Subset: {len(first_data)} rows, "
          f"{first_data['StudentID_SF'].nunique()} students, "
          f"{first_data['ItemID_SF'].nunique()} items, "
          f"{first_data['Testlet_ID'].nunique()} testlets")

    # Train/test split on first attempts only
    first_attempts = first_attempt_only(first_data)
    train_data, test_data = train_test_split(first_attempts, test_fraction=TEST_FRACTION)
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

    # 3) Fit TRT
    method = args.method
    print(f"\nFitting TRT model (method={method}, device={args.device})...")

    if method == "mle":
        result, coords = fit_trt_mle(
            train_data, device=args.device, lr=args.lr, epochs=args.epochs,
        )
        print(f"  intercept={result['intercept']:.3f}  "
              f"sigma_s={result['sigma_student']:.3f}  "
              f"sigma_i={result['sigma_item']:.3f}  "
              f"sigma_t={result['sigma_testlet']:.3f}")
    else:
        result, coords = fit_trt_bayesian(
            train_data, device=args.device,
            warmup=args.warmup, num_samples=args.samples,
            num_chains=args.chains, target_accept=args.target_accept,
        )

    # 4) Evaluate (ROC-AUC) + Person fit
    print("\nEvaluating on test set...")
    pred_mean, auc_value = eval_auc(result, coords, test_data, method=method)
    print(f"  AUC: {auc_value:.3f}")

    print("\nComputing person fit (infit/outfit)...")
    person_fit = compute_person_fit(result, coords, train_data, method=method)
    person_fit_path = os.path.join(output_dir, "person_fit.csv")
    person_fit.to_csv(person_fit_path, index=False)
    print(f"  Saved: {person_fit_path}")
    print(person_fit.head(10).to_string(index=False))

    # 5) Random Effects
    print("\nExtracting random effects...")
    ability_df, item_df, testlet_df = extract_random_effects(result, coords, method=method)

    ability_df.to_csv(os.path.join(output_dir, "ability_re.csv"), index=False)
    item_df.to_csv(os.path.join(output_dir, "item_re.csv"), index=False)
    testlet_df.to_csv(os.path.join(output_dir, "testlet_re.csv"), index=False)
    print(f"  Saved random effects to {output_dir}/")

    # Map items to testlets
    item_map_df = first_data[["ItemID_SF", "Testlet_ID"]].drop_duplicates()
    combined_df = (
        item_df.merge(item_map_df, on="ItemID_SF", how="left")
        .merge(testlet_df, on="Testlet_ID", how="left", suffixes=("_item", "_testlet"))
    )
    combined_df["total_difficulty"] = combined_df["Estimate_item"] + combined_df["Estimate_testlet"]
    combined_df = combined_df[
        ["ItemID_SF", "Testlet_ID", "Estimate_item", "Estimate_testlet", "total_difficulty"]
    ]
    combined_df.to_csv(os.path.join(output_dir, "combined_difficulty.csv"), index=False)

    # Per-testlet item difficulties
    difficulty_df = item_df[["ItemID_SF", "Estimate"]].copy()
    item_difficulty_by_testlet = difficulty_df.merge(item_map_df, on="ItemID_SF", how="left")
    item_difficulty_by_testlet = item_difficulty_by_testlet.dropna(subset=["Testlet_ID"])

    # 6) Plots
    print("\nGenerating plots...")

    # Distribution histograms (matching cirt.py / elo.py style)
    plot_param_hist(
        ability_df["Estimate"].values, r"$\theta$", "theta",
        output_dir, r"$\theta$ (Student Ability)",
    )
    plot_param_hist(
        item_df["Estimate"].values, r"$b$", "difficulty",
        output_dir, r"$b$ (Item Difficulty)", bins=50, pct_clip=(2, 98),
    )
    plot_param_hist(
        testlet_df["Estimate"].values, r"$\gamma$", "testlet_effect",
        output_dir, r"$\gamma$ (Testlet Effect)",
    )

    # Testlet-specific plots
    plot_roc(test_data, pred_mean, auc_value,
             os.path.join(output_dir, "roc_curve.png"))
    plot_testlet_vs_correctness(testlet_df, first_data,
                                os.path.join(output_dir, "testlet_vs_correctness.png"))
    plot_item_difficulties_by_testlet(item_difficulty_by_testlet,
                                      os.path.join(output_dir, "item_difficulties_by_testlet.png"))

    # Testlet range summary
    top_ranges, bottom_ranges = summarise_testlet_ranges(item_difficulty_by_testlet)
    print("\nTestlets with LARGEST difficulty range:")
    print(top_ranges.to_string(index=False))
    print("\nTestlets with SMALLEST difficulty range:")
    print(bottom_ranges.to_string(index=False))

    # Save all metrics
    metrics = {
        "method": method,
        "auc": auc_value,
        "n_train": len(train_data),
        "n_test": len(test_data),
        "n_students": int(first_data["StudentID_SF"].nunique()),
        "n_items": int(first_data["ItemID_SF"].nunique()),
        "n_testlets": int(first_data["Testlet_ID"].nunique()),
        "seed": args.seed,
        "device": args.device,
    }
    if method == "mle":
        metrics.update({"lr": args.lr, "epochs": args.epochs})
    else:
        metrics.update({
            "warmup": args.warmup, "samples": args.samples,
            "chains": args.chains, "target_accept": args.target_accept,
        })
    metrics_path = os.path.join(output_dir, "fit_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved: {metrics_path}")
    print(f"Results saved to: {output_dir}/")

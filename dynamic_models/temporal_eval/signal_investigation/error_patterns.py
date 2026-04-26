import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from huggingface_hub import snapshot_download

snapshot_dir = snapshot_download("CodeInsightTeam/simulation_output", repo_type="dataset", local_files_only=True)
csv_dir = snapshot_download("CodeInsightTeam/code_insights_csv", repo_type="dataset", local_files_only=True)

# Load sim data
sim_records = []
with open(f"{snapshot_dir}/v4_profile_mindiff/glm_v4_merged.jsonl") as f:
    for line in f:
        sim_records.append(json.loads(line))

sim_df = pd.DataFrame(sim_records)
print(f"Sim records: {len(sim_df)}, columns: {list(sim_df.columns)}")

# Load real data
real_df = pd.read_csv(f"{csv_dir}/main_data.csv")
print(f"Real records: {len(real_df)}, columns: {list(real_df.columns)}")

# Filter real: Submit only, first submit per (student, question)
real_sub = real_df[real_df["response_type"] == "Submit"].copy()
real_sub = real_sub.sort_values("attempt_id")
real_sub = real_sub.drop_duplicates(subset=["student_id", "question_unittest_id"], keep="first")
real_sub = real_sub[real_sub["pass"].notna() & (real_sub["pass"] != "")]
print(f"Real first-submits with pass: {len(real_sub)}")

# Filter sim: Submit only, first submit per (student_id, question_unittest_id)
sim_sub = sim_df.copy()
if "response_type" in sim_df.columns:
    sim_sub = sim_sub[sim_sub["response_type"] == "Submit"]
print(f"Sim submit records: {len(sim_sub)}")

# Deduplicate sim to first per (student_id, question_unittest_id)
if "attempt_id" in sim_sub.columns:
    sim_sub = sim_sub.sort_values("attempt_id")
sim_sub = sim_sub.drop_duplicates(subset=["student_id", "question_unittest_id"], keep="first")
sim_sub = sim_sub[sim_sub["pass"].notna() & (sim_sub["pass"] != "")]
print(f"Sim first-submits with pass: {len(sim_sub)}")

# Per-question per-test-case failure rates
def failure_rates(df, qid_col="question_unittest_id", pass_col="pass"):
    rates = {}
    for qid, grp in df.groupby(qid_col):
        passes = grp[pass_col].astype(str)
        max_len = passes.str.len().max()
        passes_padded = passes.apply(lambda s: s.ljust(max_len, '0'))
        arr = np.array([[int(c) for c in row] for row in passes_padded])
        rates[qid] = 1 - arr.mean(axis=0)  # failure rate per test case
    return rates

sim_rates = failure_rates(sim_sub)
real_rates = failure_rates(real_sub)

common_qids = set(sim_rates.keys()) & set(real_rates.keys())
print(f"\nCommon questions: {len(common_qids)}")

# Correlate per-question: match test-case vectors by position (min length)
pearson_rs, spearman_rs, qids_used = [], [], []
for qid in common_qids:
    sv = sim_rates[qid]
    rv = real_rates[qid]
    n = min(len(sv), len(rv))
    if n < 2:
        continue
    sv, rv = sv[:n], rv[:n]
    if sv.std() == 0 or rv.std() == 0:
        continue
    pr, _ = pearsonr(sv, rv)
    sr, _ = spearmanr(sv, rv)
    pearson_rs.append(pr)
    spearman_rs.append(sr)
    qids_used.append(qid)

print(f"Questions with valid correlation: {len(qids_used)}")
print(f"\nPer-question Pearson r (sim vs real test-case failure rates):")
print(f"  Mean:   {np.mean(pearson_rs):.4f}")
print(f"  Median: {np.median(pearson_rs):.4f}")
print(f"  Std:    {np.std(pearson_rs):.4f}")
print(f"  >0.3:   {np.mean(np.array(pearson_rs) > 0.3):.2%}")
print(f"  >0:     {np.mean(np.array(pearson_rs) > 0):.2%}")

print(f"\nPer-question Spearman r:")
print(f"  Mean:   {np.mean(spearman_rs):.4f}")
print(f"  Median: {np.median(spearman_rs):.4f}")

# Global: pool all (sim_failure_rate, real_failure_rate) pairs across all questions
all_sim, all_real = [], []
for qid in qids_used:
    sv = sim_rates[qid]
    rv = real_rates[qid]
    n = min(len(sv), len(rv))
    all_sim.extend(sv[:n])
    all_real.extend(rv[:n])

all_sim = np.array(all_sim)
all_real = np.array(all_real)
gp, _ = pearsonr(all_sim, all_real)
gs, _ = spearmanr(all_sim, all_real)
print(f"\nGlobal (pooled across all questions):")
print(f"  Pearson r:  {gp:.4f}")
print(f"  Spearman r: {gs:.4f}")
print(f"  N test-case pairs: {len(all_sim)}")

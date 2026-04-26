"""
Investigates whether LLM attempt count (as a difficulty proxy) predicts
real student outcomes.
"""
import json
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score

SIM_DIR = "/Users/fagunpatel/.cache/huggingface/hub/datasets--CodeInsightTeam--simulation_output/snapshots/82bafd4c518609edebd904c0fd3eab6ee9c9a69a/v4_profile_mindiff"
REAL_CSV = "/Users/fagunpatel/.cache/huggingface/hub/datasets--CodeInsightTeam--code_insights_csv/snapshots/a88c99da850ddd26e2f4612b5147eb9efead9aa9/main_data.csv"
MAX_ATTEMPTS = 50


def is_full_pass(pass_field) -> bool:
    s = str(pass_field).strip()
    return len(s) > 0 and all(c == '1' for c in s)


def load_sim_metrics() -> pd.DataFrame:
    shards = sorted(glob.glob(f"{SIM_DIR}/glm_*.jsonl"))
    # group attempts per (student, question) simulation run
    # attempt_id is 0-indexed within the 50-attempt budget
    rows = []
    with open(shards[0]) as f0, open(shards[1]) as f1, open(shards[2]) as f2:
        for fh in (f0, f1, f2):
            for line in fh:
                rec = json.loads(line)
                if rec.get('response_type') != 'Submit':
                    continue
                rows.append({
                    'student_id': rec['student_id'],
                    'qid': rec['question_unittest_id'],
                    'attempt_id': int(rec['attempt_id']),
                    'full_pass': is_full_pass(rec['pass']),
                })
    sim = pd.DataFrame(rows)

    # For each (student, question) run: find first full-pass attempt
    def first_pass(g):
        passed = g[g['full_pass']]['attempt_id']
        if passed.empty:
            return pd.Series({'first_pass_attempt': np.nan, 'never_passed': True})
        return pd.Series({'first_pass_attempt': passed.min(), 'never_passed': False})

    metrics = sim.groupby(['student_id', 'qid']).apply(first_pass).reset_index()

    # Per-question aggregates
    q_metrics = metrics.groupby('qid').agg(
        mean_attempts=('first_pass_attempt', lambda x: x.mean()),
        never_pass_frac=('never_passed', 'mean'),
    ).reset_index()
    return q_metrics, metrics


def load_real_pass_rates() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(REAL_CSV, on_bad_lines='skip', low_memory=False)
    submits = df[df['response_type'] == 'Submit'].copy()
    submits['pass_str'] = submits['pass'].astype(str).str.strip()
    submits['full_pass'] = submits['pass_str'].apply(
        lambda s: len(s) > 0 and s not in ('nan', '') and all(c == '1' for c in s)
    )

    # Per-question: whether ANY submit was full pass (student solved it)
    student_q = submits.groupby(['student_id', 'question_unittest_id'])['full_pass'].max().reset_index()
    student_q.columns = ['student_id', 'qid', 'solved']

    q_real = student_q.groupby('qid')['solved'].mean().reset_index()
    q_real.columns = ['qid', 'real_pass_rate']
    q_real['qid'] = q_real['qid'].astype(str)

    student_q['student_id'] = student_q['student_id'].astype(str)
    student_q['qid'] = student_q['qid'].astype(str)
    return q_real, student_q


def main():
    print("Loading sim data...")
    q_sim, student_sim = load_sim_metrics()
    q_sim['qid'] = q_sim['qid'].astype(str)
    student_sim['student_id'] = student_sim['student_id'].astype(str)
    student_sim['qid'] = student_sim['qid'].astype(str)

    print("Loading real data...")
    q_real, student_real = load_real_pass_rates()

    merged = q_sim.merge(q_real, on='qid')
    print(f"\nQuestions with both sim and real data: {len(merged)}")
    print(merged.describe().to_string())

    # --- Question-level correlations ---
    mask = merged['mean_attempts'].notna()
    m = merged[mask]

    sp_att, sp_att_p = spearmanr(m['mean_attempts'], m['real_pass_rate'])
    pe_att, pe_att_p = pearsonr(m['mean_attempts'], m['real_pass_rate'])
    sp_nev, sp_nev_p = spearmanr(merged['never_pass_frac'], merged['real_pass_rate'])
    pe_nev, pe_nev_p = pearsonr(merged['never_pass_frac'], merged['real_pass_rate'])

    print("\n=== Question-level: LLM difficulty vs real pass rate ===")
    print(f"mean_attempts_to_pass  Spearman r={sp_att:.3f} (p={sp_att_p:.3g})  Pearson r={pe_att:.3f} (p={pe_att_p:.3g})  n={len(m)}")
    print(f"never_pass_fraction    Spearman r={sp_nev:.3f} (p={sp_nev_p:.3g})  Pearson r={pe_nev:.3f} (p={pe_nev_p:.3g})  n={len(merged)}")

    # --- Individual student-level AUC ---
    ind = student_real.merge(student_sim[['student_id', 'qid', 'first_pass_attempt', 'never_passed']], on=['student_id', 'qid'])
    ind = ind.merge(q_sim[['qid', 'mean_attempts', 'never_pass_frac']], on='qid')
    print(f"\nStudent-question pairs for individual AUC: {len(ind)}")

    # Predict solved=1 from low LLM difficulty (invert: low attempts → easy → more likely solved)
    # Use never_pass_frac and mean_attempts as predictors (negated so higher = more likely real pass)
    ind_valid = ind.dropna(subset=['mean_attempts', 'solved'])
    if ind_valid['solved'].nunique() > 1:
        auc_att = roc_auc_score(ind_valid['solved'], -ind_valid['mean_attempts'])
        auc_nev = roc_auc_score(ind_valid['solved'], -ind_valid['never_pass_frac'])
        print(f"\n=== Individual-level AUC (predict student solved) ===")
        print(f"  -mean_attempts_to_pass : AUC = {auc_att:.4f}")
        print(f"  -never_pass_fraction   : AUC = {auc_nev:.4f}")
    else:
        print("Not enough label variation for AUC.")


if __name__ == '__main__':
    main()

"""Bayesian Knowledge Tracing (BKT) adapter for temporal evaluation.

Per-item HMM with binary latent state (known/unknown), fit via EM:

    P(correct | known) = 1 - P(S)
    P(correct | unknown) = P(G)
    P(known_t+1 | unknown_t) = P(T)

Parameters per item: P(L0), P(T), P(G), P(S).
"""

import numpy as np

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..temporal_split import TemporalSplit

CLIP_LO = 0.01
CLIP_HI = 0.99


def _em_single_item(sequences, n_iter=50):
    """Fit BKT parameters for one item via EM over student attempt sequences.

    sequences: list of 1D arrays, each array is a student's binary responses
               across attempts for this item.
    Returns: (p_l0, p_t, p_g, p_s)
    """
    p_l0, p_t, p_g, p_s = 0.3, 0.1, 0.2, 0.1

    for _ in range(n_iter):
        sum_gamma0_known = 0.0
        sum_xi_learn = 0.0
        sum_xi_stay_unknown = 0.0
        sum_obs_correct_given_known = 0.0
        sum_obs_given_known = 0.0
        sum_obs_correct_given_unknown = 0.0
        sum_obs_given_unknown = 0.0
        n_sequences = 0

        for seq in sequences:
            T = len(seq)
            if T == 0:
                continue
            n_sequences += 1

            # Forward pass
            alpha = np.zeros((T, 2))  # col 0 = unknown, col 1 = known
            # t=0
            emit_known = (1 - p_s) if seq[0] == 1 else p_s
            emit_unknown = p_g if seq[0] == 1 else (1 - p_g)
            alpha[0, 1] = p_l0 * emit_known
            alpha[0, 0] = (1 - p_l0) * emit_unknown
            alpha[0] /= alpha[0].sum() + 1e-30

            for t in range(1, T):
                emit_known = (1 - p_s) if seq[t] == 1 else p_s
                emit_unknown = p_g if seq[t] == 1 else (1 - p_g)
                # transition: unknown->unknown (1-p_t), unknown->known (p_t), known->known (1)
                pred_unknown = alpha[t - 1, 0] * (1 - p_t)
                pred_known = alpha[t - 1, 0] * p_t + alpha[t - 1, 1]
                alpha[t, 0] = pred_unknown * emit_unknown
                alpha[t, 1] = pred_known * emit_known
                norm = alpha[t].sum() + 1e-30
                alpha[t] /= norm

            # Backward pass
            beta = np.zeros((T, 2))
            beta[T - 1] = 1.0

            for t in range(T - 2, -1, -1):
                emit_known_next = (1 - p_s) if seq[t + 1] == 1 else p_s
                emit_unknown_next = p_g if seq[t + 1] == 1 else (1 - p_g)
                beta[t, 0] = ((1 - p_t) * emit_unknown_next * beta[t + 1, 0]
                              + p_t * emit_known_next * beta[t + 1, 1])
                beta[t, 1] = emit_known_next * beta[t + 1, 1]
                norm = beta[t].sum() + 1e-30
                beta[t] /= norm

            # Gamma (posterior state probabilities)
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-30

            # Accumulate sufficient statistics
            sum_gamma0_known += gamma[0, 1]

            for t in range(T - 1):
                emit_known_next = (1 - p_s) if seq[t + 1] == 1 else p_s
                emit_unknown_next = p_g if seq[t + 1] == 1 else (1 - p_g)

                # xi: P(z_t=unknown, z_{t+1}=known | obs)
                xi_learn = (alpha[t, 0] * p_t * emit_known_next * beta[t + 1, 1])
                xi_stay = (alpha[t, 0] * (1 - p_t) * emit_unknown_next * beta[t + 1, 0])
                xi_norm = xi_learn + xi_stay + 1e-30
                xi_learn /= xi_norm
                xi_stay /= xi_norm

                # Weight by P(unknown at t)
                p_unknown_t = gamma[t, 0]
                sum_xi_learn += xi_learn * p_unknown_t
                sum_xi_stay_unknown += xi_stay * p_unknown_t

            for t in range(T):
                if seq[t] == 1:
                    sum_obs_correct_given_known += gamma[t, 1]
                    sum_obs_correct_given_unknown += gamma[t, 0]
                sum_obs_given_known += gamma[t, 1]
                sum_obs_given_unknown += gamma[t, 0]

        if n_sequences == 0:
            return 0.5, 0.1, 0.25, 0.1

        # M-step
        p_l0 = np.clip(sum_gamma0_known / n_sequences, CLIP_LO, CLIP_HI)

        denom_t = sum_xi_learn + sum_xi_stay_unknown + 1e-30
        p_t = np.clip(sum_xi_learn / denom_t, CLIP_LO, CLIP_HI)

        p_s = np.clip(1.0 - sum_obs_correct_given_known / (sum_obs_given_known + 1e-30),
                       CLIP_LO, CLIP_HI)
        p_g = np.clip(sum_obs_correct_given_unknown / (sum_obs_given_unknown + 1e-30),
                       CLIP_LO, CLIP_HI)

    return p_l0, p_t, p_g, p_s


def _bkt_predict(seq, p_l0, p_t, p_g, p_s):
    """Run BKT forward pass and return P(known) after observing the sequence."""
    p_known = p_l0
    for r in seq:
        if r == 1:
            p_known_post = (p_known * (1 - p_s)) / (
                p_known * (1 - p_s) + (1 - p_known) * p_g + 1e-30
            )
        else:
            p_known_post = (p_known * p_s) / (
                p_known * p_s + (1 - p_known) * (1 - p_g) + 1e-30
            )
        p_known = p_known_post + (1 - p_known_post) * p_t
    return p_known


class BKTAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "BKT"

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        em_iters: int = 50,
        **kwargs,
    ) -> PredictionResult:
        np.random.seed(seed)

        N = data.n_students
        T = data.n_max_attempts
        corr = data.correctness_matrix.numpy()

        # ---- Fit per-item BKT on train items ----
        train_idx = split.train_item_indices.numpy()
        Q_train = len(train_idx)

        item_params_list = []
        for qi, item_i in enumerate(train_idx):
            sequences = []
            for s in range(N):
                seq = corr[s, item_i, :T]
                valid = seq[seq != -1]
                if len(valid) > 0:
                    sequences.append(valid.astype(float))
            params = _em_single_item(sequences, n_iter=em_iters)
            item_params_list.append(params)

            if (qi + 1) % 50 == 0 or qi == 0:
                print(f"    [BKT] EM fit: {qi+1}/{Q_train} items", flush=True)

        item_params_arr = np.array(item_params_list)  # (Q_train, 4)

        # Global average params for test items
        avg_params = item_params_arr.mean(axis=0)
        avg_p_l0, avg_p_t, avg_p_g, avg_p_s = avg_params
        print(f"    [BKT] Avg params: L0={avg_p_l0:.3f} T={avg_p_t:.3f} "
              f"G={avg_p_g:.3f} S={avg_p_s:.3f}", flush=True)

        # ---- Compute per-student mastery from train history ----
        student_mastery = np.full(N, avg_p_l0)
        for s in range(N):
            mastery_sum = 0.0
            n_items_seen = 0
            for qi, item_i in enumerate(train_idx):
                seq = corr[s, item_i, :T]
                valid = seq[seq != -1]
                if len(valid) > 0:
                    p_l0, p_t, p_g, p_s = item_params_list[qi]
                    m = _bkt_predict(valid, p_l0, p_t, p_g, p_s)
                    mastery_sum += m
                    n_items_seen += 1
            if n_items_seen > 0:
                student_mastery[s] = mastery_sum / n_items_seen

        # ---- Predict on test items ----
        test_idx = split.test_item_indices.numpy()
        Q_test = len(test_idx)

        test_corr = corr[:, test_idx, :T]
        y_flat = test_corr.reshape(-1)
        valid_mask = y_flat != -1
        valid_indices = np.where(valid_mask)[0]

        test_student_idx = valid_indices // (Q_test * T)
        test_local_q_idx = (valid_indices // T) % Q_test
        test_attempt_idx = valid_indices % T
        test_item_idx = test_idx[test_local_q_idx]

        y_true = y_flat[valid_mask].astype(float)

        # Predict: P(correct) = P(known)*P(~slip) + P(~known)*P(guess)
        p_known = student_mastery[test_student_idx]
        y_pred_prob = p_known * (1 - avg_p_s) + (1 - p_known) * avg_p_g
        y_pred_prob = np.clip(y_pred_prob, 1e-6, 1 - 1e-6)

        if len(y_true) == 0:
            raise ValueError(
                f"No test observations for cutoff_week={split.cutoff_week}"
            )

        print(f"    [BKT] {N} students, {Q_train} train items, "
              f"{Q_test} test items, {len(y_true)} test obs", flush=True)

        return PredictionResult(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            student_indices=test_student_idx,
            item_indices=test_item_idx,
            attempt_indices=test_attempt_idx,
            losses=None,
            student_params={
                "mastery": student_mastery,
            },
            item_params={
                "P(L0)": item_params_arr[:, 0],
                "P(T)": item_params_arr[:, 1],
                "P(G)": item_params_arr[:, 2],
                "P(S)": item_params_arr[:, 3],
            },
            model_state={
                "item_params": item_params_arr,
                "avg_params": avg_params,
                "student_mastery": student_mastery,
                "em_iters": em_iters,
            },
        )

    def fit_and_predict_student_split(
        self, data, split, seed=42, em_iters=50, **kwargs,
    ):
        np.random.seed(seed)

        T = data.n_max_attempts
        corr = data.correctness_matrix.numpy()
        Q = data.n_items
        train_students = split.train_student_indices
        test_students = split.test_student_indices

        # ---- Training: fit per-item BKT on train students, ALL items ----
        item_params_all = []
        for qi in range(Q):
            sequences = []
            for s in train_students:
                seq = corr[s, qi, :]
                valid = seq[seq != -1]
                if len(valid) > 0:
                    sequences.append(valid.astype(float))
            params = _em_single_item(sequences, n_iter=em_iters)
            item_params_all.append(params)

            if (qi + 1) % 100 == 0 or qi == 0:
                print(f"    [BKT] Training: {qi+1}/{Q} items", flush=True)

        item_params_arr = np.array(item_params_all)  # (Q, 4)

        # ---- Scoring: test students, weeks 1-W → estimate mastery ----
        scoring_items = split.train_item_indices
        N_test = len(test_students)

        student_mastery = np.full(N_test, 0.5)
        for si, s in enumerate(test_students):
            mastery_sum = 0.0
            n_seen = 0
            for qi in scoring_items:
                seq = corr[s, qi, :]
                valid = seq[seq != -1]
                if len(valid) > 0:
                    p_l0, p_t, p_g, p_s = item_params_all[qi]
                    m = _bkt_predict(valid, p_l0, p_t, p_g, p_s)
                    mastery_sum += m
                    n_seen += 1
            if n_seen > 0:
                student_mastery[si] = mastery_sum / n_seen

        print(f"    [BKT] Calibrated {N_test} test students", flush=True)

        # ---- Predict: test students, weeks W+1+ items ----
        test_items = split.test_item_indices
        Q_test = len(test_items)

        pred_corr = corr[np.ix_(test_students, test_items)][:, :, :T]
        y_flat = pred_corr.reshape(-1)
        valid_mask = y_flat != -1
        valid_indices = np.where(valid_mask)[0]

        s_idx = valid_indices // (Q_test * T)
        q_idx = (valid_indices // T) % Q_test
        a_idx = valid_indices % T
        item_idx = test_items[q_idx]

        y_true = y_flat[valid_mask].astype(float)

        # Per-item calibrated params for test items
        test_p_g = item_params_arr[test_items, 2][q_idx]
        test_p_s = item_params_arr[test_items, 3][q_idx]
        p_known = student_mastery[s_idx]

        y_pred_prob = p_known * (1 - test_p_s) + (1 - p_known) * test_p_g
        y_pred_prob = np.clip(y_pred_prob, 1e-6, 1 - 1e-6)

        print(f"    [BKT] Predict: {len(y_true)} test obs", flush=True)

        return PredictionResult(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            student_indices=s_idx,
            item_indices=item_idx,
            attempt_indices=a_idx,
            student_params={"mastery": student_mastery},
            item_params={
                "P(L0)": item_params_arr[:, 0],
                "P(T)": item_params_arr[:, 1],
                "P(G)": item_params_arr[:, 2],
                "P(S)": item_params_arr[:, 3],
            },
            model_state={
                "item_params": item_params_arr,
                "student_mastery": student_mastery,
            },
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        return 3.0

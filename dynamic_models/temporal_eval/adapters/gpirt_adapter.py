"""GPIRT adapter for temporal evaluation.

Masks test-item columns as -1 in the correctness matrix, runs blocked ESS
inference with testlet effects, then predicts test items using posterior
abilities + prior difficulty (0).
"""

import os
import sys

import numpy as np
import torch

from ..base_adapter import ModelAdapter, PredictionResult
from ..data_loader import UnifiedData
from ..temporal_split import TemporalSplit


class GPIRTAdapter(ModelAdapter):

    @property
    def name(self) -> str:
        return "GPIRT"

    def fit_and_predict(
        self,
        data: UnifiedData,
        split: TemporalSplit,
        seed: int = 42,
        n_samples: int = 2000,
        warmup: int = 500,
        kernel: str = "RBF",
        length_scale: float = 1.0,
        thin: int = 5,
        testlet: bool = True,
        **kwargs,
    ) -> PredictionResult:
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        from dynamic_models.gpirt import preprocess

        # Mask test items in both correctness and time matrices
        # Both must be masked so preprocess() builds consistent indexes
        masked_corr = data.correctness_matrix.clone()
        masked_corr[:, split.test_item_indices, :] = -1

        masked_time = data.time_matrix.clone()
        masked_time[:, split.test_item_indices, :] = -1

        # Preprocess (builds GP priors from time_matrix)
        low_rank_configs = {
            "type": "GP",
            "kernel": kernel,
            "length_scale": length_scale,
        }
        all_indexes = preprocess(
            masked_corr,
            masked_time,
            low_rank_configs,
            device=device,
        )

        n_students = data.n_students

        # Build model adapter with testlet effects for better mixing
        if testlet:
            from dynamic_models.gpirt import GPIRTTestletModelAdapter

            qi = data.question_infos
            item_to_question = torch.tensor(
                qi["qidx"].values, dtype=torch.long
            )
            model_adapter = GPIRTTestletModelAdapter(
                masked_corr, all_indexes, device, item_to_question,
            )
        else:
            from dynamic_models.gpirt import GPIRTModelAdapter

            model_adapter = GPIRTModelAdapter(
                masked_corr, all_indexes, device,
            )

        # Run blocked ESS inference
        from dynamic_models.gpirt import BlockedGibbsESSampler

        total_draws = n_samples + warmup
        sampler = BlockedGibbsESSampler(model_adapter, device=device)
        draw_result = sampler.draw(n=total_draws, thin=thin)
        ability_chain = draw_result[0]  # [n_stored, total_ability_dims]

        # Discard warmup from chain
        warmup_stored = warmup // max(thin, 1)
        ability_chain = ability_chain[warmup_stored:]

        # Posterior mean abilities per dimension
        mean_abilities = ability_chain.mean(dim=0)  # [total_ability_dims]

        # Per-student ability offsets (maps student → slice of flat abilities)
        offsets = model_adapter.ability_offsets

        # Build predictions for test items
        test_corr = data.correctness_matrix[:, split.test_item_indices, :]
        test_obs_mask = (test_corr != -1)

        y_true_list = []
        y_pred_list = []
        student_idx_list = []
        item_idx_list = []

        for sidx in range(n_students):
            student_test = test_corr[sidx]  # [Q_test, T]
            student_mask = test_obs_mask[sidx]  # [Q_test, T]

            if not student_mask.any():
                continue

            y_vals = student_test[student_mask].numpy()
            y_true_list.append(y_vals)

            # Track which (student, item) each observation belongs to
            obs_items = student_mask.nonzero(as_tuple=False)[:, 0]  # local test item idx
            global_items = split.test_item_indices[obs_items].numpy()
            student_idx_list.append(np.full(len(y_vals), sidx))
            item_idx_list.append(global_items)

            # Use last train time point ability as best current estimate
            start = offsets[sidx]
            end = offsets[sidx + 1]
            if end > start:
                ability_val = mean_abilities[end - 1].item()
            else:
                ability_val = 0.0  # no train observations → prior mean

            # Test difficulty = 0 (prior), testlet effect = 0 (prior)
            prob = 1.0 / (1.0 + np.exp(-ability_val))
            y_pred_list.append(np.full(len(y_vals), prob))

        if not y_true_list:
            raise ValueError(
                f"No test observations for cutoff_week={split.cutoff_week}"
            )

        y_true = np.concatenate(y_true_list)
        y_pred_prob = np.concatenate(y_pred_list)

        return PredictionResult(
            y_true=y_true,
            y_pred_prob=y_pred_prob,
            student_indices=np.concatenate(student_idx_list),
            item_indices=np.concatenate(item_idx_list),
        )

    def estimated_runtime_minutes(self, data: UnifiedData) -> float:
        # Blocked ESS with testlet on full dataset: ~2-3 hours per horizon
        return 180.0

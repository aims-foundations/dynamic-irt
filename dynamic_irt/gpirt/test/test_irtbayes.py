import random
import unittest

import numpy as np
import scipy

import torch
from irt_bayes import IRTBayes
from utils import get_ability_priors


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


def irt_1d_1pl(ability, difficulty):
    return torch.sigmoid(ability + difficulty)


class IRTBayesTestingSingleAttempt(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(IRTBayesTestingSingleAttempt, self).__init__(*args, **kwargs)
        set_seed(42)
        self.n_students = 25
        self.n_questions = 10
        self.n_testcases = 2
        self.n_attempts = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.true_time = torch.randn(
            self.n_students,
            self.n_questions * self.n_attempts,
            device=self.device,
        )
        self.true_time = torch.sort(self.true_time, dim=1)[0]
        self.response_time_matrix = self.true_time.reshape(
            self.n_students, self.n_questions, 1, self.n_attempts
        ).expand(-1, -1, self.n_testcases, -1)
        self.response_time_matrix = self.response_time_matrix.flatten(1, 2)

        self.true_ability = []
        for sidx in range(self.n_students):
            self.true_ability.append(
                get_ability_priors(
                    self.true_time[sidx],
                    kernel="RBF",
                    length_scale=1.0,
                    device=self.device,
                )
                .sample()
                .reshape(self.n_questions, self.n_attempts)
            )
        self.true_ability = torch.stack(self.true_ability)

        self.true_difficulty = torch.randn(
            (self.n_questions, self.n_testcases),
            device=self.device,
        ).flatten()
        # >>> q1_t1, q1_t2, ..., qn_tm

        self.response_matrix = torch.distributions.bernoulli.Bernoulli(
            probs=irt_1d_1pl(
                self.true_ability.repeat_interleave(self.n_testcases, dim=1),
                self.true_difficulty[:, None],
            )
        ).sample()
        # >>> n_students x n_questions x 1

    def test_hmc_1d_1pl(self):
        set_seed(42)
        irt_model = IRTBayes(
            D=1,
            PL=1,
            device=self.device,
            report_to=None,
            low_rank_configs={
                "type": "GP",
                "kernel": "RBF",
                "length_scale": 1.0,
            },
        )

        irt_model.fit(
            max_epoch=50,
            warmup_steps=10,
            response_matrix=self.response_matrix,
            response_time_matrix=self.response_time_matrix,
            method="hmc",
            embedding=None,
            model_features=None,
            amortized_question_hyperparams=None,
            amortized_model_hyperparams=None,
        )
        ability = irt_model.get_abilities()
        ability = torch.stack(ability)
        # >>> n_students x n_samples x ( n_questions * 1 )

        difficulty = irt_model.get_difficulties()
        # >>> n_samples x n_questions

        list_ability_spearman = []
        for sample_id in range(ability.shape[1]):
            ability_spearman = scipy.stats.spearmanr(
                ability[:, sample_id].flatten().cpu().detach().numpy(),
                self.true_ability.cpu().numpy().flatten(),
            )[0]
            list_ability_spearman.append(ability_spearman)
        self.assertGreater(np.mean(list_ability_spearman), 0.80)

        list_difficulty_spearman = []
        for sample_id in range(difficulty.shape[0]):
            difficulty_spearman = scipy.stats.spearmanr(
                difficulty[sample_id].cpu().detach().numpy(),
                self.true_difficulty.cpu().numpy(),
            )[0]
            list_difficulty_spearman.append(difficulty_spearman)
        self.assertGreater(np.mean(list_difficulty_spearman), 0.60)


def main():
    unittest.main()


if __name__ == "__main__":
    main()

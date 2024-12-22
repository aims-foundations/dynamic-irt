import logging

import pyro
import pyro.distributions as dist

import torch
import torch.nn as nn
from pyro.infer import MCMC, NUTS
from tqdm import tqdm

from utils import get_ability_priors_pyro


class IRTBayes(nn.Module):
    def __init__(
        self,
        low_rank_configs=None,
        D=1,
        PL=1,
        device="cpu",
        report_to=None,
    ):
        super().__init__()
        self.D = D
        self.PL = PL
        self.device = device
        self.report_to = report_to
        self.low_rank_configs = low_rank_configs
        if low_rank_configs is None:
            self.low_rank_configs = {
                "type": "GP",
                "kernel": "RBF",
                "length_scale": 1.0,
            }

    @classmethod
    def compute_prob(
        cls,
        ability,
        difficulty,
        disciminatory=None,
        guessing=None,
        loading_factor=None,
    ):
        ab_shape = list(ability.size())
        df_shape = list(difficulty.size())
        if disciminatory is None:
            dc_shape = df_shape[:-1] + [ab_shape[-2]] + [df_shape[-1]]
            disciminatory = torch.ones(dc_shape).to(ability)
        if guessing is None:
            gs_shape = df_shape[:-1] + [ab_shape[-2]] + [df_shape[-1]]
            guessing = torch.zeros(gs_shape).to(ability)
        if loading_factor is None:
            lf_shape = ab_shape[:-1] + [1] + [ab_shape[-1]]
            loading_factor = torch.ones(lf_shape).to(ability)

        if ab_shape == df_shape:
            return guessing + (1 - guessing) * torch.sigmoid(
                disciminatory * (ability * loading_factor) + difficulty
            )
        else:
            ability = ability[..., None, :]
            difficulty = difficulty[..., None, :]
            return guessing + (1 - guessing) * torch.sigmoid(
                disciminatory * (ability * loading_factor).sum(-1) + difficulty
            )

    def log_likelihood(self, *args, **kwargs):
        probs = self.compute_prob(*args, **kwargs)
        return (
            torch.distributions.Bernoulli(probs=probs)
            .log_prob(self.flat_response_matrix)
            .sum()
        )

    def init_parameters(
        self,
        response_matrix,
        response_time_matrix=None,
        amortize_item=False,
        amortize_student=False,
        amortized_question_hyperparams=None,
        amortized_model_hyperparams=None,
        fitting_method=None,
    ):
        self.n_students, self.n_questions = response_matrix.shape[:2]
        if response_matrix.ndim == 3:
            self.n_max_attempts = response_matrix.shape[2]

        self.amortize_item = amortize_item
        self.amortize_student = amortize_student
        self.amortized_question_hyperparams = amortized_question_hyperparams
        self.amortized_model_hyperparams = amortized_model_hyperparams

        if self.amortize_student:
            raise NotImplementedError("Work in progress")

        else:
            if response_time_matrix is None:
                raise NotImplementedError("Work in progress")

            # response_time_matrix: n_students x n_questions x n_attempts
            if self.low_rank_configs["type"] == "GP":
                self.observation_mask = (response_matrix != -1).cpu()
                self.response_time_indexes = []
                self.student_observation_indexes = []
                self.question_expanding_indexes = []
                self.ability_prior_dists = []

                for sidx in tqdm(range(self.n_students), desc="Constructing indexes"):
                    uni_time = response_time_matrix[sidx].unique()
                    has_missing = 1 if -1 in uni_time else 0

                    # Get the time index for each attempt
                    time_index = (
                        torch.searchsorted(uni_time, response_time_matrix[sidx])
                        - has_missing
                    )
                    ### REMEMBER: time_index is 0-indexed. In case, element 0 is -1
                    ### We need to subtract 1 to get the correct index
                    self.response_time_indexes.append(time_index[time_index != -1])
                    self.student_observation_indexes.extend(
                        [sidx] * len(self.response_time_indexes[-1])
                    )
                    self.question_expanding_indexes.append(
                        torch.arange(self.n_questions, device=self.device)[
                            :, None
                        ].expand(-1, self.n_max_attempts)[self.observation_mask[sidx]]
                    )

                    # Remove the first element since it is -1
                    uni_time = uni_time[has_missing:]
                    self.ability_prior_dists.append(
                        get_ability_priors_pyro(
                            uni_time,
                            kernel=self.low_rank_configs["kernel"],
                            length_scale=self.low_rank_configs["length_scale"],
                            device=self.device,
                        )
                    )

                self.student_observation_indexes = torch.tensor(
                    self.student_observation_indexes
                ).cpu()
                self.question_expanding_indexes = torch.concatenate(
                    self.question_expanding_indexes
                ).cpu()

                self.question_observation_indexes = []
                for qidx in tqdm(
                    range(self.n_questions), desc="Constructing question indexes"
                ):
                    self.question_observation_indexes.append(
                        torch.where(self.question_expanding_indexes == qidx)[0][
                            0
                        ].item()
                    )
                self.question_observation_indexes = torch.tensor(
                    self.question_observation_indexes, device="cpu"
                )

                self.ability = None

            else:
                raise NotImplementedError("Work in progress")

        if self.D != 1:
            raise NotImplementedError("Work in progress")

        if self.PL != 1:
            raise NotImplementedError("Work in progress")

        if self.amortize_item:
            raise NotImplementedError("Work in progress")

        else:
            if self.low_rank_configs["type"] == "GP":
                if fitting_method == "hmc":
                    self.item_prior_dists = dist.Normal(
                        torch.zeros(self.n_questions, device=self.device),
                        torch.ones(self.n_questions, device=self.device),
                    )
                else:
                    self.item_prior_dists = torch.distributions.Normal(
                        torch.zeros(self.n_questions, device=self.device).double(),
                        torch.ones(self.n_questions, device=self.device).double(),
                    )
                self.difficulty = None
            else:
                raise NotImplementedError("Work in progress")

    def fit(
        self,
        method="hmc",
        max_epoch=1000,
        warmup_steps=200,
        response_matrix=None,
        response_time_matrix=None,
        embedding=None,
        model_features=None,
        amortized_question_hyperparams=None,
        amortized_model_hyperparams=None,
    ):
        if embedding:
            amortize_item = True
            assert (
                amortized_question_hyperparams is not None
            ), "Please provide hyperparameters for item amortization"
        else:
            amortize_item = False

        if model_features:
            amortize_student = True
            assert (
                amortized_model_hyperparams is not None
            ), "Please provide hyperparameters for student amortization"
        else:
            amortize_student = False

        assert response_matrix.shape == response_time_matrix.shape, (
            f"response_matrix and response_time_matrix should have the same shape. "
            f"Got {response_matrix.shape} and {response_time_matrix.shape}"
        )

        with torch.no_grad():
            self.init_parameters(
                response_matrix=response_matrix,
                response_time_matrix=response_time_matrix,
                amortize_item=amortize_item,
                amortize_student=amortize_student,
                amortized_question_hyperparams=None,
                amortized_model_hyperparams=None,
                fitting_method=method,
            )

        if method == "hmc":
            return self.infer_hmc(
                max_epoch=max_epoch,
                warmup_steps=warmup_steps,
                response_matrix=response_matrix,
                response_time_matrix=response_time_matrix,
                embedding=embedding,
                model_features=model_features,
            )
        else:
            raise ValueError(f"{method} is not supported")

    def forward(self):
        raise NotImplementedError("Work in progress")

    def _irt_pyro_model(
        self,
        response_matrix: torch.Tensor,
        response_time_matrix: torch.Tensor,
    ):
        # Initialize difficulty parameters
        with pyro.plate("input_difficulty"):
            difficulty = pyro.sample("difficulty", self.item_prior_dists)
        difficulty = difficulty[self.question_expanding_indexes]

        # Initialize ability parameters
        abilities = []
        for sid, (ability_prior, time_index) in enumerate(
            zip(self.ability_prior_dists, self.response_time_indexes)
        ):
            ab = pyro.sample(f"ability_{sid}", ability_prior)
            abilities.append(ab[time_index])
        abilities = torch.cat(abilities)

        # Compute probability matrix
        prob_matrix = self.compute_prob(
            abilities, difficulty, disciminatory=1, guessing=0, loading_factor=1
        )
        with pyro.plate("output"):
            y = pyro.sample("y", dist.Bernoulli(prob_matrix), obs=response_matrix)

        return y

    def infer_hmc(
        self,
        max_epoch: int,
        warmup_steps: int,
        response_matrix: torch.Tensor,
        response_time_matrix: torch.Tensor,
        embedding=None,
        model_features=None,
    ):
        assert max_epoch > warmup_steps, "max_epoch should be greater than warmup_steps"
        logging.info("Running inference...")

        kernel = NUTS(
            self._irt_pyro_model,
            jit_compile=True,
            ignore_jit_warnings=True,
        )

        # We'll define a hook_fn to log potential energy values during inference.
        # This is helpful to diagnose whether the chain is mixing.
        energies = []

        def hook_fn(kernel, *unused):
            e = float(kernel._potential_energy_last)
            energies.append(e)
            logging.info("potential = {:0.6g}".format(e))

        mcmc = MCMC(
            kernel,
            hook_fn=hook_fn,
            num_samples=max_epoch - warmup_steps,
            warmup_steps=warmup_steps,
        )

        # Run HMC
        mcmc.run(
            response_matrix=response_matrix[self.observation_mask],
            response_time_matrix=response_time_matrix,
        )

        mcmc.summary()

        # Get samples
        samples = mcmc.get_samples()
        self.ability = []
        for sidx in range(self.n_students):
            self.ability.append(samples[f"ability_{sidx}"])

        self.difficulty = samples["difficulty"]

        return mcmc

    def get_abilities(self):
        return self.ability

    def get_difficulties(self):
        return self.difficulty

    def get_disciminatory(self):
        raise NotImplementedError("Work in progress")

    def get_guessing(self):
        raise NotImplementedError("Work in progress")

    def get_loading_factor(self):
        raise NotImplementedError("Work in progress")

    def get_item_parameters(self):
        raise NotImplementedError("Work in progress")

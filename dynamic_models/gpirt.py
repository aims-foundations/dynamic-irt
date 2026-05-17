"""
GPIRT inference using Elliptical Slice Sampling (ESS).

ESS is a gradient-free MCMC method well-suited for models with Gaussian priors
(like GP-IRT). It avoids the costly gradient computation of HMC/NUTS and scales
better to large numbers of students.

Usage:
    cd CodeInsights/dynamic_irt
    CUDA_VISIBLE_DEVICES=3 python gpirt.py --course_name dsa_hk231 --n_samples 500 --warmup 100
    CUDA_VISIBLE_DEVICES=3 python gpirt.py --course_name dsa_hk231 --n_students 200 --n_samples 200 --warmup 50
"""

import argparse
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import snapshot_download
# Repo root (CodeInsights/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


# ---------------------------------------------------------------------------
# Helpers (inlined from old utils.py)
# ---------------------------------------------------------------------------

def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _rbf_kernel(x, lengthscale):
    """RBF (squared-exponential) kernel. x: (n, 1) double tensor."""
    diff = x - x.T
    return torch.exp(-diff.pow(2) / (2.0 * float(lengthscale) ** 2))


def _matern52_kernel(x, lengthscale):
    """Matern-5/2 kernel. x: (n, 1) double tensor."""
    r = (x - x.T).abs()
    s = (5.0 ** 0.5) * r / float(lengthscale)
    return (1.0 + s + s.pow(2) / 3.0) * torch.exp(-s)


def get_ability_priors(unique_time_vec, kernel, length_scale=1.0, device="cpu"):
    """Build a MultivariateNormal prior over abilities given time points."""
    uni_time = unique_time_vec.unsqueeze(-1).double()
    if kernel == "Matern":
        covar = _matern52_kernel(uni_time, length_scale)
    elif kernel == "RBF":
        covar = _rbf_kernel(uni_time, length_scale)
    else:
        raise ValueError("Invalid kernel type")

    covar = covar + 1e-4 * torch.eye(
        uni_time.shape[0], device=uni_time.device, dtype=torch.double,
    )
    return torch.distributions.MultivariateNormal(
        torch.zeros(covar.shape[0], device=uni_time.device, dtype=torch.double), covar
    )


def preprocess(response_matrix, response_time_matrix, low_rank_configs, device):
    n_students, n_questions = response_matrix.shape[:2]
    if response_matrix.ndim == 3:
        n_max_attempts = response_matrix.shape[2]

    if low_rank_configs["type"] == "GP":
        observation_mask = (response_matrix != -1).cpu()
        response_time_indexes = []
        question_expanding_indexes = []
        ability_prior_dists = []

        for sidx in tqdm(range(n_students), desc="Constructing indexes"):
            uni_time = response_time_matrix[sidx].unique()
            has_missing = 1 if -1 in uni_time else 0

            time_index = (
                torch.searchsorted(uni_time, response_time_matrix[sidx]) - has_missing
            )
            response_time_indexes.append(time_index[time_index != -1])
            question_expanding_indexes.append(
                torch.arange(n_questions, device=device)[:, None].expand(
                    -1, n_max_attempts
                )[observation_mask[sidx]]
            )

            uni_time = uni_time[has_missing:]
            ability_prior_dists.append(
                get_ability_priors(
                    uni_time,
                    kernel=low_rank_configs["kernel"],
                    length_scale=low_rank_configs["length_scale"],
                    device=device,
                )
            )

        question_expanding_indexes = torch.concatenate(question_expanding_indexes).cpu()

        question_observation_indexes = []
        for qidx in tqdm(range(n_questions), desc="Constructing question indexes"):
            matches = torch.where(question_expanding_indexes == qidx)[0]
            if len(matches) > 0:
                question_observation_indexes.append(matches[0].item())
            else:
                question_observation_indexes.append(-1)
        question_observation_indexes = torch.tensor(
            question_observation_indexes, device="cpu"
        )
    else:
        raise NotImplementedError("Work in progress")

    if low_rank_configs["type"] == "GP":
        item_prior_dists = torch.distributions.Normal(
            torch.zeros(n_questions, device=device),
            torch.ones(n_questions, device=device),
        )
    else:
        raise NotImplementedError("Work in progress")

    return (
        observation_mask,
        response_time_indexes,
        question_expanding_indexes,
        ability_prior_dists,
        item_prior_dists,
    )


def build_item_to_question(course_name):
    """Return (n_items,) int tensor mapping each item index to its question index."""
    import pandas as pd
    data_folder = snapshot_download(
        repo_id="CodeInsightTeam/code_insights_matrices", repo_type="dataset"
    )
    qinfo = pd.read_csv(os.path.join(data_folder, course_name, "question_infos.csv"))
    return torch.tensor(qinfo["qidx"].values, dtype=torch.long)


class GPIRTModelAdapter:
    """Adapter that exposes the GPIRT model interface required by GibbsESSampler.

    Required methods (original, for joint ESS):
        sample_theta_prior() -> (abilities_vector, support_points)
        sample_item_prior()  -> difficulties_vector
        log_likelihood(ability, difficulty, ...) -> scalar log-likelihood

    Additional methods (for blocked/per-student ESS):
        sample_student_prior(s) -> ability_vector for student s
        student_log_likelihood(s, student_ability, difficulty) -> scalar
    """

    def __init__(self, response_matrix, all_indexes, device):
        (
            self.observation_mask,
            self.response_time_indexes,
            self.question_expanding_indexes,
            self.ability_prior_dists,
            self.item_prior_dists,
        ) = all_indexes

        # Flattened observed responses
        self.response_data = response_matrix[self.observation_mask].to(device)
        self.question_expanding_indexes = self.question_expanding_indexes.to(device)
        self.n_students = len(self.ability_prior_dists)
        self.n_questions = self.item_prior_dists.loc.shape[0]
        self.device = device

        # Precompute per-student segment sizes and time indexes on device
        self.segment_sizes = []
        self.time_indexes_device = []
        for prior, time_idx in zip(self.ability_prior_dists, self.response_time_indexes):
            n_timepoints = prior.loc.shape[0]
            self.segment_sizes.append(n_timepoints)
            self.time_indexes_device.append(time_idx.to(device))

        # Precompute per-student observation slices for blocked ESS
        # response_data and question_expanding_indexes are flattened across all
        # students in order. We need the start/end index for each student.
        self.student_obs_slices = []
        obs_offset = 0
        for time_idx in self.time_indexes_device:
            n_obs = time_idx.shape[0]
            self.student_obs_slices.append((obs_offset, obs_offset + n_obs))
            obs_offset += n_obs

        # --- Precomputed tensors for batched ESS ---
        n_obs = self.response_data.shape[0]

        # obs_to_student: (n_obs,) maps each observation to its student index
        self.obs_to_student = torch.zeros(n_obs, dtype=torch.long, device=device)
        for s, (start, end) in enumerate(self.student_obs_slices):
            self.obs_to_student[start:end] = s

        # ability_offsets: cumulative dims per student in concatenated ability vector
        self.ability_offsets = [0]
        for seg_size in self.segment_sizes:
            self.ability_offsets.append(self.ability_offsets[-1] + seg_size)
        self.total_ability_dims = self.ability_offsets[-1]

        # obs_to_ability_idx: (n_obs,) maps each obs to index in flat ability vector
        self.obs_to_ability_idx = torch.empty(n_obs, dtype=torch.long, device=device)
        for s, (start, end) in enumerate(self.student_obs_slices):
            self.obs_to_ability_idx[start:end] = (
                self.ability_offsets[s] + self.time_indexes_device[s]
            )

        # dim_to_student: (total_ability_dims,) maps each ability dim to its student
        self.dim_to_student = torch.empty(
            self.total_ability_dims, dtype=torch.long, device=device
        )
        for s, seg_size in enumerate(self.segment_sizes):
            self.dim_to_student[
                self.ability_offsets[s]:self.ability_offsets[s + 1]
            ] = s

        # Pre-extract Cholesky factors on GPU for faster sequential sampling
        self._cholesky_factors = []
        for s, prior in enumerate(self.ability_prior_dists):
            self._cholesky_factors.append(
                prior._unbroadcasted_scale_tril.to(device)
            )

    def batched_sample_ability_prior(self):
        """Sample from all students' GP priors, concatenated flat.

        Uses pre-cached Cholesky factors on GPU for L @ z sampling.

        Returns:
            (total_ability_dims,) flat vector of prior samples.
        """
        samples = []
        for s in range(self.n_students):
            L = self._cholesky_factors[s]  # (T_s, T_s) already on GPU
            z = torch.randn(L.shape[0], 1, dtype=L.dtype, device=self.device)
            samples.append((L @ z).squeeze(-1))
        return torch.cat(samples)

    def batched_student_ability_ll(self, abilities_flat, difficulties):
        """Compute per-student log-likelihoods from a flat ability vector.

        Args:
            abilities_flat: (total_ability_dims,) concatenated abilities
            difficulties: (n_items,) difficulty vector

        Returns:
            (n_students,) per-student log-likelihoods
        """
        ab_at_obs = abilities_flat[self.obs_to_ability_idx]
        diff_at_obs = difficulties[self.question_expanding_indexes]
        logit = ab_at_obs + diff_at_obs
        log_probs = self.response_data * logit - F.softplus(logit)
        student_lls = torch.zeros(
            self.n_students, dtype=logit.dtype, device=self.device
        )
        student_lls.scatter_add_(0, self.obs_to_student, log_probs)
        return student_lls

    def sample_theta_prior(self):
        """Sample abilities from the GP prior for each student, concatenated."""
        samples = []
        for prior in self.ability_prior_dists:
            samples.append(prior.sample().to(self.device))
        return torch.cat(samples), None

    def sample_student_prior(self, s):
        """Sample ability from the GP prior for a single student."""
        return self.ability_prior_dists[s].sample().to(self.device)

    def sample_item_prior(self):
        """Sample difficulties from Normal(0, 1)."""
        return self.item_prior_dists.sample().to(self.device)

    def log_likelihood(self, ability, difficulty, **kwargs):
        """Compute total Bernoulli log-likelihood for observed responses."""
        # Expand difficulty to observation-level via question_expanding_indexes
        diff_expanded = difficulty[self.question_expanding_indexes]

        # Extract ability at each observed (student, item, attempt) triple
        abilities_at_obs = []
        offset = 0
        for seg_size, time_idx in zip(self.segment_sizes, self.time_indexes_device):
            student_ability = ability[offset:offset + seg_size]
            abilities_at_obs.append(student_ability[time_idx])
            offset += seg_size
        abilities_at_obs = torch.cat(abilities_at_obs)

        # Bernoulli log-likelihood: sum over all observations
        logit = abilities_at_obs + diff_expanded
        log_prob = torch.distributions.Bernoulli(logits=logit).log_prob(self.response_data)
        return log_prob.sum()

    def student_log_likelihood(self, s, student_ability, difficulty):
        """Compute log-likelihood for a single student's observations.

        Args:
            s: Student index.
            student_ability: (n_timepoints_s,) ability vector for student s.
            difficulty: (n_questions,) full difficulty vector.

        Returns:
            Scalar log-likelihood summed over this student's observations.
        """
        obs_start, obs_end = self.student_obs_slices[s]
        time_idx = self.time_indexes_device[s]
        q_idx = self.question_expanding_indexes[obs_start:obs_end]
        resp = self.response_data[obs_start:obs_end]

        logit = student_ability[time_idx] + difficulty[q_idx]
        return torch.distributions.Bernoulli(logits=logit).log_prob(resp).sum()


class GPIRTTestletModelAdapter(GPIRTModelAdapter):
    """Extends GPIRT with per-(student, question) testlet effects γ_{sj}.

    Generative model:
        θ_s ~ GP(0, K)
        σ²_j = 1  (fixed; avoids MCEM degeneracy)
        γ_{sj} ~ N(0, σ²_j)
        z_{jk} ~ N(0, 1)
        y_{sjk} ~ Bernoulli(sigmoid(θ_s(t) + γ_{sj} + z_{jk}))
    """

    def __init__(self, response_matrix, all_indexes, device, item_to_question):
        super().__init__(response_matrix, all_indexes, device)

        # item_to_question: (n_items,) int tensor, values in [0, n_actual_questions)
        self.item_to_question = item_to_question.to(device)
        self.n_actual_questions = int(item_to_question.max().item()) + 1

        # Per-observation question index (actual question, not item)
        self.obs_to_question = self.item_to_question[
            self.question_expanding_indexes.to(device)
        ]

        # obs_to_student is already built in base class

        # Testlet parameters (initialized at 0 / 1)
        self.gamma = torch.zeros(
            self.n_students, self.n_actual_questions, device=device
        )
        self.sigma2 = torch.ones(self.n_actual_questions, device=device)

        # --- Precompute per-question item/observation indices for blocked
        #     difficulty ESS (230 blocks of ~10 dims each) ---
        self.question_item_indices = []   # question_item_indices[q] = item indices
        self.question_obs_indices = []    # question_obs_indices[q] = obs indices
        for q in range(self.n_actual_questions):
            items = (self.item_to_question == q).nonzero(as_tuple=True)[0]
            self.question_item_indices.append(items)
            if len(items) > 0:
                obs_mask = torch.isin(self.question_expanding_indexes, items)
                self.question_obs_indices.append(
                    obs_mask.nonzero(as_tuple=True)[0])
            else:
                self.question_obs_indices.append(
                    torch.tensor([], dtype=torch.long, device=device))

        # Global map: item → local index within its question (for fast LL)
        self.item_local_idx = torch.zeros(
            self.n_questions, dtype=torch.long, device=device)
        for q in range(self.n_actual_questions):
            items = self.question_item_indices[q]
            for local, item in enumerate(items):
                self.item_local_idx[item] = local

    def sample_gamma_prior(self):
        """Sample γ ~ N(0, σ²_j) for all (student, question) pairs."""
        std = self.sigma2.sqrt()  # (n_actual_questions,)
        return torch.randn(
            self.n_students, self.n_actual_questions, device=self.device
        ) * std.unsqueeze(0)

    def sample_student_gamma_prior(self, s):
        """Sample γ_s ~ N(0, σ²_j) for a single student (n_actual_questions,)."""
        std = self.sigma2.sqrt()
        return torch.randn(self.n_actual_questions, device=self.device) * std

    def batched_student_ability_ll(self, abilities_flat, difficulties):
        """Override base: include gamma in per-student log-likelihoods."""
        ab_at_obs = abilities_flat[self.obs_to_ability_idx]
        diff_at_obs = difficulties[self.question_expanding_indexes]
        gam_at_obs = self.gamma[self.obs_to_student, self.obs_to_question]
        logit = ab_at_obs + diff_at_obs + gam_at_obs
        log_probs = self.response_data * logit - F.softplus(logit)
        student_lls = torch.zeros(
            self.n_students, dtype=logit.dtype, device=self.device
        )
        student_lls.scatter_add_(0, self.obs_to_student, log_probs)
        return student_lls

    def batched_student_gamma_ll(self, gamma, abilities_flat, difficulties):
        """Compute per-student log-likelihoods from a gamma matrix.

        Args:
            gamma: (n_students, n_actual_questions) proposed gamma
            abilities_flat: (total_ability_dims,) concatenated abilities
            difficulties: (n_items,) difficulty vector

        Returns:
            (n_students,) per-student log-likelihoods
        """
        ab_at_obs = abilities_flat[self.obs_to_ability_idx]
        diff_at_obs = difficulties[self.question_expanding_indexes]
        gam_at_obs = gamma[self.obs_to_student, self.obs_to_question]
        logit = ab_at_obs + diff_at_obs + gam_at_obs
        log_probs = self.response_data * logit - F.softplus(logit)
        student_lls = torch.zeros(
            self.n_students, dtype=logit.dtype, device=self.device
        )
        student_lls.scatter_add_(0, self.obs_to_student, log_probs)
        return student_lls

    def student_gamma_log_likelihood(self, s, gamma_s, ability_s, difficulty):
        """Log-likelihood for student s given their gamma vector.

        Args:
            s: Student index.
            gamma_s: (n_actual_questions,) gamma vector for student s.
            ability_s: (n_timepoints_s,) ability vector for student s.
            difficulty: (n_questions,) full difficulty vector.
        """
        obs_start, obs_end = self.student_obs_slices[s]
        time_idx = self.time_indexes_device[s]
        q_idx = self.question_expanding_indexes[obs_start:obs_end]
        tq_idx = self.obs_to_question[obs_start:obs_end]
        resp = self.response_data[obs_start:obs_end]

        logit = ability_s[time_idx] + difficulty[q_idx] + gamma_s[tq_idx]
        return torch.distributions.Bernoulli(logits=logit).log_prob(resp).sum()

    def assemble_abilities_at_obs(self, student_abilities):
        """Map per-student ability vectors to observation-level abilities."""
        parts = []
        for s, (start, end) in enumerate(self.student_obs_slices):
            parts.append(student_abilities[s][self.time_indexes_device[s]])
        return torch.cat(parts)

    def assemble_gamma_at_obs(self):
        """Map gamma matrix to observation-level gamma values."""
        return self.gamma[self.obs_to_student, self.obs_to_question]

    def question_difficulty_log_likelihood(self, q, diff_q,
                                           abilities_at_obs, gamma_at_obs):
        """Log-likelihood for observations involving question q's items.

        Only sums over observations that use items belonging to question q,
        so this is efficient for per-question blocked difficulty ESS.

        Args:
            q: Question index.
            diff_q: (n_items_in_q,) proposed difficulty for items in question q.
            abilities_at_obs: (n_obs,) precomputed ability at each observation.
            gamma_at_obs: (n_obs,) precomputed gamma at each observation.
        """
        obs_idx = self.question_obs_indices[q]
        if len(obs_idx) == 0:
            return torch.tensor(0.0, device=self.device)

        resp = self.response_data[obs_idx]
        ab = abilities_at_obs[obs_idx]
        gam = gamma_at_obs[obs_idx]

        # Map observation item indices → local index within diff_q
        q_idx = self.question_expanding_indexes[obs_idx]
        local_idx = self.item_local_idx[q_idx]
        diff_at_obs = diff_q[local_idx]

        logit = ab + diff_at_obs + gam
        return torch.distributions.Bernoulli(logits=logit).log_prob(resp).sum()

    def log_likelihood(self, ability, difficulty, gamma=None):
        """Total Bernoulli log-likelihood including testlet effects."""
        if gamma is None:
            gamma = self.gamma
        diff_expanded = difficulty[self.question_expanding_indexes]
        gamma_expanded = gamma[self.obs_to_student, self.obs_to_question]

        abilities_at_obs = []
        offset = 0
        for seg_size, time_idx in zip(self.segment_sizes, self.time_indexes_device):
            abilities_at_obs.append(ability[offset:offset + seg_size][time_idx])
            offset += seg_size
        abilities_at_obs = torch.cat(abilities_at_obs)

        logit = abilities_at_obs + diff_expanded + gamma_expanded
        return torch.distributions.Bernoulli(logits=logit).log_prob(
            self.response_data
        ).sum()

    def student_log_likelihood(self, s, student_ability, difficulty, gamma=None):
        """Per-student log-likelihood including testlet effects."""
        if gamma is None:
            gamma = self.gamma
        obs_start, obs_end = self.student_obs_slices[s]
        time_idx = self.time_indexes_device[s]
        q_idx = self.question_expanding_indexes[obs_start:obs_end]
        tq_idx = self.obs_to_question[obs_start:obs_end]
        resp = self.response_data[obs_start:obs_end]

        logit = student_ability[time_idx] + difficulty[q_idx] + gamma[s, tq_idx]
        return torch.distributions.Bernoulli(logits=logit).log_prob(resp).sum()


# ---------------------------------------------------------------------------
# ESS samplers
# ---------------------------------------------------------------------------

class GibbsESSampler:
    """Base Elliptical Slice Sampler with joint ability/difficulty updates."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.abilities = None
        self.difficulties = None
        self.device = device

    def draw(self, n: int = 1):
        """Draw n samples, returning (abilities, difficulties) tensors."""
        list_ability = []
        list_difficulty = []
        pbar = tqdm(range(n))
        for _ in pbar:
            self.step()
            list_ability.append(self.abilities.cpu())
            list_difficulty.append(self.difficulties.cpu())
            pbar.set_postfix({"llh": self.log_likelihood.item()})
        return torch.stack(list_ability), torch.stack(list_difficulty)

    def step(self):
        """Take one Gibbs step: update abilities then difficulties."""
        if self.abilities is None:
            self.abilities, self.support_points = self.model.sample_theta_prior()
        if self.difficulties is None:
            self.difficulties = self.model.sample_item_prior()

        self.abilities = self.step_ability()
        self.difficulties = self.step_difficulty()
        return self.abilities, self.difficulties

    def step_ability(self):
        nu, points = self.model.sample_theta_prior()
        theta = self._draw_angle(
            self.abilities, self.difficulties, nu=nu, is_ability=True)
        return self._get_cart_coords(self.abilities, nu=nu, theta=theta)

    def step_difficulty(self):
        nu = self.model.sample_item_prior()
        theta = self._draw_angle(
            self.difficulties, self.abilities, nu=nu, is_ability=False)
        return self._get_cart_coords(self.difficulties, nu=nu, theta=theta)

    def _get_cart_coords(self, input_vec, nu, theta):
        return input_vec * torch.cos(theta) + nu * torch.sin(theta)

    def _draw_angle(self, previous_f, other_f, nu, is_ability):
        """Shrinkage-based angle selection for ESS."""
        if is_ability:
            ll_current = self.model.log_likelihood(
                ability=previous_f, difficulty=other_f,
                disciminatory=1, guessing=0, loading_factor=1)
        else:
            ll_current = self.model.log_likelihood(
                ability=other_f, difficulty=previous_f,
                disciminatory=1, guessing=0, loading_factor=1)
        ll_thres = ll_current + torch.log(torch.rand(1, device=self.device))

        angle = torch.rand(1, device=self.device) * 2 * np.pi
        angle_min, angle_max = angle - 2 * np.pi, angle

        while True:
            next_f = self._get_cart_coords(previous_f, nu, angle)
            if is_ability:
                self.log_likelihood = self.model.log_likelihood(
                    ability=next_f, difficulty=other_f,
                    disciminatory=1, guessing=0, loading_factor=1)
            else:
                self.log_likelihood = self.model.log_likelihood(
                    ability=other_f, difficulty=next_f,
                    disciminatory=1, guessing=0, loading_factor=1)

            if self.log_likelihood >= ll_thres:
                break
            else:
                if angle == 0:
                    break
                if angle < 0:
                    angle_min = angle
                else:
                    angle_max = angle
                angle = (
                    torch.rand(1, device=self.device) * (angle_max - angle_min)
                    + angle_min
                )
        return angle


class BlockedGibbsESSampler:
    """Blocked Gibbs ESS sampler with per-student ability updates.

    Instead of one ESS step for all abilities concatenated (~334K dims),
    this sweeps through each student individually (~50-350 dims each),
    which dramatically improves mixing.

    Each Gibbs iteration:
        1. For each student s: ESS update of ability_s | difficulty, other abilities
        2. One ESS update of difficulty | all abilities
    """

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        # Per-student ability blocks
        self.student_abilities = [None] * model.n_students
        self.difficulties = None
        self.log_likelihood = torch.tensor(0.0)
        self._is_testlet = isinstance(model, GPIRTTestletModelAdapter)

    def _ess_step(self, current, nu, log_lik_fn):
        """Core ESS angle-drawing logic.

        Args:
            current: current parameter vector
            nu: prior sample (proposal direction)
            log_lik_fn: callable(candidate) -> scalar log-likelihood

        Returns:
            Updated parameter vector.
        """
        ll_current = log_lik_fn(current)
        ll_thres = ll_current + torch.log(torch.rand(1, device=self.device))

        angle = torch.rand(1, device=self.device) * 2 * np.pi
        angle_min, angle_max = angle - 2 * np.pi, angle

        while True:
            candidate = current * torch.cos(angle) + nu * torch.sin(angle)
            ll_candidate = log_lik_fn(candidate)

            if ll_candidate >= ll_thres:
                return candidate, ll_candidate
            else:
                if angle == 0:
                    return current, ll_current
                if angle < 0:
                    angle_min = angle
                else:
                    angle_max = angle
                angle = (
                    torch.rand(1, device=self.device) * (angle_max - angle_min)
                    + angle_min
                )

    def _batched_ess_ability_step(self, current_flat, nu_flat, batch_ll_fn,
                                   max_shrink=100):
        """Batched ESS for ability in the flat concatenated representation.

        All students are updated simultaneously. Each student has ONE angle
        that rotates its entire ability vector on the ESS ellipse.

        Args:
            current_flat: (total_ability_dims,) current flat abilities
            nu_flat: (total_ability_dims,) prior sample
            batch_ll_fn: callable(flat_abilities) -> (n_students,) per-student LLs
            max_shrink: maximum bracket shrinking iterations

        Returns:
            (total_ability_dims,) updated flat abilities
        """
        model = self.model
        B = model.n_students
        device = self.device

        # Compute current LL for all students — ONE GPU pass
        ll_cur = batch_ll_fn(current_flat)  # (B,)
        ll_thres = ll_cur + torch.log(torch.rand(B, device=device))

        # Per-student angles
        angle = torch.rand(B, device=device) * 2 * np.pi
        angle_min = angle - 2 * np.pi
        angle_max = angle.clone()

        active = torch.ones(B, dtype=torch.bool, device=device)
        result_flat = current_flat.clone()
        result_lls = ll_cur.clone()

        for _ in range(max_shrink):
            if not active.any():
                break

            # Expand per-student angles to per-dimension
            angle_expanded = angle[model.dim_to_student]  # (total_dims,)

            # Propose candidates for all students
            candidate_flat = (current_flat * torch.cos(angle_expanded)
                              + nu_flat * torch.sin(angle_expanded))

            # Evaluate per-student LLs — ONE GPU pass
            ll_cand = batch_ll_fn(candidate_flat)

            # Accept where LL exceeds threshold and student is still active
            accepted = active & (ll_cand >= ll_thres)

            if accepted.any():
                acc_dims = accepted[model.dim_to_student]
                result_flat[acc_dims] = candidate_flat[acc_dims]
                result_lls[accepted] = ll_cand[accepted]
                active[accepted] = False

            # Shrink brackets for still-active students
            if active.any():
                neg = active & (angle < 0)
                pos = active & (angle >= 0)
                angle_min[neg] = angle[neg]
                angle_max[pos] = angle[pos]

                # Zero-width bracket — keep current value
                zero_bracket = active & (angle_min >= angle_max)
                if zero_bracket.any():
                    active[zero_bracket] = False

                # New angles for remaining active
                rem = active
                if rem.any():
                    angle[rem] = (
                        torch.rand(rem.sum(), device=device)
                        * (angle_max[rem] - angle_min[rem])
                        + angle_min[rem]
                    )

        return result_flat, result_lls

    def _batched_ess_gamma_step(self, current_gamma, nu_gamma, batch_ll_fn,
                                 max_shrink=100):
        """Batched ESS for gamma (fixed-size per student).

        Args:
            current_gamma: (n_students, n_actual_questions) current gamma
            nu_gamma: (n_students, n_actual_questions) prior sample
            batch_ll_fn: callable(gamma_matrix) -> (n_students,) per-student LLs
            max_shrink: maximum bracket shrinking iterations

        Returns:
            (n_students, n_actual_questions) updated gamma
        """
        B = current_gamma.shape[0]
        device = self.device

        ll_cur = batch_ll_fn(current_gamma)
        ll_thres = ll_cur + torch.log(torch.rand(B, device=device))

        angle = torch.rand(B, device=device) * 2 * np.pi
        angle_min = angle - 2 * np.pi
        angle_max = angle.clone()

        active = torch.ones(B, dtype=torch.bool, device=device)
        result_gamma = current_gamma.clone()
        result_lls = ll_cur.clone()

        for _ in range(max_shrink):
            if not active.any():
                break

            cos_a = torch.cos(angle).unsqueeze(1)  # (B, 1)
            sin_a = torch.sin(angle).unsqueeze(1)
            candidate = current_gamma * cos_a + nu_gamma * sin_a

            ll_cand = batch_ll_fn(candidate)

            accepted = active & (ll_cand >= ll_thres)
            if accepted.any():
                result_gamma[accepted] = candidate[accepted]
                result_lls[accepted] = ll_cand[accepted]
                active[accepted] = False

            if active.any():
                neg = active & (angle < 0)
                pos = active & (angle >= 0)
                angle_min[neg] = angle[neg]
                angle_max[pos] = angle[pos]

                zero_bracket = active & (angle_min >= angle_max)
                if zero_bracket.any():
                    active[zero_bracket] = False

                rem = active
                if rem.any():
                    angle[rem] = (
                        torch.rand(rem.sum(), device=device)
                        * (angle_max[rem] - angle_min[rem])
                        + angle_min[rem]
                    )

        return result_gamma, result_lls

    def draw(self, n: int = 1, checkpoint_dir: str = None,
             checkpoint_every: int = 100, thin: int = 1):
        """Draw n samples (each is a full Gibbs sweep).

        Args:
            n: Number of samples to draw.
            checkpoint_dir: If provided, save partial chains every
                ``checkpoint_every`` iterations so runs can be resumed.
            checkpoint_every: How often to write a checkpoint (in iterations).
            thin: Store every ``thin``-th sample in the trace (default 1 = all).
                Reduces RAM and disk usage for large datasets.
        """
        list_ability = []
        list_difficulty = []
        list_gamma = []
        start_iter = 0

        # --- Resume from checkpoint if available ---
        if checkpoint_dir is not None:
            ckpt_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                # Chains are stored/appended as CPU tensors
                list_ability = [t.cpu() for t in ckpt["ability_chain"]]
                list_difficulty = [t.cpu() for t in ckpt["difficulty_chain"]]
                self.student_abilities = [
                    a.to(self.device) for a in ckpt["student_abilities"]
                ]
                self.difficulties = ckpt["difficulties"].to(self.device)
                if self._is_testlet and "gamma" in ckpt:
                    self.model.gamma = ckpt["gamma"].to(self.device)
                    self.model.sigma2 = ckpt["sigma2"].to(self.device)
                if self._is_testlet and "gamma_chain" in ckpt:
                    list_gamma = [t.cpu() for t in ckpt["gamma_chain"]]
                start_iter = ckpt["iteration"]
                thin_ckpt = ckpt.get("thin", 1)
                if thin_ckpt != thin:
                    print(f"WARNING: checkpoint thin={thin_ckpt} != requested thin={thin}; using {thin}")
                print(f"Resumed from checkpoint at iteration {start_iter} "
                      f"({len(list_ability)} stored samples)")

        pbar = tqdm(range(start_iter, n), initial=start_iter, total=n)
        for i in pbar:
            self.step()

            # Store trace only every `thin` iterations
            if (i + 1) % thin == 0:
                list_ability.append(torch.cat(self.student_abilities).cpu())
                list_difficulty.append(self.difficulties.cpu())
                if self._is_testlet:
                    list_gamma.append(self.model.gamma.cpu())

            pbar.set_postfix({"llh": self.log_likelihood.item()})

            # --- Periodic checkpoint ---
            if (checkpoint_dir is not None
                    and (i + 1) % checkpoint_every == 0):
                ckpt_data = {
                    "iteration": i + 1,
                    "thin": thin,
                    "ability_chain": torch.stack(list_ability),
                    "difficulty_chain": torch.stack(list_difficulty),
                    "student_abilities": [a.cpu() for a in self.student_abilities],
                    "difficulties": self.difficulties.cpu(),
                }
                if self._is_testlet:
                    ckpt_data["gamma"] = self.model.gamma.cpu()
                    ckpt_data["sigma2"] = self.model.sigma2.cpu()
                    ckpt_data["gamma_chain"] = torch.stack(list_gamma)
                tmp_path = os.path.join(checkpoint_dir, "checkpoint.pt.tmp")
                torch.save(ckpt_data, tmp_path)
                os.replace(tmp_path, os.path.join(checkpoint_dir, "checkpoint.pt"))

        # Save final checkpoint (so a future --resume sees completion)
        if checkpoint_dir is not None:
            ckpt_data = {
                "iteration": n,
                "thin": thin,
                "ability_chain": torch.stack(list_ability),
                "difficulty_chain": torch.stack(list_difficulty),
                "student_abilities": [a.cpu() for a in self.student_abilities],
                "difficulties": self.difficulties.cpu(),
            }
            if self._is_testlet:
                ckpt_data["gamma"] = self.model.gamma.cpu()
                ckpt_data["sigma2"] = self.model.sigma2.cpu()
                ckpt_data["gamma_chain"] = torch.stack(list_gamma)
            tmp_path = os.path.join(checkpoint_dir, "checkpoint.pt.tmp")
            torch.save(ckpt_data, tmp_path)
            os.replace(tmp_path, os.path.join(checkpoint_dir, "checkpoint.pt"))

        if self._is_testlet:
            return torch.stack(list_ability), torch.stack(list_difficulty), torch.stack(list_gamma)
        return torch.stack(list_ability), torch.stack(list_difficulty)

    def step(self):
        """One full Gibbs sweep: batched ability, per-question difficulty, batched gamma."""
        # Initialize if needed
        if self.student_abilities[0] is None:
            for s in range(self.model.n_students):
                self.student_abilities[s] = self.model.sample_student_prior(s)
        if self.difficulties is None:
            self.difficulties = self.model.sample_item_prior()

        # --- 1. BATCHED ABILITY UPDATE ---
        current_flat = torch.cat(self.student_abilities)  # (total_ability_dims,)
        nu_flat = self.model.batched_sample_ability_prior()  # (total_ability_dims,)

        _diff = self.difficulties
        def ability_batch_ll(af, _d=_diff):
            return self.model.batched_student_ability_ll(af, _d)

        updated_flat, _ = self._batched_ess_ability_step(
            current_flat, nu_flat, ability_batch_ll
        )

        # Scatter back to per-student list (checkpoint compat)
        for s in range(self.model.n_students):
            start = self.model.ability_offsets[s]
            end = self.model.ability_offsets[s + 1]
            self.student_abilities[s] = updated_flat[start:end]

        # --- 2. DIFFICULTY UPDATE (per-question, kept sequential) ---
        if self._is_testlet:
            ab_at_obs = updated_flat[self.model.obs_to_ability_idx]
            gam_at_obs = self.model.gamma[
                self.model.obs_to_student, self.model.obs_to_question
            ]

            for q in range(self.model.n_actual_questions):
                item_idx = self.model.question_item_indices[q]
                if len(item_idx) == 0:
                    continue
                nu_q = torch.randn(len(item_idx), device=self.device)

                def diff_q_ll(dq, _q=q, _ab=ab_at_obs, _gam=gam_at_obs):
                    return self.model.question_difficulty_log_likelihood(
                        _q, dq, _ab, _gam)

                new_dq, _ = self._ess_step(
                    self.difficulties[item_idx], nu_q, diff_q_ll)
                self.difficulties[item_idx] = new_dq

            self.log_likelihood = self.model.log_likelihood(
                ability=updated_flat, difficulty=self.difficulties)
        else:
            nu_diff = self.model.sample_item_prior()

            def diff_ll(diff, _ab=updated_flat):
                return self.model.log_likelihood(ability=_ab, difficulty=diff)

            self.difficulties, self.log_likelihood = self._ess_step(
                self.difficulties, nu_diff, diff_ll
            )

        # --- 3. BATCHED GAMMA UPDATE ---
        if self._is_testlet:
            nu_gamma = self.model.sample_gamma_prior()  # (n_students, n_questions)

            _ab_flat = updated_flat
            _d = self.difficulties
            def gamma_batch_ll(g, _af=_ab_flat, _dd=_d):
                return self.model.batched_student_gamma_ll(g, _af, _dd)

            self.model.gamma, _ = self._batched_ess_gamma_step(
                self.model.gamma, nu_gamma, gamma_batch_ll
            )



if __name__ == "__main__":
    from mcmc_diagnostics import visualize

    parser = argparse.ArgumentParser(description="GPIRT inference with ESS")
    parser.add_argument("--course_name", type=str, default="all",
                        help="Course name or 'all' for combined dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kernel", type=str, default="RBF")
    parser.add_argument("--length_scale", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of posterior samples to draw")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Number of warmup (burn-in) samples to discard")
    parser.add_argument("--n_students", type=int, default=0,
                        help="Limit to first N students (0 = all)")
    parser.add_argument("--thin", type=int, default=1,
                        help="Store every thin-th sample in trace (reduces RAM/disk)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--blocked", action="store_true",
                        help="Use blocked (per-student) ESS for better mixing.")
    parser.add_argument("--testlet", action="store_true",
                        help="Use GPIRT-Testlet model with per-(student,question) "
                             "random effects γ_{sj}. Requires --blocked.")
    args = parser.parse_args()

    if args.testlet and not args.blocked:
        print("--testlet requires --blocked; enabling --blocked automatically.")
        args.blocked = True

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.testlet:
        method_tag = "blocked_ess_testlet"
    elif args.blocked:
        method_tag = "blocked_ess"
    else:
        method_tag = "ess"
    if args.smoke:
        n_stu_tag = "_smoke"
    elif args.n_students > 0:
        n_stu_tag = f"_n{args.n_students}"
    else:
        n_stu_tag = ""
    result_folder = os.path.join(
        REPO_ROOT, "results", "gpirt",
        f"{args.course_name}_s{args.seed}_{method_tag}"
        f"_kernel{args.kernel}_ls{args.length_scale}{n_stu_tag}",
    )
    ensure_dir(result_folder)

    # Download and load data
    if args.course_name == "all":
        # Build combined matrix from raw CSV (all courses, shared items/students)
        import pandas as pd
        import sys
        sys.path.insert(0, os.path.join(REPO_ROOT, "data_collection"))
        from csv2matrices import build_matrices

        matrices_cache = f"{result_folder}/matrices_cache.pt"
        if os.path.exists(matrices_cache):
            print("Loading cached combined matrices...")
            cache = torch.load(matrices_cache, map_location="cpu")
            response_matrix = cache["correctness"].float()
            response_time_matrix = cache["time"].float()
            item_to_question_raw = cache["item_to_question"]
        else:
            csv_folder = snapshot_download(
                repo_id="CodeInsightTeam/code_insights_csv", repo_type="dataset"
            )
            main_data = pd.read_csv(
                os.path.join(csv_folder, "main_data.csv"), low_memory=False
            )
            question_infos = pd.read_csv(
                os.path.join(csv_folder, "question_infos.csv")
            )
            main_data = main_data[
                main_data["response_type"].isin(["Submit", "Prechecked"])
            ].copy()
            main_data = main_data.dropna(subset=["pass"])
            main_data = main_data[main_data["pass"].astype(str).str.strip() != ""]
            print(f"Loaded {len(main_data)} submissions across all courses")

            _, question_info_list, correctness, time_mat, _ = build_matrices(
                main_data, question_infos, "all", "cpu"
            )
            # Build item_to_question from question_info_list
            itq = torch.tensor(
                [q["qidx"] for q in question_info_list], dtype=torch.long
            )
            torch.save(
                {"correctness": correctness, "time": time_mat,
                 "item_to_question": itq},
                matrices_cache,
            )
            response_matrix = correctness.float()
            response_time_matrix = time_mat.float()
            item_to_question_raw = itq
    else:
        # Load pre-built per-course matrices
        data_folder = snapshot_download(
            repo_id="CodeInsightTeam/code_insights_matrices", repo_type="dataset"
        )
        data_folder = os.path.join(data_folder, args.course_name)

        response_matrix = torch.load(f"{data_folder}/correctness_matrix.pt").float()
        response_time_matrix = torch.load(f"{data_folder}/time_matrix.pt").float()
        item_to_question_raw = None

    if args.smoke:
        response_matrix = response_matrix[:2]
        response_time_matrix = response_time_matrix[:2]
    elif args.n_students > 0:
        response_matrix = response_matrix[:args.n_students]
        response_time_matrix = response_time_matrix[:args.n_students]

    n_students = response_matrix.shape[0]
    print(f"Data: {n_students} students x {response_matrix.shape[1]} items x {response_matrix.shape[2]} max_attempts")

    # Preprocess (build GP priors and index structures)
    cache_path = f"{result_folder}/all_indexes.pkl"
    if os.path.exists(cache_path):
        print("Loading cached preprocessing...")
        all_indexes = pickle.load(open(cache_path, "rb"))
    else:
        print("Preprocessing...")
        with torch.no_grad():
            all_indexes = preprocess(
                response_matrix=response_matrix,
                response_time_matrix=response_time_matrix,
                low_rank_configs={
                    "type": "GP",
                    "kernel": args.kernel,
                    "length_scale": args.length_scale,
                },
                device=device,
            )
        pickle.dump(all_indexes, open(cache_path, "wb"))

    # Build item-to-question mapping (for testlet model)
    item_to_question = None
    if args.testlet:
        itq_cache = f"{result_folder}/item_to_question.pt"
        if args.course_name == "all" and item_to_question_raw is not None:
            # Already built during matrix construction
            item_to_question = item_to_question_raw.to(device)
            torch.save(item_to_question, itq_cache)
            print(f"item_to_question from combined build "
                  f"({item_to_question.max().item()+1} questions)")
        elif os.path.exists(itq_cache):
            item_to_question = torch.load(itq_cache, map_location=device)
            print(f"Loaded item_to_question from cache ({item_to_question.max().item()+1} questions)")
        else:
            print("Building item-to-question mapping from CSV...")
            # build_item_to_question is defined above in this file
            item_to_question = build_item_to_question(args.course_name).to(device)
            torch.save(item_to_question, itq_cache)
            print(f"  {item_to_question.shape[0]} items → {item_to_question.max().item()+1} questions")

    # Build model adapter and sampler
    if args.testlet:
        model = GPIRTTestletModelAdapter(response_matrix, all_indexes, device, item_to_question)
        print(f"Using GPIRTTestletModelAdapter "
              f"({model.n_students} students, {model.n_actual_questions} questions, "
              f"{model.n_questions} test-case items)")
    else:
        model = GPIRTModelAdapter(response_matrix, all_indexes, device)

    if args.blocked:
        sampler = BlockedGibbsESSampler(model, device=device)
        print("Using BlockedGibbsESSampler (per-student ability updates)")
    else:
        sampler = GibbsESSampler(model, device=device)
        print("Using GibbsESSampler (joint ability updates)")

    total_samples = args.warmup + args.n_samples
    print(f"\nRunning: {args.warmup} warmup + {args.n_samples} posterior samples")

    # Draw all samples (warmup + posterior), with checkpointing for resume
    thin = args.thin
    if thin > 1:
        print(f"Trace thinning: storing every {thin}-th sample")
    draw_result = sampler.draw(
        n=total_samples,
        checkpoint_dir=result_folder if args.blocked else None,
        checkpoint_every=500,
        thin=thin,
    )

    if args.testlet:
        all_abilities, all_difficulties, all_gammas = draw_result
    else:
        all_abilities, all_difficulties = draw_result
        all_gammas = None

    # Discard warmup (account for thinning: stored indices are original_iter // thin)
    warmup_stored = args.warmup // thin
    ability_samples = all_abilities[warmup_stored:]   # (n_post_warmup, total_ability_dim)
    difficulty_samples = all_difficulties[warmup_stored:]  # (n_post_warmup, n_questions)
    print(f"Total stored samples: {len(all_abilities)}, discarding {warmup_stored} warmup")

    # Split ability samples back into per-student tensors
    print("Saving parameters...")
    ability = []
    offset = 0
    for seg_size in model.segment_sizes:
        ability.append(ability_samples[:, offset:offset + seg_size])
        offset += seg_size

    torch.save(ability, f"{result_folder}/ability.pt")
    torch.save(difficulty_samples, f"{result_folder}/difficulty.pt")

    # Save full chain (including warmup) for diagnostics
    torch.save(all_abilities, f"{result_folder}/ability_chain.pt")
    torch.save(all_difficulties, f"{result_folder}/difficulty_chain.pt")

    if args.testlet:
        gamma_samples = all_gammas[warmup_stored:]
        torch.save(gamma_samples, f"{result_folder}/gamma.pt")
        torch.save(model.sigma2.cpu(), f"{result_folder}/sigma2.pt")
        torch.save(all_gammas, f"{result_folder}/gamma_chain.pt")
        print(f"  gamma.pt: shape {gamma_samples.shape}")
        print(f"  sigma2.pt: shape {model.sigma2.shape}")

    n_posterior = len(ability_samples)
    print(f"\nResults saved to {result_folder}/")
    print(f"  ability.pt: {len(ability)} students, {n_posterior} posterior samples each")
    print(f"  difficulty.pt: shape {difficulty_samples.shape}")

    # Save plots to shared results directory
    shared_result_dir = os.path.join(REPO_ROOT, "results", "gpirt", method_tag)
    ensure_dir(shared_result_dir)

    print(f"\nGenerating plots to {shared_result_dir}/...")
    visualize(
        ability_samples.cpu(), difficulty_samples.cpu(),
        all_abilities.cpu(), all_difficulties.cpu(),
        model, shared_result_dir,
    )

 
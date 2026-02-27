"""
GPIRT inference using Elliptical Slice Sampling (ESS).

ESS is a gradient-free MCMC method well-suited for models with Gaussian priors
(like GP-IRT). It avoids the costly gradient computation of HMC/NUTS and scales
better to large numbers of students.

Usage:
    cd CodeInsights/dynamic_irt/gpirt
    CUDA_VISIBLE_DEVICES=3 python inference_ess.py --course_name dsa_hk231 --n_samples 500 --warmup 100
    CUDA_VISIBLE_DEVICES=3 python inference_ess.py --course_name dsa_hk231 --n_students 200 --n_samples 200 --warmup 50
"""

import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import snapshot_download
from ess.src.ess import GibbsESSampler
from tueplots import bundles, figsizes
from utils import ensure_dir, preprocess, set_seed

plt.rcParams.update(bundles.aaai2024())

# Standardized color palette (Paul Tol qualitative)
COLORS = ["#4477aa", "#ee6677", "#228833", "#aa3377", "#ccbb44"]

# Repo root (CodeInsights/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


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


# ---------------------------------------------------------------------------
# MCMC Diagnostics
# ---------------------------------------------------------------------------

def _autocorr(x, max_lag=None):
    """Compute autocorrelation for a 1-D chain using FFT."""
    n = len(x)
    if max_lag is None:
        max_lag = n
    x = x - x.mean()
    # FFT-based autocorrelation
    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real
    acf /= acf[0] if acf[0] != 0 else 1.0
    return acf[:max_lag]


def effective_sample_size(chain):
    """Estimate ESS from a 1-D chain using initial monotone sequence estimator."""
    n = len(chain)
    acf = _autocorr(chain)
    # Sum pairs of autocorrelations until they become negative
    # (Geyer's initial positive sequence estimator)
    tau = 1.0
    for lag in range(1, n // 2):
        pair_sum = acf[2 * lag - 1] + acf[2 * lag] if 2 * lag < n else 0
        if pair_sum < 0:
            break
        tau += 2 * pair_sum
    return n / tau


def _rhat_from_chains(chains):
    """Compute R-hat from a list of 1-D chains (Gelman-Rubin diagnostic).

    Args:
        chains: list of 1-D numpy arrays (one per chain), all same length.

    Returns:
        R-hat statistic. < 1.01 is excellent, < 1.1 is acceptable.
    """
    m = len(chains)
    n = len(chains[0])
    chain_means = np.array([c.mean() for c in chains])
    chain_vars = np.array([c.var(ddof=1) for c in chains])

    # Within-chain variance
    W = chain_vars.mean()
    # Between-chain variance
    grand_mean = chain_means.mean()
    B = n * np.var(chain_means, ddof=1)
    # Marginal posterior variance estimate
    var_hat = ((n - 1) / n) * W + (1.0 / n) * B

    if W == 0:
        return float('nan')
    return np.sqrt(var_hat / W)


def split_rhat(chain):
    """Compute split-R-hat from a single chain (split in half)."""
    n = len(chain)
    mid = n // 2
    return _rhat_from_chains([chain[:mid], chain[mid:2 * mid]])


def multi_chain_rhat(chains):
    """Compute R-hat from multiple independent chains.

    Each chain is split in half, giving 2*M sub-chains for M chains.
    This is the recommended approach from BDA3 (Gelman et al.).
    """
    sub_chains = []
    for chain in chains:
        mid = len(chain) // 2
        sub_chains.append(chain[:mid])
        sub_chains.append(chain[mid:2 * mid])
    return _rhat_from_chains(sub_chains)


def compute_diagnostics(posterior_samples, param_names=None):
    """Compute R-hat, ESS, mean, std for each parameter column.

    Args:
        posterior_samples: (n_samples, n_params) numpy array
        param_names: optional list of names

    Returns:
        List of dicts with diagnostics per parameter.
    """
    n_samples, n_params = posterior_samples.shape
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]

    results = []
    for j in range(n_params):
        chain = posterior_samples[:, j]
        results.append({
            "name": param_names[j],
            "mean": chain.mean(),
            "std": chain.std(ddof=1),
            "rhat": split_rhat(chain),
            "ess": effective_sample_size(chain),
        })
    return results


def plot_trace(chains_dict, result_dir, warmup=0):
    """Plot trace + posterior density for selected parameters.

    Args:
        chains_dict: dict mapping param_name -> 1-D numpy array (full chain incl. warmup)
        result_dir: output directory
        warmup: number of warmup samples to shade
    """
    n_params = len(chains_dict)
    fig, axes = plt.subplots(n_params, 2, figsize=(
        figsizes.aaai2024_full()["figure.figsize"][0],
        1.5 * n_params,
    ))
    if n_params == 1:
        axes = axes[np.newaxis, :]

    for i, (name, chain) in enumerate(chains_dict.items()):
        # Trace plot
        ax_trace = axes[i, 0]
        ax_trace.plot(chain, alpha=0.6, linewidth=0.5, color=COLORS[0])
        if warmup > 0:
            ax_trace.axvspan(0, warmup, alpha=0.15, color=COLORS[1], label="Warmup")
        ax_trace.set_ylabel(name)
        if i == 0:
            ax_trace.set_title("Trace")
        if i == n_params - 1:
            ax_trace.set_xlabel("Sample")

        # Posterior density (post-warmup only)
        ax_hist = axes[i, 1]
        posterior = chain[warmup:]
        ax_hist.hist(posterior, bins=40, density=True, alpha=0.3, color=COLORS[0])
        sns.kdeplot(posterior, color=COLORS[0], linewidth=1.5, ax=ax_hist)
        if i == 0:
            ax_hist.set_title("Posterior")

    fig.tight_layout()
    save_path = os.path.join(result_dir, "trace_plots.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_autocorr(chains_dict, result_dir, max_lag=100):
    """Plot autocorrelation functions for selected parameters."""
    n_params = len(chains_dict)
    fig, axes = plt.subplots(1, n_params, figsize=(
        figsizes.aaai2024_full()["figure.figsize"][0],
        figsizes.aaai2024_half()["figure.figsize"][1],
    ))
    if n_params == 1:
        axes = [axes]

    for i, (name, chain) in enumerate(chains_dict.items()):
        acf = _autocorr(chain, max_lag=max_lag)
        axes[i].bar(range(len(acf)), acf, width=1.0, alpha=0.5, color=COLORS[0])
        axes[i].axhline(0, color="black", linewidth=0.5)
        axes[i].set_xlabel("Lag")
        axes[i].set_title(name)
        if i == 0:
            axes[i].set_ylabel("ACF")

    fig.tight_layout()
    save_path = os.path.join(result_dir, "autocorrelation.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def run_diagnostics(ability_chain, difficulty_chain, segment_sizes,
                    result_dir, warmup=100, gamma_chain=None):
    """Run full posterior diagnostics and save results.

    Args:
        ability_chain: (n_total, total_ability_dim) full chain including warmup
        difficulty_chain: (n_total, n_questions) full chain including warmup
        segment_sizes: list of per-student ability dimensions
        result_dir: output directory
        warmup: number of warmup samples
    """
    import pandas as pd

    posterior_ability = ability_chain[warmup:]    # (n_post, total_ability_dim)
    posterior_diff = difficulty_chain[warmup:]    # (n_post, n_questions)
    n_post = posterior_ability.shape[0]
    n_questions = posterior_diff.shape[1]
    n_students = len(segment_sizes)

    print(f"\n{'='*60}")
    print(f"MCMC Diagnostics ({n_post} posterior samples)")
    print(f"{'='*60}")

    # --- Difficulty diagnostics ---
    diff_names = [f"beta_{j}" for j in range(n_questions)]
    diff_diag = compute_diagnostics(posterior_diff, diff_names)

    rhats_diff = np.array([d["rhat"] for d in diff_diag])
    ess_diff = np.array([d["ess"] for d in diff_diag])

    print(f"\nDifficulty parameters (n={n_questions}):")
    print(f"  R-hat: median={np.nanmedian(rhats_diff):.4f}, "
          f"max={np.nanmax(rhats_diff):.4f}, "
          f"pct>1.1={100*np.nanmean(rhats_diff > 1.1):.1f}%")
    print(f"  ESS:   median={np.nanmedian(ess_diff):.1f}, "
          f"min={np.nanmin(ess_diff):.1f}, "
          f"pct<100={100*np.nanmean(ess_diff < 100):.1f}%")

    # --- Per-student ability diagnostics (posterior mean per student at each step) ---
    # For each student, compute the mean ability at each MCMC step
    student_mean_chains = []
    offset = 0
    for s, seg_size in enumerate(segment_sizes):
        student_chain = posterior_ability[:, offset:offset + seg_size]  # (n_post, seg_size)
        student_mean_chains.append(student_chain.mean(axis=1))  # (n_post,)
        offset += seg_size
    student_mean_chains = np.stack(student_mean_chains, axis=1)  # (n_post, n_students)

    ability_names = [f"theta_mean_{s}" for s in range(n_students)]
    ability_diag = compute_diagnostics(student_mean_chains, ability_names)

    rhats_ab = np.array([d["rhat"] for d in ability_diag])
    ess_ab = np.array([d["ess"] for d in ability_diag])

    print(f"\nAbility parameters (n={n_students} students, mean over time):")
    print(f"  R-hat: median={np.nanmedian(rhats_ab):.4f}, "
          f"max={np.nanmax(rhats_ab):.4f}, "
          f"pct>1.1={100*np.nanmean(rhats_ab > 1.1):.1f}%")
    print(f"  ESS:   median={np.nanmedian(ess_ab):.1f}, "
          f"min={np.nanmin(ess_ab):.1f}, "
          f"pct<100={100*np.nanmean(ess_ab < 100):.1f}%")

    # --- Testlet effect diagnostics ---
    gamma_diag = []
    if gamma_chain is not None:
        posterior_gamma = gamma_chain[warmup:]  # (n_post, n_students, n_actual_questions)
        n_students_g, n_q_g = posterior_gamma.shape[1], posterior_gamma.shape[2]
        # Sample 10 random (student, question) pairs for diagnostics
        rng = np.random.default_rng(0)
        n_sample = min(10, n_students_g * n_q_g)
        s_idxs = rng.integers(0, n_students_g, size=n_sample)
        q_idxs = rng.integers(0, n_q_g, size=n_sample)
        gamma_sample = posterior_gamma[:, s_idxs, q_idxs]  # (n_post, n_sample)
        gamma_names = [f"gamma_s{s}_q{q}" for s, q in zip(s_idxs, q_idxs)]
        gamma_diag = compute_diagnostics(gamma_sample, gamma_names)

        rhats_g = np.array([d["rhat"] for d in gamma_diag])
        ess_g = np.array([d["ess"] for d in gamma_diag])
        print(f"\nTestlet effects γ (sample of {n_sample} s×q pairs):")
        print(f"  R-hat: median={np.nanmedian(rhats_g):.4f}, "
              f"max={np.nanmax(rhats_g):.4f}, "
              f"pct>1.1={100*np.nanmean(rhats_g > 1.1):.1f}%")
        print(f"  ESS:   median={np.nanmedian(ess_g):.1f}, "
              f"min={np.nanmin(ess_g):.1f}, "
              f"pct<50={100*np.nanmean(ess_g < 50):.1f}%")

    # --- Save summary CSV ---
    summary_rows = []
    for d in diff_diag:
        summary_rows.append({**d, "type": "difficulty"})
    for d in ability_diag:
        summary_rows.append({**d, "type": "ability_mean"})
    for d in gamma_diag:
        summary_rows.append({**d, "type": "testlet_effect"})
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(result_dir, "diagnostics_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n  Diagnostics CSV: {csv_path}")

    # --- Trace plots for select parameters ---
    # Pick a few representative difficulty and ability chains
    trace_params = {}
    # 3 difficulties: easy (min mean), medium (median), hard (max mean)
    diff_means = posterior_diff.mean(axis=0)
    for tag, idx in [("easiest", np.argmin(diff_means)),
                     ("median", np.argsort(diff_means)[n_questions // 2]),
                     ("hardest", np.argmax(diff_means))]:
        trace_params[f"beta_{tag}"] = difficulty_chain[:, idx]

    # 2 student abilities (most/least variable)
    ab_stds = student_mean_chains.std(axis=0)
    for tag, idx in [("most_var", np.argmax(ab_stds)),
                     ("least_var", np.argmin(ab_stds))]:
        trace_params[f"theta_{tag}"] = np.concatenate([
            ability_chain[:warmup, :],  # dummy — we need full chain for student means
        ], axis=0) if False else None

    # Actually compute full-chain student means for trace
    full_student_means = []
    offset = 0
    for seg_size in segment_sizes:
        student_chain = ability_chain[:, offset:offset + seg_size]
        full_student_means.append(student_chain.mean(axis=1))
        offset += seg_size
    full_student_means = np.stack(full_student_means, axis=1)

    trace_params[r"$\bar{\theta}$ (most variable)"] = full_student_means[:, np.argmax(ab_stds)]
    trace_params[r"$\bar{\theta}$ (least variable)"] = full_student_means[:, np.argmin(ab_stds)]

    # Remove the None placeholders
    trace_params = {k: v for k, v in trace_params.items() if v is not None}

    plot_trace(trace_params, result_dir, warmup=warmup)

    # --- Autocorrelation plots (post-warmup only) ---
    acf_params = {}
    for tag, idx in [("easiest", np.argmin(diff_means)),
                     ("hardest", np.argmax(diff_means))]:
        acf_params[f"beta_{tag}"] = posterior_diff[:, idx]
    acf_params[r"$\bar{\theta}$ (most var)"] = student_mean_chains[:, np.argmax(ab_stds)]
    acf_params[r"$\bar{\theta}$ (least var)"] = student_mean_chains[:, np.argmin(ab_stds)]

    plot_autocorr(acf_params, result_dir, max_lag=min(100, n_post // 2))

    # --- R-hat distribution plot ---
    fig, axes = plt.subplots(1, 2, figsize=figsizes.aaai2024_full()["figure.figsize"])

    axes[0].hist(rhats_diff, bins=40, alpha=0.5, color=COLORS[0], density=True)
    axes[0].axvline(1.1, color=COLORS[1], linestyle="--", linewidth=1, label=r"$\hat{R}=1.1$")
    axes[0].set_xlabel(r"Split-$\hat{R}$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"$\beta$ (Difficulty)")
    axes[0].legend()

    axes[1].hist(rhats_ab, bins=40, alpha=0.5, color=COLORS[2], density=True)
    axes[1].axvline(1.1, color=COLORS[1], linestyle="--", linewidth=1, label=r"$\hat{R}=1.1$")
    axes[1].set_xlabel(r"Split-$\hat{R}$")
    axes[1].set_title(r"$\bar{\theta}$ (Mean Ability)")
    axes[1].legend()

    fig.tight_layout()
    save_path = os.path.join(result_dir, "rhat_distribution.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")

    # --- ESS distribution plot ---
    fig, axes = plt.subplots(1, 2, figsize=figsizes.aaai2024_full()["figure.figsize"])

    axes[0].hist(ess_diff, bins=40, alpha=0.5, color=COLORS[0], density=True)
    axes[0].axvline(100, color=COLORS[1], linestyle="--", linewidth=1, label="ESS=100")
    axes[0].set_xlabel("ESS")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"$\beta$ (Difficulty)")
    axes[0].legend()

    axes[1].hist(ess_ab, bins=40, alpha=0.5, color=COLORS[2], density=True)
    axes[1].axvline(100, color=COLORS[1], linestyle="--", linewidth=1, label="ESS=100")
    axes[1].set_xlabel("ESS")
    axes[1].set_title(r"$\bar{\theta}$ (Mean Ability)")
    axes[1].legend()

    fig.tight_layout()
    save_path = os.path.join(result_dir, "ess_distribution.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")

    print(f"{'='*60}\n")


def run_multichain_diagnostics(chain_folders, segment_sizes, result_dir, warmup=200):
    """Run multi-chain diagnostics from saved chain files.

    Args:
        chain_folders: list of result folder paths, each containing
            ability_chain.pt and difficulty_chain.pt
        segment_sizes: list of per-student ability dimensions
        result_dir: output directory for plots and CSVs
        warmup: number of warmup samples per chain
    """
    import pandas as pd

    n_chains = len(chain_folders)
    print(f"\n{'='*60}")
    print(f"Multi-Chain Diagnostics ({n_chains} chains)")
    print(f"{'='*60}")

    # Load all chains
    diff_chains = []  # list of (n_post, n_questions) arrays
    ability_mean_chains = []  # list of (n_post, n_students) arrays

    for folder in chain_folders:
        diff_chain = torch.load(f"{folder}/difficulty_chain.pt", map_location="cpu")
        ab_chain = torch.load(f"{folder}/ability_chain.pt", map_location="cpu")

        # Discard warmup
        diff_post = diff_chain[warmup:].numpy()
        ab_post = ab_chain[warmup:].numpy()

        diff_chains.append(diff_post)

        # Compute per-student mean ability at each MCMC step
        student_means = []
        offset = 0
        for seg_size in segment_sizes:
            student_means.append(ab_post[:, offset:offset + seg_size].mean(axis=1))
            offset += seg_size
        ability_mean_chains.append(np.stack(student_means, axis=1))

        print(f"  {folder}: {diff_post.shape[0]} posterior samples")

    n_post = diff_chains[0].shape[0]
    n_questions = diff_chains[0].shape[1]
    n_students = ability_mean_chains[0].shape[1]

    # --- Multi-chain R-hat and ESS ---
    print(f"\nComputing multi-chain R-hat and ESS...")

    # Difficulty
    rhats_diff = np.zeros(n_questions)
    ess_diff = np.zeros(n_questions)
    for j in range(n_questions):
        param_chains = [dc[:, j] for dc in diff_chains]
        rhats_diff[j] = multi_chain_rhat(param_chains)
        # Total ESS across chains
        ess_diff[j] = sum(effective_sample_size(c) for c in param_chains)

    print(f"\nDifficulty parameters (n={n_questions}):")
    print(f"  R-hat: median={np.nanmedian(rhats_diff):.4f}, "
          f"max={np.nanmax(rhats_diff):.4f}, "
          f"pct>1.1={100*np.nanmean(rhats_diff > 1.1):.1f}%")
    print(f"  ESS:   median={np.nanmedian(ess_diff):.1f}, "
          f"min={np.nanmin(ess_diff):.1f}, "
          f"pct<100={100*np.nanmean(ess_diff < 100):.1f}%")

    # Ability
    rhats_ab = np.zeros(n_students)
    ess_ab = np.zeros(n_students)
    for s in range(n_students):
        param_chains = [ac[:, s] for ac in ability_mean_chains]
        rhats_ab[s] = multi_chain_rhat(param_chains)
        ess_ab[s] = sum(effective_sample_size(c) for c in param_chains)

    print(f"\nAbility parameters (n={n_students} students, mean over time):")
    print(f"  R-hat: median={np.nanmedian(rhats_ab):.4f}, "
          f"max={np.nanmax(rhats_ab):.4f}, "
          f"pct>1.1={100*np.nanmean(rhats_ab > 1.1):.1f}%")
    print(f"  ESS:   median={np.nanmedian(ess_ab):.1f}, "
          f"min={np.nanmin(ess_ab):.1f}, "
          f"pct<100={100*np.nanmean(ess_ab < 100):.1f}%")

    # --- Save summary CSV ---
    summary_rows = []
    for j in range(n_questions):
        summary_rows.append({
            "name": f"beta_{j}", "type": "difficulty",
            "rhat": rhats_diff[j], "ess": ess_diff[j],
            "mean": np.mean([dc[:, j].mean() for dc in diff_chains]),
            "std": np.mean([dc[:, j].std() for dc in diff_chains]),
        })
    for s in range(n_students):
        summary_rows.append({
            "name": f"theta_mean_{s}", "type": "ability_mean",
            "rhat": rhats_ab[s], "ess": ess_ab[s],
            "mean": np.mean([ac[:, s].mean() for ac in ability_mean_chains]),
            "std": np.mean([ac[:, s].std() for ac in ability_mean_chains]),
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(result_dir, "multichain_diagnostics.csv"), index=False)
    print(f"\n  CSV: {os.path.join(result_dir, 'multichain_diagnostics.csv')}")

    # --- Trace overlay plot: show all chains for select parameters ---
    diff_means_overall = np.mean([dc.mean(axis=0) for dc in diff_chains], axis=0)
    representative_diff = {
        "easiest": np.argmin(diff_means_overall),
        "median": np.argsort(diff_means_overall)[n_questions // 2],
        "hardest": np.argmax(diff_means_overall),
    }

    ab_stds_overall = np.mean([ac.std(axis=0) for ac in ability_mean_chains], axis=0)
    representative_ab = {
        "most_var": np.argmax(ab_stds_overall),
        "least_var": np.argmin(ab_stds_overall),
    }

    n_trace = len(representative_diff) + len(representative_ab)
    fig, axes = plt.subplots(n_trace, 2, figsize=(
        figsizes.aaai2024_full()["figure.figsize"][0],
        1.5 * n_trace,
    ))

    row = 0
    for tag, idx in representative_diff.items():
        ax_trace, ax_hist = axes[row, 0], axes[row, 1]
        for c_i, dc in enumerate(diff_chains):
            full_chain = torch.load(
                f"{chain_folders[c_i]}/difficulty_chain.pt", map_location="cpu"
            ).numpy()[:, idx]
            ax_trace.plot(full_chain, alpha=0.5, linewidth=0.4,
                          color=COLORS[c_i % len(COLORS)],
                          label=f"Chain {c_i+1}")
            ax_hist.hist(dc[:, idx], bins=30, density=True, alpha=0.2,
                         color=COLORS[c_i % len(COLORS)])
        ax_trace.axvspan(0, warmup, alpha=0.1, color="gray")
        ax_trace.set_ylabel(f"beta_{tag}")
        ax_hist.set_title(f"R-hat={rhats_diff[idx]:.3f}" if row == 0 else
                          f"R-hat={rhats_diff[idx]:.3f}")
        if row == 0:
            ax_trace.set_title("Trace (all chains)")
            ax_trace.legend(fontsize=5)
        row += 1

    for tag, idx in representative_ab.items():
        ax_trace, ax_hist = axes[row, 0], axes[row, 1]
        for c_i, ac in enumerate(ability_mean_chains):
            # Reconstruct full chain means
            ab_chain = torch.load(
                f"{chain_folders[c_i]}/ability_chain.pt", map_location="cpu"
            ).numpy()
            offset = sum(segment_sizes[:idx])
            full_student = ab_chain[:, offset:offset + segment_sizes[idx]].mean(axis=1)
            ax_trace.plot(full_student, alpha=0.5, linewidth=0.4,
                          color=COLORS[c_i % len(COLORS)])
            ax_hist.hist(ac[:, idx], bins=30, density=True, alpha=0.2,
                         color=COLORS[c_i % len(COLORS)])
        ax_trace.axvspan(0, warmup, alpha=0.1, color="gray")
        ax_trace.set_ylabel(f"theta_{tag}")
        ax_hist.set_title(f"R-hat={rhats_ab[idx]:.3f}")
        row += 1

    axes[-1, 0].set_xlabel("Sample")
    fig.tight_layout()
    save_path = os.path.join(result_dir, "multichain_traces.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")

    # --- R-hat distribution plot ---
    fig, axes = plt.subplots(1, 2, figsize=figsizes.aaai2024_full()["figure.figsize"])

    axes[0].hist(rhats_diff, bins=50, alpha=0.5, color=COLORS[0], density=True)
    axes[0].axvline(1.1, color=COLORS[1], linestyle="--", linewidth=1, label=r"$\hat{R}=1.1$")
    axes[0].set_xlabel(r"$\hat{R}$")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"$\beta$ (Difficulty)")
    axes[0].legend()

    axes[1].hist(rhats_ab, bins=50, alpha=0.5, color=COLORS[2], density=True)
    axes[1].axvline(1.1, color=COLORS[1], linestyle="--", linewidth=1, label=r"$\hat{R}=1.1$")
    axes[1].set_xlabel(r"$\hat{R}$")
    axes[1].set_title(r"$\bar{\theta}$ (Mean Ability)")
    axes[1].legend()

    fig.tight_layout()
    save_path = os.path.join(result_dir, "multichain_rhat.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")

    # --- ESS distribution plot ---
    fig, axes = plt.subplots(1, 2, figsize=figsizes.aaai2024_full()["figure.figsize"])

    axes[0].hist(ess_diff, bins=50, alpha=0.5, color=COLORS[0], density=True)
    axes[0].axvline(100, color=COLORS[1], linestyle="--", linewidth=1, label="ESS=100")
    axes[0].set_xlabel("ESS (total across chains)")
    axes[0].set_ylabel("Density")
    axes[0].set_title(r"$\beta$ (Difficulty)")
    axes[0].legend()

    axes[1].hist(ess_ab, bins=50, alpha=0.5, color=COLORS[2], density=True)
    axes[1].axvline(100, color=COLORS[1], linestyle="--", linewidth=1, label="ESS=100")
    axes[1].set_xlabel("ESS (total across chains)")
    axes[1].set_title(r"$\bar{\theta}$ (Mean Ability)")
    axes[1].legend()

    fig.tight_layout()
    save_path = os.path.join(result_dir, "multichain_ess.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")

    print(f"{'='*60}\n")

    return rhats_diff, rhats_ab, ess_diff, ess_ab


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_param_hist(values, param_name, filename, result_dir, xlabel,
                    bins=30, pct_clip=(1, 99)):
    """Plot histogram + KDE, clipping x-axis to the given percentile range."""
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


def plot_trajectories(ability_means, segment_sizes, ability_prior_dists,
                      result_dir, n_top=5):
    """Plot posterior mean ability trajectories for top students by final ability."""
    # Compute final ability for each student
    final_abilities = []
    offset = 0
    for seg_size in segment_sizes:
        student_traj = ability_means[offset:offset + seg_size]
        final_abilities.append(student_traj[-1])
        offset += seg_size
    final_abilities = np.array(final_abilities)

    # Pick top students by final ability
    top_idx = np.argsort(final_abilities)[::-1][:n_top]

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
    offset = 0
    student_i = 0
    color_i = 0
    for seg_size in segment_sizes:
        if student_i in top_idx:
            traj = ability_means[offset:offset + seg_size]
            # Use the GP prior's time points as x-axis
            time_points = ability_prior_dists[student_i].loc.cpu().numpy()
            t_axis = np.arange(len(traj)) if len(time_points) != len(traj) else np.arange(len(traj))
            ax.plot(t_axis, traj, color=COLORS[color_i % len(COLORS)],
                    label=f"Student {student_i}")
            color_i += 1
        offset += seg_size
        student_i += 1

    ax.set_xlabel("Time index")
    ax.set_ylabel(r"$\theta(t)$")
    ax.set_title(r"Ability Trajectories: Top Students (Posterior Mean)")
    ax.legend()
    save_path = os.path.join(result_dir, "ability_trajectories.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def plot_log_likelihood_chain(all_abilities, all_difficulties, model, result_dir):
    """Plot log-likelihood over the full MCMC chain (warmup + posterior)."""
    n_total = all_abilities.shape[0]
    llhs = []
    for i in range(n_total):
        with torch.no_grad():
            ll = model.log_likelihood(
                all_abilities[i].to(model.device),
                all_difficulties[i].to(model.device),
            )
        llhs.append(ll.item())

    fig, ax = plt.subplots(figsize=figsizes.aaai2024_full()["figure.figsize"])
    ax.plot(llhs, alpha=0.7, color=COLORS[0])
    ax.set_xlabel("Sample")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("GPIRT ESS: Log-likelihood Chain")
    ax.grid(True, alpha=0.3)
    save_path = os.path.join(result_dir, "log_likelihood_chain.png")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {save_path}")


def visualize(ability_samples, difficulty_samples, all_abilities, all_difficulties,
              model, result_dir):
    """Generate all GPIRT plots."""
    # Posterior mean difficulty
    diff_mean = difficulty_samples.mean(dim=0).numpy()
    plot_param_hist(diff_mean, r"$\beta$", "difficulty", result_dir,
                    r"$\beta$ (Item Difficulty)", bins=80)

    # Posterior mean ability (concatenated across students)
    ability_mean = ability_samples.mean(dim=0).numpy()
    plot_param_hist(ability_mean, r"$\theta$", "ability", result_dir,
                    r"$\theta$ (Ability)", bins=80, pct_clip=(5, 95))

    # Trajectories
    plot_trajectories(ability_mean, model.segment_sizes,
                      model.ability_prior_dists, result_dir)

    # Log-likelihood chain
    plot_log_likelihood_chain(all_abilities, all_difficulties, model, result_dir)


if __name__ == "__main__":
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
    result_folder = (
        f"results/{args.course_name}_s{args.seed}_{method_tag}"
        f"_kernel{args.kernel}_ls{args.length_scale}{n_stu_tag}"
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
                repo_id="stair-lab/code_insights_csv", repo_type="dataset"
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
            repo_id="stair-lab/code_insights_matrices", repo_type="dataset"
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
            from utils import build_item_to_question
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

 
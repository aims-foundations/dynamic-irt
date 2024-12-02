import numpy as np
import torch
from torch.distributions.bernoulli import Bernoulli


class IRTLikelihood:
    def __init__(self, device):
        self.device = device

    def log_likelihood(self, theta, z, y):
        probs = torch.sigmoid(theta + z)
        likelihood_dist = Bernoulli(probs=probs)
        return likelihood_dist.log_prob(y).sum()


class GibbsESSampler:
    def __init__(
        self,
        likelihood,
        theta_prior_dists,
        z_prior_dists,
        y_train,
        train_test_split_idx,
        list_saidx,
        all_squidx,
        student_idxs,
        list_saidx2aidx,
        unique_time_obs,
        device,
        n_points=100,
    ):
        self.device = device
        self.theta_prior_dists = theta_prior_dists
        self.z_prior_dists = z_prior_dists
        self.n_students = len(theta_prior_dists)
        self.n_testcases = z_prior_dists.loc.shape[0]
        self.y_train = y_train
        self.train_test_split_idx = train_test_split_idx
        self.list_saidx = list_saidx
        self.all_squidx = all_squidx
        self.student_idxs = student_idxs
        self.list_saidx2aidx = list_saidx2aidx
        self.unique_time_obs = unique_time_obs
        self.n_points = n_points

        self.student_masks = []
        for sidx in range(self.n_students):
            if list_saidx[sidx] is None:
                self.student_masks.append(None)
            else:
                self.student_masks.append(self.student_idxs == sidx)

        # Initialize likelihood
        self.likelihood = likelihood(device=device)
        self.list_points = []
        self.list_thetas = []
        self.list_zs = []

    def load_state(self, result_folder, continue_iter=0):
        if continue_iter == 0:
            print("Initializing thetas and zs")
            self.iter = 0
            self.previous_thetas, self.previous_points = self.sample_theta_prior(
                sample_size=64
            )
            self.previous_zs = self.sample_z_prior()

        else:
            print("Loading thetas and zs")
            self.iter = continue_iter
            list_points = torch.load(
                f"{result_folder}/ess_points_by_iter_{continue_iter}.pt"
            )
            list_thetas = torch.load(
                f"{result_folder}/ess_thetas_by_iter_{continue_iter}.pt"
            )
            list_zs = torch.load(f"{result_folder}/ess_zs_by_iter_{continue_iter}.pt")

            self.previous_points = list_points[-1].to(self.device).float()
            self.previous_thetas = list_thetas[-1].to(self.device).float()
            self.previous_zs = list_zs[-1].to(self.device)[self.all_squidx].float()

    def sample(
        self,
        sampling_theta=False,
        sampling_z=False,
    ):
        if sampling_theta == sampling_z == False:
            raise ValueError(
                "At least one of sampling_theta and sampling_z must be True"
            )

        if sampling_theta:
            previous_f = self.previous_thetas
            other_f = self.previous_zs
            nu, nu_points = self.sample_theta_prior()

        elif sampling_z:
            previous_f = self.previous_zs
            other_f = self.previous_thetas
            nu = self.sample_z_prior()

        ll_current = self.likelihood.log_likelihood(
            previous_f[: self.train_test_split_idx],
            other_f[: self.train_test_split_idx],
            self.y_train,
        )
        ll_thres = ll_current + torch.log(torch.rand(1, device=self.device))

        angle = torch.rand(1, device=self.device) * 2 * np.pi
        angle_min, angle_max = angle - 2 * np.pi, angle

        while True:
            next_f = torch.cos(angle) * previous_f + torch.sin(angle) * nu
            log_likelihood = self.likelihood.log_likelihood(
                next_f[: self.train_test_split_idx],
                other_f[: self.train_test_split_idx],
                self.y_train,
            )

            if log_likelihood > ll_thres:
                break
            else:
                if angle == 0:
                    next_f = previous_f
                    break

                if angle < 0:
                    angle_min = angle
                else:
                    angle_max = angle
                angle = (
                    torch.rand(1, device=self.device) * (angle_max - angle_min)
                    + angle_min
                )

        # Save thetas and zs
        if sampling_theta:
            self.list_thetas.append(next_f.to(torch.bfloat16).cpu())
            self.previous_thetas = next_f

            self.previous_points = (
                torch.cos(angle) * self.previous_points + torch.sin(angle) * nu_points
            )
            self.list_points.append(self.previous_points.to(torch.bfloat16).cpu())
            self.iter += 1
            if self.iter % 100 == 0:
                print(f"Iteration {self.iter}\tLog likelihood: {log_likelihood}")

        elif sampling_z:
            constructed_z = torch.zeros((self.n_testcases,), device=self.device)
            constructed_z[self.all_squidx] = next_f
            self.list_zs.append(constructed_z.to(torch.bfloat16).cpu())
            self.previous_zs = next_f

        return log_likelihood

    def sample_z_prior(self):
        return self.z_prior_dists.sample()[self.all_squidx]

    def sample_theta_prior_by_student(self, sidx, sample_size=1):
        if self.theta_prior_dists[sidx] is None:
            return []
        else:
            return self.theta_prior_dists[sidx].sample(torch.Size([sample_size]))

    def sample_theta_prior(self, sample_size=1):
        thetas = []
        points = []
        for sidx in range(self.n_students):
            if self.list_saidx[sidx] is None:
                continue

            prior_samples = self.sample_theta_prior_by_student(sidx).mean(dim=0)

            thetas.append(prior_samples[: -self.n_points][self.list_saidx[sidx]])
            points.append(prior_samples[-self.n_points :])

        return torch.cat(thetas), torch.cat(points)

    def save_state(self, result_folder, iteration):
        # Save thetas and zs with torch.save
        torch.save(
            self.list_points,
            f"{result_folder}/ess_points_by_iter_{iteration}.pt",
        )
        torch.save(
            self.list_thetas,
            f"{result_folder}/ess_thetas_by_iter_{iteration}.pt",
        )
        torch.save(
            self.list_zs,
            f"{result_folder}/ess_zs_by_iter_{iteration}.pt",
        )
        print(f"Saved thetas and zs at iteration {iteration}")

        # Clear thetas and zs
        self.list_points = []
        self.list_thetas = []
        self.list_zs = []

import numpy as np
import torch
from torch.distributions.bernoulli import Bernoulli


class IRTLikelihood:
    def __init__(self, device):
        self.device = device

    def log_likelihood(self, theta, z, y):
        probs = torch.sigmoid(theta + z)
        likelihood_dist = Bernoulli(probs=probs)
        return likelihood_dist.log_prob(y).sum()


class GibbsESSampler:
    def __init__(
        self,
        likelihood,
        theta_prior_dists,
        z_prior_dists,
        y_train,
        train_test_split_idx,
        list_saidx,
        all_squidx,
        student_idxs,
        list_saidx2aidx,
        unique_time_obs,
        device,
        n_points=100,
    ):
        self.device = device
        self.theta_prior_dists = theta_prior_dists
        self.z_prior_dists = z_prior_dists
        self.n_students = len(theta_prior_dists)
        self.n_testcases = z_prior_dists.loc.shape[0]
        self.y_train = y_train
        self.train_test_split_idx = train_test_split_idx
        self.list_saidx = list_saidx
        self.all_squidx = all_squidx
        self.student_idxs = student_idxs
        self.list_saidx2aidx = list_saidx2aidx
        self.unique_time_obs = unique_time_obs
        self.n_points = n_points

        self.student_masks = []
        for sidx in range(self.n_students):
            if list_saidx[sidx] is None:
                self.student_masks.append(None)
            else:
                self.student_masks.append(self.student_idxs == sidx)

        # Initialize likelihood
        self.likelihood = likelihood(device=device)
        self.list_points = []
        self.list_thetas = []
        self.list_zs = []

    def load_state(self, result_folder, continue_iter=0):
        if continue_iter == 0:
            print("Initializing thetas and zs")
            self.iter = 0
            self.previous_thetas, self.previous_points = self.sample_theta_prior(
                sample_size=64
            )
            self.previous_zs = self.sample_z_prior()

        else:
            print("Loading thetas and zs")
            self.iter = continue_iter
            list_points = torch.load(
                f"{result_folder}/ess_points_by_iter_{continue_iter}.pt"
            )
            list_thetas = torch.load(
                f"{result_folder}/ess_thetas_by_iter_{continue_iter}.pt"
            )
            list_zs = torch.load(f"{result_folder}/ess_zs_by_iter_{continue_iter}.pt")

            self.previous_points = list_points[-1].to(self.device).float()
            self.previous_thetas = list_thetas[-1].to(self.device).float()
            self.previous_zs = list_zs[-1].to(self.device)[self.all_squidx].float()

    def sample(
        self,
        sampling_theta=False,
        sampling_z=False,
    ):
        if sampling_theta == sampling_z == False:
            raise ValueError(
                "At least one of sampling_theta and sampling_z must be True"
            )

        if sampling_theta:
            previous_f = self.previous_thetas
            other_f = self.previous_zs
            nu, nu_points = self.sample_theta_prior()

        elif sampling_z:
            previous_f = self.previous_zs
            other_f = self.previous_thetas
            nu = self.sample_z_prior()

        ll_current = self.likelihood.log_likelihood(
            previous_f[: self.train_test_split_idx],
            other_f[: self.train_test_split_idx],
            self.y_train,
        )
        ll_thres = ll_current + torch.log(torch.rand(1, device=self.device))

        angle = torch.rand(1, device=self.device) * 2 * np.pi
        angle_min, angle_max = angle - 2 * np.pi, angle

        while True:
            next_f = torch.cos(angle) * previous_f + torch.sin(angle) * nu
            log_likelihood = self.likelihood.log_likelihood(
                next_f[: self.train_test_split_idx],
                other_f[: self.train_test_split_idx],
                self.y_train,
            )

            if log_likelihood > ll_thres:
                break
            else:
                if angle == 0:
                    return previous_f

                if angle < 0:
                    angle_min = angle
                else:
                    angle_max = angle
                angle = (
                    torch.rand(1, device=self.device) * (angle_max - angle_min)
                    + angle_min
                )

        # Save thetas and zs
        if sampling_theta:
            self.list_thetas.append(next_f.to(torch.bfloat16).cpu())
            self.previous_thetas = next_f

            self.previous_points = (
                torch.cos(angle) * self.previous_points + torch.sin(angle) * nu_points
            )
            self.list_points.append(self.previous_points.to(torch.bfloat16).cpu())
            self.iter += 1
            if self.iter % 100 == 0:
                print(f"Iteration {self.iter}\tLog likelihood: {log_likelihood}")

        elif sampling_z:
            constructed_z = torch.zeros((self.n_testcases,), device=self.device)
            constructed_z[self.all_squidx] = next_f
            self.list_zs.append(constructed_z.to(torch.bfloat16).cpu())
            self.previous_zs = next_f

        return log_likelihood

    def sample_z_prior(self):
        return self.z_prior_dists.sample()[self.all_squidx]

    def sample_theta_prior_by_student(self, sidx, sample_size=1):
        if self.theta_prior_dists[sidx] is None:
            return []
        else:
            return self.theta_prior_dists[sidx].sample(torch.Size([sample_size]))

    def sample_theta_prior(self, sample_size=1):
        thetas = []
        points = []
        for sidx in range(self.n_students):
            if self.list_saidx[sidx] is None:
                continue

            prior_samples = self.sample_theta_prior_by_student(sidx).mean(dim=0)

            thetas.append(prior_samples[: -self.n_points][self.list_saidx[sidx]])
            points.append(prior_samples[-self.n_points :])

        return torch.cat(thetas), torch.cat(points)

    def save_state(self, result_folder, iteration):
        # Save thetas and zs with torch.save
        torch.save(
            self.list_points,
            f"{result_folder}/ess_points_by_iter_{iteration}.pt",
        )
        torch.save(
            self.list_thetas,
            f"{result_folder}/ess_thetas_by_iter_{iteration}.pt",
        )
        torch.save(
            self.list_zs,
            f"{result_folder}/ess_zs_by_iter_{iteration}.pt",
        )
        print(f"Saved thetas and zs at iteration {iteration}")

        # Clear thetas and zs
        self.list_points = []
        self.list_thetas = []
        self.list_zs = []

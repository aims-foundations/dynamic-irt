import argparse
import pickle

import torch
import wandb

from es_sampler import GibbsESSampler, IRTLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel

from huggingface_hub import snapshot_download
from torch.distributions import Normal
from tqdm import tqdm
from utils import ensure_dir, plot_prior_distribution, set_seed

device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_max_testcases(y_obs, n_question):
    print("Computing maximum number of testcases for each question")
    list_max_testcases = {}
    for qidx in range(n_question):
        list_max_testcases[qidx] = 0
        for student in y_obs:
            list_n_testcases = []
            for x in student[qidx]:
                if not isinstance(x, int):
                    list_n_testcases.append(len(x))
                else:
                    list_n_testcases.append(0)
            list_max_testcases[qidx] = max(
                list_max_testcases[qidx], max(list_n_testcases)
            )
    return list_max_testcases


def load_data(data_folder, smoke_test, test_split=0.2):
    y_obs = pickle.load(open(f"{data_folder}/correctness_bytc_matrix.pkl", "rb"))
    # u_obs has shape of n_student x n_question x n_attempt x n_testcase
    # The number of testcases can be different for each question

    # Number of questions
    n_question = len(y_obs[0])

    # Compute maximum number of testcases for each question
    list_max_testcases = compute_max_testcases(y_obs, n_question)
    total_testcases = sum(list_max_testcases.values())

    print("Loading preprocessed data")
    y_obs = torch.load("data/y_obs.pt")
    qidx_obs = torch.load("data/tidx_obs.pt")
    time_obs = torch.load("data/time_obs.pt")

    y_obs = torch.flatten(y_obs, start_dim=1)
    qidx_obs = torch.flatten(qidx_obs, start_dim=1)
    time_obs = torch.flatten(time_obs, start_dim=1)

    n_student = len(y_obs)
    first_idx = torch.arange(start=0, end=n_student).reshape(-1, 1)
    sorted_idx = torch.argsort(time_obs, dim=1)
    y_obs = y_obs[first_idx, sorted_idx]
    qidx_obs = qidx_obs[first_idx, sorted_idx]
    time_obs = time_obs[first_idx, sorted_idx]

    if smoke_test:
        n_student = 2
        y_obs = y_obs[:n_student]
        qidx_obs = qidx_obs[:n_student]
        time_obs = time_obs[:n_student]

    masked_idx = y_obs != -1

    # Create y_train
    print("Splitting train and test data")
    num_train = int((1 - test_split) * n_student)
    y_train = []
    y_test = []
    for sidx in range(n_student):
        if masked_idx[sidx].sum() == 0:
            continue

        if sidx < num_train:
            y_train.append(y_obs[sidx][masked_idx[sidx]])
        else:
            y_test.append(y_obs[sidx][masked_idx[sidx]])
    y_train = torch.concatenate(y_train).float()
    y_test = torch.concatenate(y_test).float()

    unique_time_obs = []
    aidx_obs = []  # student attempt index
    for tidx, time_ob in enumerate(time_obs):
        uni_time = time_ob.unique()
        aidx_ob = torch.searchsorted(uni_time, time_ob)
        # Replace the last element with -1
        aidx_ob[aidx_ob == len(uni_time) - 1] = -1
        aidx_obs.append(aidx_ob)
        unique_time_obs.append(uni_time[:-1])

    # Create index vectors for students and testcases
    list_available_sidx = []
    list_saidx = []
    list_sqidx = []
    train_test_split_idx = 0
    student_idxs = []
    for sidx in range(n_student):
        if masked_idx[sidx].sum() == 0:
            list_saidx.append(None)
            # list_sqidx.append(None)
            continue

        saidx = aidx_obs[sidx][masked_idx[sidx]]  # attemp index for student
        list_saidx.append(saidx)
        list_available_sidx.append(sidx)

        sqidx = qidx_obs[sidx][masked_idx[sidx]]  # global testcase index
        list_sqidx.append(sqidx)

        student_idxs.extend([sidx] * saidx.shape[0])

        if sidx < num_train:
            train_test_split_idx += saidx.shape[0]

    student_idxs = torch.tensor(student_idxs)
    all_squidx = torch.cat(list_sqidx)

    # Save student indexes
    with open(f"{result_folder}/student_idxs.pkl", "wb") as f:
        pickle.dump(student_idxs, f)

    # Save student saidx indexes
    with open(f"{result_folder}/list_saidx.pkl", "wb") as f:
        pickle.dump(list_saidx, f)

    # Save all squidx indexes
    with open(f"{result_folder}/all_squidx.pkl", "wb") as f:
        pickle.dump(all_squidx, f)

    # Save list of available student indexes
    with open(f"{result_folder}/list_available_sidx.pkl", "wb") as f:
        pickle.dump(list_available_sidx, f)

    # Reverse the attempt indexes of students
    print("Reverse the attempt indexes of students")
    list_saidx2aidx = []
    for sidx in range(n_student):
        if masked_idx[sidx].sum() == 0:
            list_saidx2aidx.append(None)
            continue

        saidx2aidx = []
        for aidx in list_saidx[sidx].unique().sort()[0]:
            saidx2aidx.append(torch.where(list_saidx[sidx] == aidx)[0][0])

        list_saidx2aidx.append(torch.tensor(saidx2aidx))

    # Save attempt indexes
    with open(f"{result_folder}/list_saidx2aidx.pkl", "wb") as f:
        pickle.dump(list_saidx2aidx, f)

    return (
        n_question,
        n_student,
        num_train,
        total_testcases,
        y_train,
        y_test,
        unique_time_obs,
        list_saidx,
        all_squidx,
        student_idxs,
        list_saidx2aidx,
        train_test_split_idx,
        masked_idx,
    )


def get_theta_priors(unique_time_obs, sidx, npoints, kernel, length_scale):
    time_obs_s = unique_time_obs[sidx]
    points = torch.linspace(
        unique_time_obs[sidx].min(),
        unique_time_obs[sidx].max(),
        npoints,
        device=device,
    )
    time_obs_s = torch.cat([time_obs_s, points])
    time_obs_s = time_obs_s.reshape(-1, 1)
    if kernel == "Matern":
        kernel = MaternKernel(nu=2.5).to(device)
    elif kernel == "RBF":
        kernel = RBFKernel().to(device)
        kernel._set_lengthscale(length_scale)
    else:
        raise ValueError("Invalid kernel type")

    covar = kernel(time_obs_s)
    return MultivariateNormal(
        torch.zeros(covar.shape[0], device=device), covariance_matrix=covar
    )


if __name__ == "__main__":
    # wandb.init(project="code_insights")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--course_name", help="Course Name", type=str, default="dsa_hk231"
    )
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--epochs", help="Number of epochs", type=int, default=100000)
    parser.add_argument(
        "--continue_iter", help="Continue sampling", type=int, default=0
    )
    parser.add_argument(
        "--kernel",
        help="Prior Kernel",
        type=str,
        default="RBF",
        choices=["RBF", "Matern"],
    )
    parser.add_argument("--npoints", type=int, default=500)
    parser.add_argument("--length_scale", help="Length scale", type=float, default=50.0)
    parser.add_argument("--smoke_test", help="Enable smoke test", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    result_folder = f"results/{args.course_name}_seed{args.seed}_npoints{args.npoints}_kernel{args.kernel}_lengthscale{args.length_scale}"

    ensure_dir(result_folder)
    data_folder = snapshot_download(
        repo_id=f"stair-lab/{args.course_name}_wtc", repo_type="dataset"
    )

    # Load data
    (
        n_question,
        n_student,
        num_train,
        total_testcases,
        y_train,
        y_test,
        unique_time_obs,
        list_saidx,
        all_squidx,
        student_idxs,
        list_saidx2aidx,
        train_test_split_idx,
        masked_idx,
    ) = load_data(data_folder, args.smoke_test)

    # Create normal prior distribution for theta,
    # where each theta_i is corresponding to a student at a specific time
    print("Creating theta priors")
    theta_priors = []
    for sidx in range(n_student):
        if masked_idx[sidx].sum() == 0:
            theta_priors.append(None)
            continue
        theta_priors.append(
            get_theta_priors(
                unique_time_obs,
                sidx,
                npoints=args.npoints,
                kernel=args.kernel,
                length_scale=args.length_scale,
            )
        )

    # Create normal prior distribution for z,
    # where each z_i is corresponding to a testcase in a question
    print("Creating z priors")
    z_priors = Normal(
        loc=torch.zeros((total_testcases,), device=device),
        scale=torch.ones((total_testcases,), device=device),
    )

    # Initialize sampler
    ges_sampler = GibbsESSampler(
        likelihood=IRTLikelihood,
        theta_prior_dists=theta_priors,
        z_prior_dists=z_priors,
        y_train=y_train,
        train_test_split_idx=train_test_split_idx,
        list_saidx=list_saidx,
        all_squidx=all_squidx,
        unique_time_obs=unique_time_obs,
        student_idxs=student_idxs,
        list_saidx2aidx=list_saidx2aidx,
        device=device,
        n_points=args.npoints,
    )

    ges_sampler.load_state(result_folder, continue_iter=args.continue_iter)

    for epoch in tqdm(range(args.continue_iter, args.epochs), desc="Sampling"):
        # Sampling for z
        ges_sampler.sample(sampling_z=True)

        # Sampling for theta
        ges_sampler.sample(sampling_theta=True)

        # Save thetas and zs
        if (epoch + 1) % 1000 == 0:
            ges_sampler.save_state(result_folder, epoch + 1)

    # wandb.finish()

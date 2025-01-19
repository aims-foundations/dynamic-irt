import argparse
import logging
import os
import pickle
from typing import Tuple

import pyro
import pyro.distributions as dist

import torch
from huggingface_hub import snapshot_download
from pyro.infer import MCMC, NUTS
from utils import ensure_dir, preprocess, set_seed


def irt_pyro_model(
    response_matrix: torch.Tensor,
    ability_prior_dists,
    item_prior_dists,
    question_expanding_indexes,
    response_time_indexes,
):
    # Initialize difficulty parameters
    with pyro.plate("input_difficulty"):
        difficulty = pyro.sample("difficulty", item_prior_dists)
    difficulty = difficulty[question_expanding_indexes]

    # Initialize ability parameters
    abilities = []
    for sid, (ability_prior, time_index) in enumerate(
        zip(ability_prior_dists, response_time_indexes)
    ):
        ab = pyro.sample(f"ability_{sid}", ability_prior)
        abilities.append(ab[time_index])
    abilities = torch.cat(abilities)

    # Compute probability matrix
    prob_matrix = torch.sigmoid(abilities + difficulty)
    with pyro.plate("output"):
        y = pyro.sample("y", dist.Bernoulli(prob_matrix), obs=response_matrix)

    return y


def infer_hmc(
    max_epoch: int,
    warmup_steps: int,
    response_matrix: torch.Tensor,
    all_indexes: Tuple,
):
    assert max_epoch > warmup_steps, "max_epoch should be greater than warmup_steps"
    (
        observation_mask,
        response_time_indexes,
        question_expanding_indexes,
        ability_prior_dists,
        item_prior_dists,
    ) = all_indexes
    logging.info("Running inference...")

    kernel = NUTS(
        irt_pyro_model,
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
        response_matrix=response_matrix[observation_mask],
        ability_prior_dists=ability_prior_dists,
        item_prior_dists=item_prior_dists,
        question_expanding_indexes=question_expanding_indexes,
        response_time_indexes=response_time_indexes,
    )

    mcmc.summary()

    return mcmc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course_name", type=str, default="dsa_hk231")
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--fitting_method", type=str, default="hmc")
    parser.add_argument("--kernel", type=str, default="RBF")
    parser.add_argument("--length_scale", type=float, default=1.0)
    parser.add_argument("--D", type=int, default=1)
    parser.add_argument("--PL", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_folder = f"results/{args.course_name}_s{args.seed}_D{args.D}_PL{args.PL}_{args.fitting_method}_kernel{args.kernel}_ls{args.length_scale}"
    ensure_dir(result_folder)

    # Download and load data
    data_folder = snapshot_download(
        repo_id=f"stair-lab/code_insights_matrices", repo_type="dataset"
    )
    data_folder = os.path.join(data_folder, args.course_name)

    # Load matrices
    response_matrix = torch.load(f"{data_folder}/correctness_matrix.pt").to(
        device, dtype=torch.float32
    )
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts

    response_time_matrix = torch.load(f"{data_folder}/time_matrix.pt").to(
        device, dtype=torch.float32
    )
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts

    if args.smoke:
        response_matrix = response_matrix[:2]
        response_time_matrix = response_time_matrix[:2]

    # Preprocess data
    if os.path.exists(f"{result_folder}/all_indexes.pkl"):
        all_indexes = pickle.load(open(f"{result_folder}/all_indexes.pkl", "rb"))
    else:
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
        pickle.dump(all_indexes, open(f"{result_folder}/all_indexes.pkl", "wb"))

    # Fit IRT model
    return_obj = infer_hmc(
        max_epoch=args.epochs,
        warmup_steps=args.warmup_steps,
        response_matrix=response_matrix,
        all_indexes=all_indexes,
    )

    print("Saving parameters...")
    samples = return_obj.get_samples()
    ability = []
    for sidx in range(response_matrix.shape[0]):
        ability.append(samples[f"ability_{sidx}"])
    torch.save(ability, f"{result_folder}/ability.pt")

    difficulty = samples["difficulty"]
    torch.save(difficulty, f"{result_folder}/difficulty.pt")

    sampling_diagnostic = return_obj.diagnostics()
    with open(f"{result_folder}/sampling_diagnostic.pkl", "w") as f:
        pickle.dump(sampling_diagnostic, f)

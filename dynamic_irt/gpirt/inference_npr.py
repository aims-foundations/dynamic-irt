import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import argparse
import gc
import pickle

import jax.numpy as jnp
import numpy as np

import torch
from huggingface_hub import snapshot_download
from irt_bayes import IRTBayes, IRTBayesJax
from utils import ensure_dir, set_seed

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
    response_matrix = torch.load(
        f"{data_folder}/correctness_matrix.pt", map_location="cpu"
    ).to(dtype=torch.float32)
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts

    response_time_matrix = torch.load(
        f"{data_folder}/time_matrix.pt", map_location="cpu"
    ).to(dtype=torch.float32)
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts

    if args.smoke:
        response_matrix = response_matrix[:2]
        response_time_matrix = response_time_matrix[:2]

    # Initialize IRT model
    irt_model = IRTBayesJax(
        D=args.D,
        PL=args.PL,
        device=device,
        low_rank_configs={
            "type": "GP",
            "kernel": args.kernel,
            "length_scale": args.length_scale,
        },
    )

    if os.path.exists(f"{result_folder}/param_cache.pkl"):
        print("Loading cached parameters...")
        all_params = pickle.load(open(f"{result_folder}/param_cache.pkl", "rb"))
        irt_model.preprocess(all_params=all_params)
    else:
        print("Initializing parameters...")
        all_params = irt_model.preprocess(
            response_matrix.to(device),
            response_time_matrix=response_time_matrix.to(device),
            fitting_method=args.fitting_method,
        )
        with open(f"{result_folder}/param_cache.pkl", "wb") as f:
            pickle.dump(all_params, f)

    # Fit IRT model
    print("Fitting model...")
    if args.fitting_method == "hmc":
        jnp_response_matrix = jnp.array(response_matrix.numpy())
        jnp_response_time_matrix = jnp.array(response_time_matrix.numpy())

        del response_matrix
        del response_time_matrix

        torch.cuda.empty_cache()
        gc.collect()

        return_obj = irt_model.infer_hmc(
            max_epoch=args.epochs,
            warmup_steps=20,
            response_matrix=jnp_response_matrix,
            response_time_matrix=jnp_response_time_matrix,
        )
    else:
        raise NotImplementedError(
            f"Fitting method {args.fitting_method} not implemented"
        )

    print("Saving model...")
    jnp.save(irt_model.ability, f"{result_folder}/ability.pt")
    jnp.save(irt_model.difficulty, f"{result_folder}/difficulty.pt")

    if args.fitting_method == "hmc":
        sampling_diagnostic = return_obj.diagnostics()
        with open(f"{result_folder}/sampling_diagnostic.pkl", "w") as f:
            pickle.dump(sampling_diagnostic, f)

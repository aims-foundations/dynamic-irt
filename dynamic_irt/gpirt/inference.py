import argparse
import os
import pickle

import torch
from amortized_irt import IRT
from huggingface_hub import snapshot_download
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

    # Fit IRT model
    irt_model = IRT(
        D=args.D, PL=args.PL, low_rank_constraint="distinctGP", device=device
    )
    return_obj = irt_model.fit(
        method=args.fitting_method,
        max_epoch=args.epochs,
        response_matrix=response_matrix,
        response_time_matrix=response_time_matrix,
        embedding=None,
        model_features=None,
    )

    print("Saving model...")
    torch.save(irt_model.ability, f"{result_folder}/ability.pt")
    torch.save(irt_model.difficulty, f"{result_folder}/difficulty.pt")

    if args.fitting_method == "hmc":
        sampling_diagnostic = return_obj.diagnostics()
        with open(f"{result_folder}/sampling_diagnostic.pkl", "w") as f:
            pickle.dump(sampling_diagnostic, f)

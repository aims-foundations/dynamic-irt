import os
import argparse
import torch
from huggingface_hub import snapshot_download

import pyro
import logging
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import pyro.contrib.gp as gp

# MAY NEED TO CHANGE THIS FUNTION TO IRT REPO
def irt_1d_1pl(ability, difficulty):
    return torch.sigmoid(ability + difficulty)

def irt_pyro_model(
    response_matrix, 
    observation_mask,
    time_index_matrix,
    ability_priors, 
    difficulty_prior,
):
    # Initialize parameters
    difficulty = pyro.sample(
        "difficulty",
        difficulty_prior
    )
    
    abilities = []
    for sid, (ability_prior, time_index) in enumerate(zip(ability_priors, time_index_matrix)):
        ab = pyro.sample(
            f"ability_{sid}",
            ability_prior
        )
        abilities.append(ab[time_index])
    abilities = torch.stack(abilities)

    prob_matrix = irt_1d_1pl(abilities, difficulty[:, None])[observation_mask]
    y = pyro.sample("y", dist.Bernoulli(prob_matrix), obs=response_matrix)
    return y


def infer_hmc(args, model, response_matrix, time_matrix):
    logging.info("Running inference...")
    kernel = NUTS(
        model,
        max_tree_depth=args.max_tree_depth,
        jit_compile=args.jit,
        ignore_jit_warnings=True,
    )

    # We'll define a hook_fn to log potential energy values during inference.
    # This is helpful to diagnose whether the chain is mixing.
    energies = []

    def hook_fn(kernel, *unused):
        e = float(kernel._potential_energy_last)
        energies.append(e)
        if args.verbose:
            logging.info("potential = {:0.6g}".format(e))

    mcmc = MCMC(
        kernel,
        hook_fn=hook_fn,
        num_samples=args.num_samples,
        warmup_steps=args.warmup_steps,
    )

    # Compute some shapes
    n_testtakers = response_matrix.shape[0]
    n_questions_testcases = response_matrix.shape[1]
    obs_mask = response_matrix != -1

    # Define difficulty prior
    difficulty_prior = dist.Normal(torch.zeros(n_questions_testcases, device=device), torch.ones(n_questions_testcases, device=device))

    # Define ability priors
    ## We could use batching here because each student has a different number of attempts
    ability_priors = []
    time_index_matrix = []
    for sidx in range(n_testtakers):
        student_time_vec = time_matrix[sidx].unique()
        
        # Get the time index for each attempt
        time_index = torch.searchsorted(student_time_vec, time_matrix[sidx])
        ### REMEMBER: time_index is 0-indexed. Element 0 is -1
        ### We need to subtract 1 to get the correct index
        time_index_matrix.append(time_index - 1)

        # Remove the first element since it is -1
        student_time_vec = student_time_vec[1:]
        
        kernel = gp.kernels.RBF(input_dim=1, lengthscale=torch.tensor(7.0))
        covar = kernel(student_time_vec) + 1e-3 * torch.eye(student_time_vec.shape[0], device=device) # Avoid numerical issues
        ability_priors.append(
            dist.MultivariateNormal(
                torch.zeros(student_time_vec.shape[0], device=device),
                covar
            )
        )

    # Run HMC
    mcmc.run(
        response_matrix=response_matrix[obs_mask],
        observation_mask=obs_mask,
        time_index_matrix=time_index_matrix,
        ability_priors=ability_priors,
        difficulty_prior=difficulty_prior,
    )

    if args.plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.plot(energies)
        plt.xlabel("MCMC step")
        plt.ylabel("potential energy")
        plt.title("MCMC energy trace")
        plt.tight_layout()

    samples = mcmc.get_samples()
    return samples

class PyroHMCArgs:
    def __init__(self):
        self.max_tree_depth = 10
        self.num_samples = 1000
        self.warmup_steps = 200
        self.verbose = True
        self.plot = True
        self.jit = False
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--course_name", type=str, default="dsa_hk231")
    parser.add_argument("--method", type=str, default="hmc")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download and load data
    data_folder = snapshot_download(
        repo_id=f"stair-lab/code_insights_matrices", repo_type="dataset"
    )
    data_folder = os.path.join(data_folder, args.course_name)
    
    # LOAD MATRICES
    response_matrix = torch.load(
        f"{data_folder}/correctness_matrix.pt"
    ).to(device, dtype=torch.float32)
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts
    
    response_time_matrix = torch.load(
        f"{data_folder}/time_matrix.pt"
    ).to(device, dtype=torch.float32)
    # >>> n_students x (n_questions * n_testcases) x n_max_attempts
    
    if args.smoke:
        response_matrix = response_matrix[:2]
        response_time_matrix = response_time_matrix[:2]
    
    hmc_args = PyroHMCArgs()
    samples = infer_hmc(
        args=hmc_args,
        model=irt_pyro_model,
        response_matrix=response_matrix,
        time_matrix=response_time_matrix,
    )

    # Save samples
    torch.save(samples, f"results/{args.course_name}.pt")

    
    ### OLD CODE ###
    # irt_model = IRT(D=1, PL=1, low_rank_constraint="distinctGP", device=device)
    
    # irt_model.fit(
    #     method="ess",
    #     max_epoch=10000,
    #     response_matrix=response_matrix,
    #     response_time_matrix=response_time_matrix,
    #     embedding=None,
    #     model_features=None
    # )
    
    # print("Saving model...")
    # torch.save(irt_model.ability, "data/ability.pt")
    # torch.save(irt_model.difficulty, "data/difficulty.pt")

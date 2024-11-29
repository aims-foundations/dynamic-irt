import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import ScaleKernel, RBFKernel, RQKernel, IndexKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
import warnings
warnings.filterwarnings("ignore")


# Define the synthetic function parameters
def log_function(x, a, b, c):
    """Logarithmic function with parameters a, b, and c."""
    return a * np.log(b * x + c)

# inverse sigmoid function
def inverse_sigmoid(x):
    return np.log(x / (1 - x))

# Define the function and groups
def synthetic_function_with_start_and_saturation(X, group_indices, params, S_values, Start_values):
    """
    Synthetic function with group-specific starting points and saturation levels.
    Args:
    - X: Input time values.
    - group_indices: Array indicating the group for each element of X.
    - params: List of parameter tuples (a, b, c) for each group.
    - S_values: List of saturation levels for each group.
    - Start_values: List of starting points for each group.
    Returns:
    - Outputs of the synthetic function.
    """
    output = np.zeros_like(X, dtype=float)
    for group, (a, b, c) in enumerate(params):
        mask = group_indices == (group + 1)  # Group is 1-based

        X_input = X[mask]

        # restart the input values at
        X_input = X_input - X_input.min()

        raw_output = log_function(X_input, a, b, c)
        # Scale and shift the group output to start at Start_values[group] and saturate at S_values[group]
        group_min = np.min(raw_output)
        group_max = np.max(raw_output)
        scaled_output = (raw_output - group_min) / (group_max - group_min)
        output[mask] = Start_values[group] + (S_values[group] - Start_values[group]) * scaled_output
    return output


# Define custom kernel
class CustomKernel(torch.nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.rbf = ScaleKernel(RBFKernel(ard_num_dims=1))
        self.rq = ScaleKernel(RQKernel(ard_num_dims=1))
        self.index = IndexKernel(num_tasks=num_groups, rank=1)

    def forward(self, x):
        group = x[..., -1:]
        x = x[..., :-1]
        return self.rbf(x) + self.rq(x) + self.index(group)
    

if __name__ == "__main__":
    # set the seed
    np.random.seed(0)
    torch.manual_seed(0)

    # Generate sample data
    N = 100  # Total number of points
    K = 5    # Number of groups
    X = np.linspace(1, 100, N*10)

    # make X irregularly spaced
    X = np.random.choice(X, N, replace=False)

    # sort X
    X = np.sort(X)

    # Assign groups
    group_sizes = [N // K] * K  # Equal group sizes
    group_sizes[-1] += N % K   # Add remainder to the last group
    group_indices = np.concatenate([np.full(size, i + 1) for i, size in enumerate(group_sizes)])

    # Parameters for each group (a, b, c), saturation levels, and starting points
    params_faster_saturation_adjusted = [
        (1.0, 0.5, 1.0) for _ in range(K)
    ]
    S_values = np.linspace(0.6, 0.9, K) # Saturation levels for groups
    Start_values = np.linspace(0.1, 0.4, K) # Starting points for groups

    # Compute the synthetic function with starting points and saturation levels
    Y_start_saturation = synthetic_function_with_start_and_saturation(
        X, group_indices, params_faster_saturation_adjusted, S_values, Start_values
    )

    # inverse sigmoid transform of the Y values
    Y_start_saturation = inverse_sigmoid(Y_start_saturation)

    # for each group, create the difficulty of the question by sample randomly from a standard normal
    difficulty_before_broadcast = np.random.normal(0, 1, K)
    # broadcast the difficulty to the Y_start_saturation
    difficulty = np.repeat(difficulty_before_broadcast, group_sizes)

    logit = (difficulty + Y_start_saturation)
    logit = 1 / (1 + np.exp(-logit))

    # response is bernouli with probability of success = logit
    response = np.random.binomial(1, logit)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))

    for group in range(1, K + 1):
        mask = group_indices == group
        ax[0].scatter(
            X[mask], Y_start_saturation[mask], 
            label=f'Group {group} (Start: {Start_values[group - 1]}, Saturation: {S_values[group - 1]})'
        )

    # plot the difficulty in the middle of each group
    for group in range(1, K + 1):
        mask = group_indices == group
        # middle x value of the group
        x = X[mask].mean()
        # y is the difficulty of the group before broadcasting
        ax[0].scatter(x, difficulty_before_broadcast[group - 1], marker='*', s=100)

    ax[0].set_ylabel("Ability and Difficulty")
    ax[0].set_xlabel("Time")

    # plot the response pattern 
    for group in range(1, K + 1):
        mask = group_indices == group
        ax[1].scatter(X[mask], response[mask], marker='x', s=50)
        ax[1].set_ylabel("Response")
        ax[1].set_xlabel("Time")

    plt.savefig("synthetic_function_problem_solver.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Prepare data for GP
    X_augmented = np.column_stack([X, group_indices])  # Combine time and group indices
    X_train = torch.tensor(X_augmented, dtype=torch.float64)
    Y_train = torch.tensor(Y_start_saturation, dtype=torch.float64).unsqueeze(-1)

    # try the custom kernel and RBF kernel
    for kernel_name, kernel in [
        ("Mix", CustomKernel(K)), 
        ("RBF", ScaleKernel(RBFKernel(ard_num_dims=1)))
    ]:
        # Create GP model
        model = SingleTaskGP(
            train_X=X_train,
            train_Y=Y_train,
            covar_module=kernel,
            input_transform=Normalize(d=X_train.shape[-1]),
            outcome_transform=Standardize(m=1),
        )

        # Fit the GP model
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Make predictions
        N = 1000  # Total number of points
        K = 5    # Number of groups
        X_test = np.linspace(1, 100, N*10)
        X_test = np.random.choice(X_test, N, replace=False)
        X_test = np.sort(X_test)
        X_test = torch.tensor(X_test, dtype=torch.float64).unsqueeze(-1)

        # Assign groups
        group_sizes = [N // K] * K  # Equal group sizes
        group_sizes[-1] += N % K   # Add remainder to the last group
        group_test = np.concatenate([np.full(size, i + 1) for i, size in enumerate(group_sizes)])
        group_test = torch.tensor(group_test, dtype=torch.float64).unsqueeze(-1)

        X_test_augmented = torch.cat([X_test, group_test], dim=-1)

        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_test_augmented)
            mean = posterior.mean
            lower, upper = posterior.mvn.confidence_region()

        # Visualize
        plt.figure(figsize=(10, 6))
        plt.plot(X_test.numpy(), mean.numpy(), label="GP Mean")
        plt.fill_between(
            X_test.numpy().flatten(),
            lower.numpy(),
            upper.numpy(),
            alpha=0.2,
            label="Confidence Interval",
        )
        plt.scatter(X_train[:, 0].numpy(), Y_train.numpy(), color="red", label="Training Data")
        plt.title(f"Fitted GP with {kernel_name} Kernel")
        plt.xlabel("Time")
        plt.ylabel("Ability")
        plt.legend()
        plt.savefig(f"synthetic_function_gp_fit_{kernel_name}.png", dpi=300, bbox_inches='tight')

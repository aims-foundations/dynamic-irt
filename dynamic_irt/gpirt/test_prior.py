import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel
from tueplots import bundles

plt.rcParams.update(bundles.aaai2024())


def set_seed(seed):
    random.seed(seed)
    # torch.backends.cudnn.deterministic=True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, default="RBF", choices=["RBF", "Matern"])
    parser.add_argument("--length_scale", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=2.5, choices=[0.5, 1.5, 2.5])
    args = parser.parse_args()

    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    times = torch.linspace(0, 100, 500, device=device).reshape(-1, 1)

    if args.kernel == "RBF":
        kernel = RBFKernel()
        kernel._set_lengthscale(args.length_scale)
    else:
        kernel = MaternKernel(nu=args.nu)
    kernel = kernel.to(device)
    covar = kernel(times, times)

    dist = MultivariateNormal(
        torch.zeros(covar.shape[0], device=device), covariance_matrix=covar
    )

    samples = dist.sample(torch.Size([10]))

    for sample in samples:
        plt.plot(sample.cpu().numpy(), alpha=0.2)
    if args.kernel == "RBF":
        plt.title(f"RBF Kernel (Length Scale: {args.length_scale})")
        plt.savefig("prior_rbf{}.png".format(args.length_scale), dpi=300)
    else:
        plt.title(f"Matern Kernel (Nu: {args.nu})")
        plt.savefig("prior_matern{}.png".format(args.nu), dpi=300)
    plt.close()

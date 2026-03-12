import os, math, argparse
import numpy as np
import matplotlib.pyplot as plt

# population dynamics implementation
from population_dynamics_BP import population_dynamics


def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def eps_ks_binary(k: int = 2) -> float:
    return 0.5 * (1.0 - 1.0 / math.sqrt(k))


def gaussian_census_accuracy_infty(eps: float) -> float:
    eps_c = eps_ks_binary(2)
    if eps > eps_c:
        return 0.5

    theta = 1.0 - 2.0 * eps
    t2 = theta * theta
    denom_base = (2.0 * t2 - 1.0)
    if abs(denom_base) < 1e-14:
        return 0.5

    gamma = (1.0 - t2) / denom_base
    gamma = max(gamma, 1e-15)

    arg = math.sqrt(1.0 / gamma)
    arg = min(arg, 12.0)
    return Phi(arg)


def popdyn_bp_error_vs_eps(eps_grid, pop_size=200_000, n_iters=150, seed=42):
    """
    Wrapper consistent with bp_finite_l_vs_popdyn.py
    Returns p_err = mean(pool < 0) when conditioning on root=+1.
    """
    eps_grid = np.asarray(eps_grid, dtype=np.float64)
    p_err = np.zeros_like(eps_grid, dtype=np.float64)
    for i, eps in enumerate(eps_grid):
        theta = 1.0 - 2.0 * float(eps)
        pool = population_dynamics(theta, M=int(pop_size), n_sweeps=int(n_iters), seed=int(seed) + i)
        p_err[i] = float(np.mean(pool < 0.0))
    return eps_grid, p_err


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps_min", type=float, default=0.0)
    ap.add_argument("--eps_max", type=float, default=0.30)
    ap.add_argument("--eps_step", type=float, default=0.01)
    ap.add_argument("--pop_size", type=int, default=200000)
    ap.add_argument("--pop_iters", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="bp_popdyn_out")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    eps_grid = np.arange(args.eps_min, args.eps_max + 1e-12, args.eps_step)
    eps_c = eps_ks_binary(2)

    # population dynamics BP error
    eps_pd, perr_pd = popdyn_bp_error_vs_eps(
        eps_grid, pop_size=args.pop_size, n_iters=args.pop_iters, seed=args.seed
    )

    # gaussian census infinite-depth -> convert to error
    acc_gauss = np.array([gaussian_census_accuracy_infty(float(e)) for e in eps_grid], dtype=np.float64)
    err_gauss = 1.0 - acc_gauss

    # plot
    plt.figure(figsize=(9, 5.5))
    plt.plot(eps_pd, perr_pd, marker="o", linestyle="None", label="BP population dynamics (L→∞)")
    plt.plot(eps_grid, err_gauss, linestyle="-", linewidth=2.0, label="Gaussian census (L→∞)")
    plt.axvline(eps_c, linestyle="--", linewidth=1.5, color="black", label=fr"$\varepsilon_c \approx {eps_c:.3f}$")

    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"reconstruction error")
    plt.title("BP population dynamics vs Gaussian census (infinite depth)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(args.outdir, "bp_popdyn_vs_gaussian_census.png")
    plt.savefig(out_png, dpi=180)
    plt.close()

    out_npz = os.path.join(args.outdir, "bp_popdyn_vs_gaussian_census_curves.npz")
    np.savez(out_npz, eps=eps_grid, bp_err=perr_pd, gauss_err=err_gauss, eps_c=eps_c)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
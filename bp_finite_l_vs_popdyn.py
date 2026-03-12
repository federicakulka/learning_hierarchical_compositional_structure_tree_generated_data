import os, json, math
import numpy as np
import matplotlib.pyplot as plt

# population dynamics implementation
from population_dynamics_BP import population_dynamics


def theta_from_eps(eps: float) -> float:
    return 1.0 - 2.0 * eps


def parent_magnetization(m1: np.ndarray, m2: np.ndarray, theta: float) -> np.ndarray:
    numer = theta * (m1 + m2)
    denom = 1.0 + (theta * theta) * (m1 * m2)

    out = np.empty_like(numer, dtype=np.float64)
    good = np.abs(denom) > 1e-12
    np.divide(numer, denom, out=out, where=good)

    # saturate rare ill-conditioned cases
    out[~good] = np.sign(numer[~good])
    out[~good & (numer == 0.0)] = 0.0
    return np.clip(out, -1.0, 1.0)

def bp_root_magnetization_from_leaves(leaves_pm: np.ndarray, eps: float) -> np.ndarray:
    th = theta_from_eps(eps)
    m = leaves_pm.astype(np.float64, copy=False)

    P, N = m.shape
    if not (N > 0 and (N & (N - 1) == 0)):
        raise ValueError(f"N must be a power of 2, got N={N}")

    while N > 1:
        m1 = m[:, 0:N:2]
        m2 = m[:, 1:N:2]
        m = parent_magnetization(m1, m2, th)
        N = m.shape[1]

    return m[:, 0]


def unpack_packed_leaves(packed: np.ndarray, N: int) -> np.ndarray:
    bits = np.unpackbits(packed, axis=1)[:, :N]
    return (2 * bits.astype(np.int8) - 1).astype(np.int8)


def load_dataset_folder(data_dir: str, split: str, P: int):
    meta_path = os.path.join(data_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # robust N extraction
    if "N" in meta:
        N = int(meta["N"])
    elif "n_leaves" in meta:
        N = int(meta["n_leaves"])
    elif "d" in meta:
        N = int(meta["d"])
    else:
        raise KeyError(f"meta.json in {data_dir} does not contain N/n_leaves/d keys")

    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))[:P].astype(np.int64)

    packed_path = os.path.join(data_dir, f"leaves_{split}_packed.npy")
    raw_path = os.path.join(data_dir, f"leaves_{split}.npy")

    if os.path.isfile(packed_path):
        packed = np.load(packed_path)[:P]
        leaves_pm = unpack_packed_leaves(packed, N)
    elif os.path.isfile(raw_path):
        leaves = np.load(raw_path)[:P]
        leaves_pm = leaves.astype(np.int8)
        if set(np.unique(leaves_pm)).issubset({0, 1}):
            leaves_pm = (2 * leaves_pm - 1).astype(np.int8)
    else:
        raise FileNotFoundError(f"Missing leaves_{split}_packed.npy or leaves_{split}.npy in {data_dir}")

    return leaves_pm, y, N

def popdyn_bp_error_vs_eps(eps_grid, pop_size=200_000, n_iters=150, seed=42):
    """Population dynamics baseline.
    We interpret:
      pop_size -> M
      n_iters  -> n_sweeps

    Returns eps_grid, p_err (root error conditioned on x0=+1), where
      p_err = mean(pool < 0)
    as defined in population_dynamics_BP.py.
    """
    eps_grid = np.asarray(eps_grid, dtype=np.float64)
    p_err = np.zeros_like(eps_grid, dtype=np.float64)

    for i, eps in enumerate(eps_grid):
        theta = 1.0 - 2.0 * float(eps)
        pool = population_dynamics(theta, M=int(pop_size), n_sweeps=int(n_iters), seed=int(seed) + i)
        p_err[i] = float(np.mean(pool < 0.0))

    return eps_grid, p_err


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--P", type=int, default=5000)
    ap.add_argument("--L_list", type=str, default="6,8,10,12", help="comma-separated L values")
    ap.add_argument("--eps_min", type=float, default=0.0)
    ap.add_argument("--eps_max", type=float, default=0.50)
    ap.add_argument("--eps_step", type=float, default=0.01)
    ap.add_argument("--outdir", type=str, default="bp_out")
    ap.add_argument("--pop_size", type=int, default=200000)
    ap.add_argument("--pop_iters", type=int, default=150,
                    help="Population dynamics sweeps (maps to n_sweeps in population_dynamics_BP.py).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed for population dynamics (one seed per epsilon).")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    L_list = [int(x) for x in args.L_list.split(",") if x.strip() != ""]
    eps_grid = np.arange(args.eps_min, args.eps_max + 1e-12, args.eps_step)
    eps_str = [f"{e:.3f}" for e in eps_grid]

    # L->\infin baseline
    eps_pd, perr_pd = popdyn_bp_error_vs_eps(
        eps_grid, pop_size=args.pop_size, n_iters=args.pop_iters, seed=args.seed
    )

    # finite-L curves
    curves = {"popdyn": (eps_pd, perr_pd)}
    for L in L_list:
        N = 2 ** L
        perr_L = []
        for es, e in zip(eps_str, eps_grid):
            d = os.path.join(args.data_root, f"N{N}", f"eps_{es}")
            leaves_pm, y, Ncheck = load_dataset_folder(d, args.split, args.P)
            if Ncheck != N:
                raise ValueError(f"N mismatch: folder {d} meta says {Ncheck}, expected {N}")

            m_root = bp_root_magnetization_from_leaves(leaves_pm, float(e))

            # y expected 0/1
            y_pm = 2 * y - 1
            yhat_pm = np.where(m_root >= 0.0, 1, -1)
            perr_L.append(float(np.mean(yhat_pm != y_pm)))

        curves[f"L{L}"] = (eps_grid.copy(), np.array(perr_L, dtype=np.float64))

    # KS threshold
    eps_c = 0.5 * (1.0 - 1.0 / np.sqrt(2.0))

    # plot
    plt.figure(figsize=(9, 5.5))
    plt.plot(eps_pd, perr_pd, marker="o", linestyle="None", label="Population dynamics BP (L→∞)")
    for L in L_list:
        e, err = curves[f"L{L}"]
        plt.plot(e, err, linestyle="-", label=fr"Finite-L BP (L={L}, N={2**L})")
    plt.axvline(eps_c, linestyle="--", label=fr"$\varepsilon_c \approx {eps_c:.3f}$")

    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$P_{\mathrm{err}}$")
    plt.title("Reconstruction error: finite-L BP vs population dynamics")
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(args.outdir, "bp_finiteL_vs_popdyn.png")
    plt.savefig(out_png, dpi=160)
    plt.close()

    # save numeric curves
    out_npz = os.path.join(args.outdir, "bp_finiteL_vs_popdyn_curves.npz")
    np.savez(out_npz, **{k: np.vstack(curves[k]) for k in curves})
    print(f"Wrote {out_png}")
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
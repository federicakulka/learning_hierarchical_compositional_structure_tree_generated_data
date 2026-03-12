import os, json, math, argparse
import numpy as np
import matplotlib.pyplot as plt

# BP finite-L
def theta_from_eps(eps: float) -> float:
    return 1.0 - 2.0 * eps

def parent_magnetization(m1: np.ndarray, m2: np.ndarray, theta: float) -> np.ndarray:
    numer = theta * (m1 + m2)
    denom = 1.0 + (theta * theta) * (m1 * m2)
    out = np.empty_like(numer, dtype=np.float64)
    good = np.abs(denom) > 1e-12
    np.divide(numer, denom, out=out, where=good)
    out[~good] = np.sign(numer[~good])
    out[~good & (numer == 0.0)] = 0.0
    return np.clip(out, -1.0, 1.0)

def bp_root_magnetization_from_leaves(leaves_pm: np.ndarray, eps: float) -> np.ndarray:
    th = theta_from_eps(eps)
    m = leaves_pm.astype(np.float64, copy=False)
    P, N = m.shape
    if not (N > 0 and (N & (N - 1) == 0)):
        raise ValueError(f"N must be power of 2, got N={N}")
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

    if "N" in meta:
        N = int(meta["N"])
    elif "n_leaves" in meta:
        N = int(meta["n_leaves"])
    elif "d" in meta:
        N = int(meta["d"])
    else:
        raise KeyError(f"meta.json in {data_dir} does not contain N/n_leaves/d")

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

def census_err_from_leaves(leaves_pm: np.ndarray, y01: np.ndarray) -> float:
    # y01 is 0/1, convert to \pm 1
    y_pm = 2 * y01 - 1
    s = leaves_pm.sum(axis=1)
    yhat_pm = np.where(s >= 0, 1, -1)
    return float(np.mean(yhat_pm != y_pm))

def bp_err_from_leaves(leaves_pm: np.ndarray, y01: np.ndarray, eps: float) -> float:
    y_pm = 2 * y01 - 1
    m_root = bp_root_magnetization_from_leaves(leaves_pm, float(eps))
    yhat_pm = np.where(m_root >= 0.0, 1, -1)
    return float(np.mean(yhat_pm != y_pm))

def eps_ks_binary(k: int = 2) -> float:
    return 0.5 * (1.0 - 1.0 / math.sqrt(k))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--P", type=int, default=5000)
    ap.add_argument("--L_list", type=str, default="6,8,10,12", help="comma-separated L values")
    ap.add_argument("--eps_min", type=float, default=0.0)
    ap.add_argument("--eps_max", type=float, default=0.30)
    ap.add_argument("--eps_step", type=float, default=0.01)
    ap.add_argument("--outdir", type=str, default="census_vs_bp_out")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    L_list = [int(x) for x in args.L_list.split(",") if x.strip() != ""]
    eps_grid = np.arange(args.eps_min, args.eps_max + 1e-12, args.eps_step)
    eps_str = [f"{e:.3f}" for e in eps_grid]
    eps_c = eps_ks_binary(2)

    curves = {}
    for L in L_list:
        N = 2 ** L
        err_bp = []
        err_census = []
        for es, e in zip(eps_str, eps_grid):
            d = os.path.join(args.data_root, f"N{N}", f"eps_{es}")
            leaves_pm, y01, Ncheck = load_dataset_folder(d, args.split, args.P)
            if Ncheck != N:
                raise ValueError(f"N mismatch: folder {d} meta says {Ncheck}, expected {N}")
            err_bp.append(bp_err_from_leaves(leaves_pm, y01, float(e)))
            err_census.append(census_err_from_leaves(leaves_pm, y01))

        curves[f"L{L}"] = (eps_grid.copy(),
                           np.array(err_bp, dtype=np.float64),
                           np.array(err_census, dtype=np.float64))

    # plot
    plt.figure(figsize=(9, 5.5))
    for L in L_list:
        e, bp_e, c_e = curves[f"L{L}"]
        plt.plot(e, bp_e, linestyle="-", linewidth=2.0, label=fr"BP (finite L), L={L}")
        plt.plot(e, c_e, linestyle="--", linewidth=2.0, label=fr"Census, L={L}")

    plt.axvline(eps_c, linestyle="--", color="black", linewidth=1.5, label=fr"$\varepsilon_c \approx {eps_c:.3f}$")
    plt.axhline(0.5, linestyle=":", color="gray", linewidth=1.0)

    plt.xlabel(r"$\varepsilon$")
    plt.ylabel("reconstruction error")
    plt.title("Census vs finite-depth BP reconstruction error")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    out_png = os.path.join(args.outdir, "census_vs_bp_finite_ell.png")
    plt.savefig(out_png, dpi=180)
    plt.close()

    # save curves
    out_npz = os.path.join(args.outdir, "census_vs_bp_finite_ell_curves.npz")
    payload = {f"L{L}": np.vstack(curves[f"L{L}"]) for L in L_list}
    payload["eps_c"] = eps_c
    np.savez(out_npz, **payload)
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
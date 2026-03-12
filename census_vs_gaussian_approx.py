import os, json, math
import numpy as np
import matplotlib.pyplot as plt


def unpack_packed_leaves(packed: np.ndarray, N: int) -> np.ndarray:
    # packed: (P, nbytes) uint8 -> leaves in {-1,+1} of shape (P,N)
    bits = np.unpackbits(packed, axis=1)[:, :N]
    return (2 * bits.astype(np.int8) - 1).astype(np.int8)


def load_leaves_and_labels(data_dir: str, split: str, P: int):
    """Returns leaves_pm (P,N) in {-1,+1}, y01 (P,) in {0,1}, N"""
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
        raise KeyError(f"meta.json in {data_dir} missing N/n_leaves/d")

    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))[:P].astype(np.int64)

    packed_path = os.path.join(data_dir, f"leaves_{split}_packed.npy")
    raw_path = os.path.join(data_dir, f"leaves_{split}.npy")

    if os.path.isfile(packed_path):
        packed = np.load(packed_path)[:P]
        leaves_pm = unpack_packed_leaves(packed, N)
    elif os.path.isfile(raw_path):
        leaves = np.load(raw_path)[:P]
        leaves_pm = leaves.astype(np.int8)
        # convert {0,1} -> {-1,+1}
        if set(np.unique(leaves_pm)).issubset({0, 1}):
            leaves_pm = (2 * leaves_pm - 1).astype(np.int8)
    else:
        raise FileNotFoundError(
            f"Missing leaves_{split}_packed.npy or leaves_{split}.npy in {data_dir}"
        )

    return leaves_pm, y, N


def census_predict(leaves_pm: np.ndarray) -> np.ndarray:
    """
    Census estimator: predict root label by majority vote over leaves.
    Returns yhat in {-1,+1}.
    """
    s = leaves_pm.sum(axis=1)  # shape (P,)
    # tie -> predict +1 (arbitrary but consistent)
    return np.where(s >= 0, 1, -1)


def census_accuracy(data_dir: str, split: str, P: int) -> float:
    leaves_pm, y01, _ = load_leaves_and_labels(data_dir, split, P)
    y_pm = 2 * y01 - 1
    yhat_pm = census_predict(leaves_pm)
    return float(np.mean(yhat_pm == y_pm))


# normal CDF
def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def factorized_clt_census_accuracy(eps: float, ell: int) -> float:
    """
    Factorized (independent-leaves) CLT baseline
    """
    theta = 1.0 - 2.0 * eps
    t = theta ** ell          # theta^ell
    t2 = t * t                # theta^(2ell)

    denom = max(1.0 - t2, 1e-15)
    arg = math.sqrt((2.0 ** ell) * t2 / denom)
    arg = min(arg, 12.0)
    return Phi(arg)


# correlated Gaussian census
def gaussian_census_accuracy(eps: float, ell: int) -> float:
    theta = 1.0 - 2.0 * eps
    t2 = theta * theta
    r = 2.0 * t2 

    denom_base = (2.0 * t2 - 1.0)
    if abs(denom_base) < 1e-14:
        return 0.5

    kappa = 1.0 - (t2 / denom_base)
    gamma = (1.0 - t2) / denom_base

    r_pow = r ** ell
    denom = kappa + gamma * r_pow
    if denom <= 0.0:
        denom = 1e-15

    arg = math.sqrt(r_pow / denom)
    arg = min(arg, 12.0)
    return Phi(arg)


def gaussian_census_accuracy_infty(eps: float) -> float:
    eps_c = 0.5 * (1.0 - 1.0 / math.sqrt(2.0))
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


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help=".../datasets/ising_q2")
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--P", type=int, default=5000)
    ap.add_argument("--L_list", type=str, default="9,13,17,21")
    ap.add_argument("--eps_min", type=float, default=0.0)
    ap.add_argument("--eps_max", type=float, default=0.30)
    ap.add_argument("--eps_step", type=float, default=0.01)
    ap.add_argument("--eps_fmt", type=str, default="{:.3f}",
                    help='Formatting for eps folder names, e.g. "{:.3f}" -> eps_0.100')
    ap.add_argument("--outdir", type=str, default="census_out")
    ap.add_argument("--overlay_infty", action="store_true",
                    help="Overlay the L→∞ correlated-Gaussian limit as an extra curve.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    L_list = [int(x) for x in args.L_list.split(",") if x.strip() != ""]
    eps_grid = np.arange(args.eps_min, args.eps_max + 1e-12, args.eps_step)
    eps_str = [args.eps_fmt.format(float(e)) for e in eps_grid]

    eps_c = 0.5 * (1.0 - 1.0 / math.sqrt(2.0))

    curves = {}
    for L in L_list:
        N = 2 ** L
        acc_census = []
        acc_gauss = []
        acc_fact = []

        for es, e in zip(eps_str, eps_grid):
            d = os.path.join(args.data_root, f"N{N}", f"eps_{es}")
            acc_census.append(census_accuracy(d, args.split, args.P))
            acc_gauss.append(gaussian_census_accuracy(float(e), L))
            acc_fact.append(factorized_clt_census_accuracy(float(e), L))

        curves[f"L{L}"] = {
            "eps": eps_grid.copy(),
            "census": np.array(acc_census, dtype=np.float64),
            "gauss": np.array(acc_gauss, dtype=np.float64),
            "fact": np.array(acc_fact, dtype=np.float64),
            "N": N,
        }

    acc_inf = None
    if args.overlay_infty:
        acc_inf = np.array([gaussian_census_accuracy_infty(float(e)) for e in eps_grid],
                           dtype=np.float64)

    # save numeric curves
    out_npz = os.path.join(args.outdir, "census_curves.npz")
    payload = {
        k: np.vstack([curves[k]["eps"], curves[k]["census"], curves[k]["gauss"], curves[k]["fact"]])
        for k in curves
    }
    if args.overlay_infty and acc_inf is not None:
        payload["gauss_infty"] = np.vstack([eps_grid.copy(), acc_inf])
    np.savez(out_npz, **payload)
    print(f"Wrote {out_npz}")

    # plot 1: census vs correlated Gaussian
    plt.figure(figsize=(9, 5.5))

    for L in L_list:
        c = curves[f"L{L}"]
        plt.plot(c["eps"], c["gauss"], linewidth=2.0, label=fr"Correlated Gaussian, L={L}")

    for L in L_list:
        c = curves[f"L{L}"]
        plt.plot(c["eps"], c["census"], marker="o", linestyle="None", markersize=3,
                 label=fr"Census (data), L={L}")

    if args.overlay_infty and acc_inf is not None:
        plt.plot(eps_grid, acc_inf, color="black", linewidth=2.5, label=r"$L\to\infty$ (Gaussian)")

    plt.axvline(eps_c, linestyle="--", color="black", linewidth=1.5)
    plt.legend(ncol=2, fontsize=9)
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$\phi^{(L)}(\varepsilon)$")
    plt.title("Census accuracy vs correlated Gaussian approximation")
    plt.ylim(0.48, 1.01)
    plt.tight_layout()

    out_png = os.path.join(args.outdir, "census_vs_gaussian_accuracy.png")
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"Wrote {out_png}")

    # plot 2: census vs factorized CLT baseline
    plt.figure(figsize=(9, 5.5))

    for L in L_list:
        c = curves[f"L{L}"]
        plt.plot(c["eps"], c["fact"], linewidth=2.0, label=fr"Factorized CLT, L={L}")

    for L in L_list:
        c = curves[f"L{L}"]
        plt.plot(c["eps"], c["census"], marker="o", linestyle="None", markersize=3,
                 label=fr"Census (data), L={L}")

    plt.axvline(eps_c, linestyle="--", color="black", linewidth=1.5)
    plt.legend(ncol=2, fontsize=9)
    plt.xlabel(r"$\varepsilon$")
    plt.ylabel(r"$\phi^{(L)}(\varepsilon)$")
    plt.title("Census accuracy vs factorized CLT baseline")
    plt.ylim(0.48, 1.01)
    plt.tight_layout()

    out_png2 = os.path.join(args.outdir, "census_vs_factorized_clt.png")
    plt.savefig(out_png2, dpi=160)
    plt.close()
    print(f"Wrote {out_png2}")


if __name__ == "__main__":
    main()
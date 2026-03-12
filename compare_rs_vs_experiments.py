import os, math, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _stable_color_map(values):
    vals = list(values)
    uniq = sorted(set(vals), key=float)
    cmap = plt.get_cmap("tab10" if len(uniq) <= 10 else "tab20")
    return {u: cmap(i % cmap.N) for i, u in enumerate(uniq)}


def Phi(x: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def fmt_alpha_folder(a: float) -> str:
    return f"{a:.2f}"


def load_rs_csv(rs_csv: str, Ns_keep=None) -> pd.DataFrame:
    df = pd.read_csv(rs_csv)
    df.columns = [c.strip() for c in df.columns]

    if "ok" in df.columns:
        df = df[df["ok"].astype(bool)].copy()

    if "epsilon" in df.columns and "eps" not in df.columns:
        df = df.rename(columns={"epsilon": "eps"})

    if "alpha" not in df.columns:
        raise ValueError(f"RS csv missing column 'alpha': {rs_csv}")
    if "eps" not in df.columns:
        raise ValueError(f"RS csv missing column 'eps' or 'epsilon': {rs_csv}")

    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["eps"]   = pd.to_numeric(df["eps"], errors="coerce")
    df = df.dropna(subset=["alpha", "eps"]).copy()

    if "N" in df.columns:
        df["N"] = pd.to_numeric(df["N"], errors="coerce")
        df = df.dropna(subset=["N"]).copy()
    else:
        if "L" not in df.columns:
            raise ValueError("RS csv must contain either N or L.")
        df["L"] = pd.to_numeric(df["L"], errors="coerce")
        df = df.dropna(subset=["L"]).copy()

        L_vals = df["L"].astype(int).tolist()
        df["N"] = [1 << int(L) for L in L_vals]

    if Ns_keep is not None:
        Ns_keep = set(int(n) for n in Ns_keep)
        # df["N"] may be float/object; compare via python int conversion
        df = df[df["N"].apply(lambda x: int(x) in Ns_keep)].copy()

    df["N"] = df["N"].astype(np.int64)

    if "acc_rep" in df.columns:
        df["acc_rep"] = pd.to_numeric(df["acc_rep"], errors="coerce")
    else:
        if not {"m", "q"}.issubset(set(df.columns)):
            raise ValueError("RS csv missing 'acc_rep' and also missing 'm'/'q' to compute it.")
        m = pd.to_numeric(df["m"], errors="coerce").to_numpy()
        q = pd.to_numeric(df["q"], errors="coerce").to_numpy()
        q = np.maximum(q, 1e-15)
        df["acc_rep"] = np.array([Phi(mi / math.sqrt(qi)) for mi, qi in zip(m, q)], dtype=float)

    df = df.dropna(subset=["acc_rep"]).copy()
    return df


def load_experiment_results_csv(path: str) -> pd.DataFrame:
    """
    results.csv format:
      kind, model, N, epsilon, value, sem, run, R
      use model=perceptron and kind in {avg,single}.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    need = {"kind", "model", "N", "epsilon", "value"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    df = df[df["model"].isin(["perceptron"])].copy()
    df = df[df["kind"].isin(["avg", "single"])].copy()

    df["N"] = df["N"].astype(int)
    df["epsilon"] = df["epsilon"].astype(float)
    df["value"] = df["value"].astype(float)
    if "sem" in df.columns:
        df["sem"] = pd.to_numeric(df["sem"], errors="coerce")
    else:
        df["sem"] = np.nan
    return df


def pick_avg_else_single(dfN: pd.DataFrame) -> pd.DataFrame:
    """Prefer avg if present, else single, for each epsilon."""
    kind_order = {"avg": 0, "single": 1}
    dfN = dfN.copy()
    dfN["kind_rank"] = dfN["kind"].map(kind_order).fillna(9).astype(int)
    dfN = dfN.sort_values(["epsilon", "kind_rank"])
    out = dfN.groupby("epsilon", as_index=False).first()
    return out[["epsilon", "value", "sem", "kind"]].sort_values("epsilon")


def plot_panels_by_alpha(out_png: str,
                         exp_data: dict,
                         rs_df: pd.DataFrame,
                         Ns: list,
                         alphas: list):
    """
    One panel per alpha. In each panel, plot curves for all N:
      - Experiments: markers, and SEM bars when available
      - RS: solid line
    """
    n = len(alphas)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    plt.figure(figsize=(12, 4.2 * nrows))

    eps_KS = (1.0 - 1.0 / math.sqrt(2.0)) / 2.0

    # consistent colors for N across all panels + consistent exp/rs pairing
    color_N = _stable_color_map([int(N) for N in Ns])

    for i, a in enumerate(alphas, 1):
        ax = plt.subplot(nrows, ncols, i)

        legend_handles = []
        legend_labels = []

        for N in Ns:
            N = int(N)
            key = (float(a), N)
            c = color_N[N]

            # RS theory
            sub_rs = rs_df[(rs_df["N"] == N) & (np.isclose(rs_df["alpha"], float(a)))].sort_values("eps")
            if not sub_rs.empty:
                h_rs = ax.plot(sub_rs["eps"], sub_rs["acc_rep"],
                               linestyle="-", linewidth=2.0, color=c,
                               label=f"RS N={N}")[0]
                legend_handles.append(h_rs)
                legend_labels.append(f"RS N={N}")

            # experiments
            if key in exp_data:
                d = exp_data[key]
                lab_exp = f"Exp N={N}"
                if d["sem"].notna().any() and np.nanmax(d["sem"].to_numpy()) > 0:
                    h_exp = ax.errorbar(d["epsilon"], d["value"], yerr=d["sem"],
                                        fmt="o", capsize=3, markersize=4,
                                        color=c, ecolor=c,
                                        label=lab_exp)
                else:
                    h_exp = ax.plot(d["epsilon"], d["value"],
                                    marker="o", linestyle="None", markersize=4,
                                    color=c, label=lab_exp)[0]
                legend_handles.append(h_exp)
                legend_labels.append(lab_exp)

        ax.axhline(0.5, linestyle="--", linewidth=1, color="gray")
        ax.axvline(eps_KS, linestyle=":", linewidth=1.2, color="black")
        ax.set_title(f"α={float(a):g}")
        ax.set_xlabel("ε")
        ax.set_ylabel("test accuracy")
        ax.grid(True, alpha=0.2)

        ax.legend(legend_handles, legend_labels, fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_panels_by_L(out_png: str,
                     exp_data: dict,
                     rs_df: pd.DataFrame,
                     Ls: list,
                     alphas: list):
    """
    One panel per L. In each panel, plot curves for all alpha:
      - Experiments: markers, and SEM bars when available
      - RS: solid line
    """
    n = len(Ls)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    plt.figure(figsize=(12, 4.2 * nrows))

    eps_KS = (1.0 - 1.0 / math.sqrt(2.0)) / 2.0

    # consistent colors for alpha across all panels + consistent exp/rs pairing
    color_a = _stable_color_map([float(a) for a in alphas])

    for i, L in enumerate(Ls, 1):
        ax = plt.subplot(nrows, ncols, i)
        N = 2 ** int(L)

        legend_handles = []
        legend_labels = []

        for a in alphas:
            a = float(a)
            key = (a, int(N))
            c = color_a[a]

            # RS theory
            sub_rs = rs_df[(rs_df["N"] == int(N)) & (np.isclose(rs_df["alpha"], a))].sort_values("eps")
            if not sub_rs.empty:
                lab_rs = f"RS α={a:g}"
                h_rs = ax.plot(sub_rs["eps"], sub_rs["acc_rep"],
                               linestyle="-", linewidth=2.0, color=c,
                               label=lab_rs)[0]
                legend_handles.append(h_rs)
                legend_labels.append(lab_rs)

            # experiments
            if key in exp_data:
                d = exp_data[key]
                lab_exp = f"Exp α={a:g}"
                if d["sem"].notna().any() and np.nanmax(d["sem"].to_numpy()) > 0:
                    h_exp = ax.errorbar(d["epsilon"], d["value"], yerr=d["sem"],
                                        fmt="o", capsize=3, markersize=4,
                                        color=c, ecolor=c,
                                        label=lab_exp)
                else:
                    h_exp = ax.plot(d["epsilon"], d["value"],
                                    marker="o", linestyle="None", markersize=4,
                                    color=c, label=lab_exp)[0]
                legend_handles.append(h_exp)
                legend_labels.append(lab_exp)

        ax.axhline(0.5, linestyle="--", linewidth=1, color="gray")
        ax.axvline(eps_KS, linestyle=":", linewidth=1.2, color="black")
        ax.set_title(f"L={int(L)} (N={N})")
        ax.set_xlabel("ε")
        ax.set_ylabel("test accuracy")
        ax.grid(True, alpha=0.2)

        ax.legend(legend_handles, legend_labels, fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rs_csv", type=str, default="rs_saddle_point_sol_lowerlambda_augmented.csv")
    ap.add_argument("--exp_root", type=str, default="./out/ising_q2/out_lbfgs")
    ap.add_argument("--job_id", type=int, required=True)
    ap.add_argument("--alphas", type=float, nargs="+", required=True,
                    help="Alpha values in the same order as SLURM task indices (0..K-1).")
    ap.add_argument("--Ns", type=int, nargs="+", default=None,
                    help="N values to include.")
    ap.add_argument("--Ls", type=int, nargs="+", default=None,
                    help="If provided, use these L values (and set Ns = [2^L]).")
    ap.add_argument("--outdir", type=str, default="rs_vs_exp_plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # decide sizes to plot
    if args.Ls is not None and len(args.Ls) > 0:
        Ls = [int(L) for L in args.Ls]
        Ns = [2 ** int(L) for L in Ls]
    else:
        if args.Ns is None or len(args.Ns) == 0:
            raise ValueError("Provide either --Ns or --Ls.")
        Ns = [int(N) for N in args.Ns]
        Ls = sorted({int(round(math.log2(N))) for N in Ns if N > 0 and (N & (N - 1) == 0)})

    rs_df = load_rs_csv(args.rs_csv)

    # load experiment CSV per alpha, then extract each N
    exp_data = {}  # (alpha, N) -> df
    for task_id, a in enumerate(args.alphas):
        alpha_str = fmt_alpha_folder(float(a))
        csv_path = os.path.join(
            args.exp_root,
            f"alpha_{alpha_str}",
            f"job_{args.job_id}_{task_id}",
            "results.csv"
        )
        if not os.path.isfile(csv_path):
            print(f"[warn] missing experiments csv: {csv_path}")
            continue

        df = load_experiment_results_csv(csv_path)
        for N in Ns:
            dfN = df[df["N"] == int(N)]
            if dfN.empty:
                continue
            picked = pick_avg_else_single(dfN)
            exp_data[(float(a), int(N))] = picked

    if not exp_data:
        raise RuntimeError("No experiment data loaded. Check exp_root, job_id, alphas, and Ns/Ls.")

    # panels by alpha (fixed alpha, varying N)
    out_png_a = os.path.join(args.outdir, "rs_vs_exp_panels_by_alpha.png")
    plot_panels_by_alpha(out_png=out_png_a, exp_data=exp_data, rs_df=rs_df, Ns=Ns, alphas=args.alphas)
    print(f"Wrote {out_png_a}")

    # panels by L (fixed L, varying alpha)
    out_png_L = os.path.join(args.outdir, "rs_vs_exp_panels_by_L.png")
    plot_panels_by_L(out_png=out_png_L, exp_data=exp_data, rs_df=rs_df, Ls=Ls, alphas=args.alphas)
    print(f"Wrote {out_png_L}")


if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def Phi(x):
    """Standard normal CDF."""
    x = np.asarray(x, dtype=float)
    return norm.cdf(x)

def theta_from_eps(eps):
    return 1.0 - 2.0 * np.asarray(eps, dtype=float)

def d_from_L(L):
    return 2 ** np.asarray(L, dtype=int)

def nu0_tree(eps, L):
    th = theta_from_eps(eps)
    th2 = th * th
    L = np.asarray(L, dtype=int)
    a = 2.0 * th2
    denom = 1.0 - a
    # nu0 = (1-th^2) * (1 - a^L) / (1-a), with limit (1-th^2)*L when denom -> 0
    num = (1.0 - th2) * (1.0 - a ** L)
    return np.where(np.abs(denom) > 1e-14, num / denom, (1.0 - th2) * L)

def acc_census_gauss(eps, L):
    th = theta_from_eps(eps)
    th2 = th * th
    L = np.asarray(L, dtype=int)

    x = (2.0 * th2) ** L  # (2 theta^2)^ell

    den0 = (2.0 * th2 - 1.0)
    # kappa(theta) = 1 - theta^2 / (2 theta^2 - 1)
    # gamma(theta) = (1 - theta^2) / (2 theta^2 - 1)
    kappa = np.where(np.abs(den0) > 1e-14, 1.0 - th2 / den0, np.nan)
    gamma = np.where(np.abs(den0) > 1e-14, (1.0 - th2) / den0, np.nan)

    denom = kappa + gamma * x
    arg = np.where((denom > 0) & np.isfinite(denom) & np.isfinite(x), np.sqrt(x / denom), np.nan)
    return Phi(arg)
    

# load RS solutions
IN_CSV = "rs_saddle_point_sol_lowerlambda.csv"
OUT_CSV = "rs_saddle_point_sol_lowerlambda_augmented.csv"
OUTDIR = "rs_plots"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(IN_CSV)

if "ok" in df.columns:
    df = df[df["ok"].astype(bool)].copy()


# RS SNR = s*m/sqrt(q), accuracy = Phi(SNR)
df["snr"] = df["s"] * df["m"] / (np.sqrt(df["q"].clip(lower=0.0)) + 1e-12)
df["acc_rep"] = Phi(df["snr"])
df["test_err_rep"] = 1.0 - df["acc_rep"]

# census Gaussian approximation
df["acc_census_gauss"] = acc_census_gauss(df["eps"].values, df["L"].values)
df["acc_gap"] = df["acc_rep"] - df["acc_census_gauss"]

# save csv
df.to_csv(OUT_CSV, index=False)
print(f"Saved augmented CSV to: {OUT_CSV}")

# plots
eps_KS = (1.0 - 1.0 / np.sqrt(2.0)) / 2.0

def plot_vs_eps(df, alpha, L_list, ycol, title, fname):
    plt.figure()
    for L in L_list:
        sub = df[(np.isclose(df["alpha"], alpha)) & (df["L"] == L)].sort_values("eps")
        if sub.empty:
            continue
        plt.plot(sub["eps"], sub[ycol], marker="o", linewidth=1, markersize=3, label=f"L={L}")
    plt.xlabel("epsilon")
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend()
    plt.axvline(eps_KS, linestyle=":", linewidth=1.2, color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
    plt.close()

def plot_acc_vs_eps_with_census_gauss(df, alpha, L, fname):
    plt.figure()
    sub = df[(np.isclose(df["alpha"], alpha)) & (df["L"] == L)].sort_values("eps")
    if sub.empty:
        print(f"[skip] no data for alpha={alpha}, L={L}")
        return
    plt.plot(sub["eps"], sub["acc_rep"], marker="o", linewidth=1, markersize=3,
             label="RS acc = Φ(s m / √q)")
    plt.plot(sub["eps"], sub["acc_census_gauss"], marker="o", linewidth=1, markersize=3,
             label="Census Gaussian accuracy baseline")
    plt.xlabel("epsilon")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy vs epsilon (alpha={alpha}, L={L})")
    plt.legend()
    plt.axvline(eps_KS, linestyle=":", linewidth=1.2, color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
    plt.close()

def plot_acc_vs_eps_panels_by_L_vary_alpha(df, L_list, alphas, fname, ncols=2):
    n = len(L_list)
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(12, 4 * nrows))

    for i, L in enumerate(L_list, 1):
        ax = plt.subplot(nrows, ncols, i)

        for a in alphas:
            sub = df[(df["L"] == L) & (np.isclose(df["alpha"], a))].sort_values("eps")
            if sub.empty:
                continue
            ax.plot(sub["eps"], sub["acc_rep"], marker="o", linewidth=1.4, markersize=3,
                    label=f"α={a:g}")

        # census Gaussian overlay
        sub0 = df[df["L"] == L].drop_duplicates("eps").sort_values("eps")
        if not sub0.empty:
            ax.plot(sub0["eps"], sub0["acc_census_gauss"], ":", linewidth=2.0,
                    label="Census Gaussian accuracy baseline")

        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_title(f"L={L}")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("RS accuracy  Φ(s m / √q)")
        ax.grid(True, alpha=0.2)
        ax.legend(ncol=2, fontsize=8)
        ax.axvline(eps_KS, linestyle=":", linewidth=1.2, color="black")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
    plt.close()

def plot_quantity_vs_L(df, alpha, eps, ycol, title, fname):
    plt.figure()
    sub = df[(np.isclose(df["alpha"], alpha)) & (np.isclose(df["eps"], eps))].copy()
    if sub.empty:
        print(f"[skip] no data for alpha={alpha}, eps={eps}")
        return
    grp = sub.groupby("L")[ycol].mean().reset_index().sort_values("L")
    plt.plot(grp["L"], grp[ycol], marker="o", linewidth=1)
    plt.xlabel("L")
    plt.ylabel(ycol)
    plt.title(title)
    plt.axvline(eps_KS, linestyle=":", linewidth=1.2, color="black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
    plt.close()

def plot_acc_vs_eps_panels_by_alpha_vary_L(df, alphas, L_list, fname, ncols=2):
    n = len(alphas)
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=(12, 4 * nrows))

    for i, a in enumerate(alphas, 1):
        ax = plt.subplot(nrows, ncols, i)

        for L in L_list:
            sub = df[(df["L"] == L) & (np.isclose(df["alpha"], a))].sort_values("eps")
            if sub.empty:
                continue
            ax.plot(sub["eps"], sub["acc_rep"], marker="o", linewidth=1.4, markersize=3,
                    label=f"L={L}")

        ax.axhline(0.5, linestyle="--", linewidth=1)
        ax.set_title(f"α={a:g}")
        ax.set_xlabel("epsilon")
        ax.set_ylabel("RS accuracy")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
        ax.axvline(eps_KS, linestyle=":", linewidth=1.2, color="black")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
    plt.close()

# choose some representative settings
alpha_pick = float(df["alpha"].max())
L_values = sorted(df["L"].unique())
L_pick = int(max(L_values))

Ls_to_show = [L for L in L_values if L in (6, 8, 10, 12, 100)]
alphas_to_show = [0.01, 0.05, 0.5, 1.0, 5.0, 10.0]

# m vs eps
plot_vs_eps(
    df, alpha=alpha_pick, L_list=Ls_to_show,
    ycol="m",
    title=f"m vs epsilon (alpha={alpha_pick:g})",
    fname=f"m_vs_eps_alpha_{alpha_pick:g}.png",
)

# s*m vs eps
df["sm"] = df["s"] * df["m"]
plot_vs_eps(
    df, alpha=alpha_pick, L_list=Ls_to_show,
    ycol="sm",
    title=f"s*m vs epsilon (alpha={alpha_pick:g})",
    fname=f"sm_vs_eps_alpha_{alpha_pick:g}.png",
)

# SNR vs eps
plot_vs_eps(
    df, alpha=alpha_pick, L_list=Ls_to_show,
    ycol="snr",
    title=f"SNR = s*m/√q vs epsilon (alpha={alpha_pick:g})",
    fname=f"snr_vs_eps_alpha_{alpha_pick:g}.png",
)

# accuracy
plot_acc_vs_eps_with_census_gauss(
    df, alpha=alpha_pick, L=L_pick,
    fname=f"acc_vs_eps_alpha_{alpha_pick:g}_L_{L_pick}.png",
)

# pick one eps for scaling vs L plots
eps_target = 0.05
eps_unique = np.array(sorted(df["eps"].unique()))
eps_pick = float(eps_unique[np.argmin(np.abs(eps_unique - eps_target))])
print("Using eps_pick =", eps_pick, "for L-scaling plots")

plot_quantity_vs_L(
    df, alpha=alpha_pick, eps=eps_pick, ycol="m",
    title=f"m vs L at eps={eps_pick:g} (alpha={alpha_pick:g})",
    fname=f"m_vs_L_eps_{eps_pick:g}_alpha_{alpha_pick:g}.png",
)
plot_quantity_vs_L(
    df, alpha=alpha_pick, eps=eps_pick, ycol="sm",
    title=f"s*m vs L at eps={eps_pick:g} (alpha={alpha_pick:g})",
    fname=f"sm_vs_L_eps_{eps_pick:g}_alpha_{alpha_pick:g}.png",
)
plot_quantity_vs_L(
    df, alpha=alpha_pick, eps=eps_pick, ycol="snr",
    title=f"SNR vs L at eps={eps_pick:g} (alpha={alpha_pick:g})",
    fname=f"snr_vs_L_eps_{eps_pick:g}_alpha_{alpha_pick:g}.png",
)

# panels by L varying alpha
plot_acc_vs_eps_panels_by_L_vary_alpha(
    df, L_list=Ls_to_show, alphas=alphas_to_show,
    fname="acc_vs_eps_panels_by_L_vary_alpha.png", ncols=2
)

# panels by alpha varying L
plot_acc_vs_eps_panels_by_alpha_vary_L(
    df, alphas=alphas_to_show, L_list=Ls_to_show,
    fname="acc_vs_eps_panels_by_alpha.png", ncols=2
)

print(f"Saved plots under: {OUTDIR}/")
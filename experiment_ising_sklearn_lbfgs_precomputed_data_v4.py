import os, math, random, csv, argparse, json
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch only needed for backend=adamw
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    Dataset = object  # type: ignore

# sklearn only needed for backend=sklearn_lbfgs
try:
    from sklearn import linear_model
except Exception:
    linear_model = None


# repro
def seed_all(seed: int, deterministic_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cuda and torch.cuda.is_available():
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


# CLI
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Root dir produced by generate_sequences_q2.py")
    ap.add_argument("--outdir", type=str, default="out", help="Directory for CSV and plots")

    # modes
    ap.add_argument("--mode", type=str, default="eps_curve",
                    choices=["eps_curve", "alpha_curve", "learning_summary", "w_scaling"],
                    help=("eps_curve: accuracy vs epsilon at fixed alpha/P. "
                          "alpha_curve: accuracy vs alpha at fixed eps. "))

    # backend selection
    ap.add_argument("--backend", type=str, default="sklearn_lbfgs",
                    choices=["sklearn_lbfgs", "adamw"],
                    help="Training backend.")

    # RS-alignment toggles
    ap.add_argument("--fit_intercept", type=int, default=0, choices=[0, 1],
                    help="Whether to fit an intercept/bias term. For RS-aligned comparisons use 0.")
    ap.add_argument("--normalize_inputs", type=int, default=1, choices=[0, 1],
                    help="Whether to divide inputs by sqrt(N) (recommended for RS alignment).")

    # regularization knob (theoretical lambda_reg)
    ap.add_argument("--lambda_reg", type=float, default=1.0,
                    help="L2 regularization strength lambda. For sklearn we set C according to reg_convention.")
    ap.add_argument("--reg_convention", type=str, default="sum", choices=["sum", "mean"],
                    help="How to map lambda_reg to sklearn C: sum -> C=1/lambda, mean -> C=P/lambda.")

    # sklearn-lbfgs options
    ap.add_argument("--max_iter", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=1e-7)

    # eps_curve settings
    ap.add_argument("--runs", type=int, default=1, help="Max runs for small-N averaging")
    ap.add_argument("--epochs", type=int, default=10, help="Training epochs (used only for backend=adamw)")
    ap.add_argument("--epochs_large", type=int, default=None, help="Override epochs for N>=1024 (adamw only)")
    ap.add_argument("--batch_size", type=int, default=256, help="Batch size (adamw only)")

    # pick alpha or P; if both are given, P takes precedence
    ap.add_argument("--alpha", type=float, default=None, help="P/N")
    ap.add_argument("--P", type=int, default=None, help="Directly specify P instead of alpha")

    ap.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars")
    ap.add_argument("--seed", type=int, default=123, help="Base RNG seed")

    # grid for eps_curve
    ap.add_argument("--eps_from", type=float, default=0.00)
    ap.add_argument("--eps_to",   type=float, default=0.30)
    ap.add_argument("--eps_step", type=float, default=0.01)

    ap.add_argument("--eps_Ls", type=int, nargs="+", default=None, help="Depths L to use in eps_curve mode "
    "(overrides default Ns_small/Ns_large). Example: --eps_Ls 6 8 10 12",)

    # Grids for Section 3 (learning from samples)
    ap.add_argument("--alpha_list", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    help="Alpha values for alpha_curve / learning_summary.")
    ap.add_argument("--eps_list", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                    help="Epsilon values for alpha_curve / learning_summary.")
    ap.add_argument("--N_list", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
                    help="Which N to use for alpha_curve / learning_summary. Keep small (e.g. 1024) for speed.")
    ap.add_argument("--bayes_ref", type=str, default="bp", choices=["bp", "census", "none"],
                    help="Which structured baseline to overlay as Bayes reference in Section 3 plots.")
    ap.add_argument("--eval_Pmax", type=int, default=5000,
                    help="Max number of test samples used to estimate baseline (BP/census) curves.")

    # AdamW options
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate (AdamW)")
    ap.add_argument("--weight_decay", type=float, default=1e-5,
                    help="AdamW weight_decay (legacy experimental regularization knob; not theory-aligned).")

    # weight/stat saving
    ap.add_argument("--save_weight_stats", action="store_true",
                    help="Save learned weight norm + alignment stats to CSV.")
    ap.add_argument("--save_weights", action="store_true",
                    help="Also save raw weights (npz). Use with care for large sweeps.")

    ap.add_argument("--write_gauss_approx", action="store_true",
                    help="Write census Gaussian approximation to CSV (no training impact).")
    ap.add_argument("--overlay_gauss_approx", action="store_true",
                    help="Overlay census Gaussian approximation on the main eps_curve plot (only for the largest N to avoid clutter).")

    # w_scaling settings
    ap.add_argument("--eps_fixed", type=float, default=0.05,
                    help="Fixed epsilon for w_scaling mode.")
    ap.add_argument("--Ls", type=int, nargs="+", default=[4, 6, 8, 10, 12],
                    help="List of depths L to sweep in w_scaling mode (N=2^L).")
    ap.add_argument("--alphas", type=float, nargs="+", default=[1,2,3,4,5,6,7,8,9,10],
                    help="List of alpha values to sweep in w_scaling mode.")
    ap.add_argument("--scaling_runs", type=int, default=1,
                    help="Number of non-overlapping chunks (runs) per (L,alpha) in w_scaling mode.")

    return ap.parse_args()


# helpers
def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def nu0_tree(L: int, eps: float) -> float:
    """Top eigenvalue along the all-ones direction for the tree covariance (q=2)."""
    theta = 1.0 - 2.0 * eps
    r = 2.0 * theta * theta
    if abs(1.0 - r) < 1e-12:
        return (1.0 - theta * theta) * float(L)
    return (1.0 - theta * theta) * (1.0 - (r ** L)) / (1.0 - r)

def c_mu_tree(L: int, eps: float) -> float:
    """c_mu = ||mu||^2 / d for the surrogate Gaussian-equivalent model."""
    theta = 1.0 - 2.0 * eps
    d = 2.0 ** L
    return d * (theta ** (2 * L))

def acc_gauss_approx_gaussian_census(L: int, eps: float) -> float:
    """census Gaussian approximation baseline: Φ(sqrt(c_mu/nu0))."""
    nu0 = nu0_tree(L, eps)
    c_mu = c_mu_tree(L, eps)
    if nu0 <= 0:
        return 0.5
    return Phi(math.sqrt(max(0.0, c_mu / nu0)))


# misc
USE_TQDM = True

def log(msg: str):
    if USE_TQDM:
        tqdm.write(msg)
    else:
        print(msg)

# q=2, binary tree with branching k=2
k = 2
def epsilon_KS_binary(k: int) -> float:
    return (1.0 - 1.0/math.sqrt(k)) / 2.0
eps_KS = epsilon_KS_binary(k)


# data utils
def load_precomputed(data_root: str, N: int, eps: float):
    base = os.path.join(data_root, f"N{N}", f"eps_{eps:.3f}")
    packed_tr = os.path.join(base, "leaves_train_packed.npy")
    meta_path = os.path.join(base, "meta.json")
    meta = json.load(open(meta_path))
    y_tr = np.load(os.path.join(base, "y_train.npy"), mmap_mode="r")
    y_te = np.load(os.path.join(base, "y_test.npy"),  mmap_mode="r")
    if os.path.exists(packed_tr) and meta.get("packbits", False):
        lt = np.load(packed_tr, mmap_mode="r")
        le = np.load(os.path.join(base,"leaves_test_packed.npy"), mmap_mode="r")
        Pmax = lt.shape[0]
        return ("packed", lt, y_tr, le, y_te, Pmax, N, meta)
    else:
        lt = np.load(os.path.join(base, "leaves_train.npy"), mmap_mode="r")
        le = np.load(os.path.join(base, "leaves_test.npy"),  mmap_mode="r")
        Pmax = lt.shape[0]
        return ("raw", lt, y_tr, le, y_te, Pmax, N, meta)


def dense_X_from_slice(mode: str, leaves: np.ndarray, sl: slice, N: int, normalize_inputs: bool) -> np.ndarray:
    """Return dense float32 spins in {-1,+1}, optionally divided by sqrt(N)."""
    if mode == "raw":
        X = np.asarray(leaves[sl], dtype=np.float32)
    else:
        rows = leaves[sl]
        bits = np.unpackbits(rows, axis=1)[:, :N].astype(np.int8)
        X = (bits << 1) - 1
        X = X.astype(np.float32, copy=False)
    if normalize_inputs:
        X = X / math.sqrt(float(N))
    return X


def census_acc(leaves, y01, sl: slice, mode: str, N: int) -> float:
    """Majority vote on spins (sum > 0)."""
    if mode == "raw":
        sums = leaves[sl].sum(axis=1)
    else:
        rows = leaves[sl]
        bits = np.unpackbits(rows, axis=1)[:, :N]
        sums = (bits.astype(np.int16) * 2 - 1).sum(axis=1)
    preds = np.zeros_like(sums, dtype=np.uint8)
    preds[sums > 0] = 1
    ties = (sums == 0)
    if np.any(ties):
        preds[ties] = np.random.randint(0, 2, size=ties.sum(), dtype=np.uint8)
    return float((preds == y01[sl]).mean())


# BP baseline (finite L, q=2)
def _theta_from_eps(eps: float) -> float:
    return 1.0 - 2.0 * float(eps)


def _bp_parent_magnetization(m1: np.ndarray, m2: np.ndarray, theta: float) -> np.ndarray:
    """Stable parent magnetization update for q=2."""
    numer = theta * (m1 + m2)
    denom = 1.0 + (theta * theta) * (m1 * m2)
    out = np.empty_like(numer, dtype=np.float64)
    good = np.abs(denom) > 1e-12
    np.divide(numer, denom, out=out, where=good)
    out[~good] = np.sign(numer[~good])
    out[~good & (numer == 0.0)] = 0.0
    return np.clip(out, -1.0, 1.0)


def _bp_root_magnetization_from_leaves(leaves_pm: np.ndarray, eps: float) -> np.ndarray:
    """Vectorized bottom-up BP on a binary tree, starting from leaf magnetizations in {-1,+1}."""
    theta = _theta_from_eps(eps)
    m = leaves_pm.astype(np.float64, copy=False)
    P, N = m.shape
    if not (N > 0 and (N & (N - 1) == 0)):
        raise ValueError(f"BP expects N to be a power of 2, got N={N}")
    while N > 1:
        m1 = m[:, 0:N:2]
        m2 = m[:, 1:N:2]
        m = _bp_parent_magnetization(m1, m2, theta)
        N = m.shape[1]
    return m[:, 0]


def bp_acc(leaves, y01, sl: slice, mode: str, N: int, eps: float) -> float:
    """Finite-L BP baseline accuracy for root reconstruction from leaves."""
    if mode == "raw":
        leaves_pm = np.asarray(leaves[sl], dtype=np.int8)
    else:
        rows = leaves[sl]
        bits = np.unpackbits(rows, axis=1)[:, :N].astype(np.int8)
        leaves_pm = (bits << 1) - 1
        leaves_pm = leaves_pm.astype(np.int8, copy=False)
    m_root = _bp_root_magnetization_from_leaves(leaves_pm, float(eps))
    yhat01 = (m_root >= 0.0).astype(np.uint8)
    return float((yhat01 == y01[sl]).mean())


# model stats
def extract_weight_stats_from_wb(w: np.ndarray, b: float, N: int, eps: float, L: Optional[int]) -> Dict[str, float]:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w_norm = float(np.linalg.norm(w))
    w_sum = float(w.sum())
    w_mean = float(w_sum / N)
    cos_ones = float(w_sum / (w_norm * math.sqrt(N))) if w_norm > 0 else 0.0
    out = {
        "w_norm": w_norm,
        "w_norm_over_sqrtN": float(w_norm / math.sqrt(N)),
        "w_mean": w_mean,
        "cos_ones": cos_ones,
        "bias": float(b),
    }
    if L is not None:
        theta = 1.0 - 2.0*eps
        sqrt_cmu = math.sqrt(N) * (theta ** L)
        out["sqrt_cmu"] = float(sqrt_cmu)
        out["m_eff_from_w"] = float(sqrt_cmu * w_mean)
    return out


# backend: sklearn lbfgs
def logreg_sklearn_train_and_eval(Xtr: np.ndarray, ytr01: np.ndarray,
                                  Xte: np.ndarray, yte01: np.ndarray,
                                  fit_intercept: bool,
                                  lambda_reg: float,
                                  reg_convention: str,
                                  max_iter: int,
                                  tol: float) -> Tuple[float, np.ndarray, float]:
    if linear_model is None:
        raise RuntimeError("sklearn is not available. Install scikit-learn or use --backend adamw.")
    ytr = np.asarray(ytr01, dtype=np.int64)
    yte = np.asarray(yte01, dtype=np.int64)

    if lambda_reg <= 0:
        penalty = "none"
        C = 1.0
    else:
        penalty = "l2"
        P = int(Xtr.shape[0])
        if reg_convention == "sum":
            C = 1.0 / float(lambda_reg)
        else:
            C = float(P) / float(lambda_reg)

    reg = linear_model.LogisticRegression(
        penalty=penalty,
        solver="lbfgs",
        fit_intercept=bool(fit_intercept),
        C=float(C),
        max_iter=int(max_iter),
        tol=float(tol),
        verbose=0,
    )
    reg.fit(Xtr, ytr)
    acc = float(reg.score(Xte, yte))
    w = np.squeeze(reg.coef_).astype(np.float64, copy=False)
    b = float(reg.intercept_[0]) if fit_intercept else 0.0
    return acc, w, b


# backend: torch adamw
if torch is not None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class PackedLeavesDataset(Dataset):
        """Unpacks 1 row of packed bits -> float32 spins in {-1,+1}."""
        def __init__(self, packed_rows: np.ndarray, y01: np.ndarray, sl: slice, N: int):
            self.packed = packed_rows
            self.y01 = y01
            self.sl = sl
            self.N = int(N)
            self.n = sl.stop - sl.start
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            j = self.sl.start + idx
            row = self.packed[j]
            bits = np.unpackbits(row)[:self.N].astype(np.int8)
            spins = (bits << 1) - 1
            x = spins.astype(np.float32)
            y = float(self.y01[j])
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    class LeavesDataset(Dataset):
        def __init__(self, leaves: np.ndarray, y01: np.ndarray, sl: slice):
            self.leaves, self.y01, self.sl = leaves, y01, sl
            self.n = sl.stop - sl.start
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            j = self.sl.start + idx
            x = self.leaves[j].astype(np.float32)
            y = float(self.y01[j])
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    class LinearBinary(nn.Module):
        def __init__(self, N: int, fit_intercept: bool, normalize_inputs: bool):
            super().__init__()
            self.N = int(N)
            self.normalize_inputs = bool(normalize_inputs)
            self.fc = nn.Linear(self.N, 1, bias=bool(fit_intercept))
        def forward(self, x):
            if self.normalize_inputs:
                x = x / math.sqrt(float(self.N))
            return self.fc(x).squeeze(1)

    def adamw_train_and_eval(mode, leaves_tr, ytr, leaves_te, yte,
                             tr_sl, te_sl, N_cur,
                             lr, wd, eps, epochs, batch_size,
                             fit_intercept: bool,
                             normalize_inputs: bool,
                             return_wb: bool = False):
        if mode == "packed":
            train_ds = PackedLeavesDataset(leaves_tr, ytr, tr_sl, N_cur)
            test_ds  = PackedLeavesDataset(leaves_te, yte, te_sl, N_cur)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
            test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
        else:
            train_ds = LeavesDataset(leaves_tr, ytr, tr_sl)
            test_ds  = LeavesDataset(leaves_te, yte, te_sl)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader  = DataLoader(test_ds,  batch_size=1024, shuffle=False, num_workers=0)

        model = LinearBinary(N_cur, fit_intercept=fit_intercept, normalize_inputs=normalize_inputs).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.BCEWithLogitsLoss()

        it = range(int(epochs))
        if USE_TQDM:
            it = tqdm(it, total=int(epochs), leave=False, desc=f"ε={eps:.3f} | lr={lr:.0e} | wd={wd:.0e}")
        model.train()
        for _ in it:
            for xb, yb in train_loader:
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                opt.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device); yb = yb.long().to(device)
                logits = model(xb)
                pred = (torch.sigmoid(logits) >= 0.5).long()
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = float(correct / max(1, total))

        if return_wb:
            w = model.fc.weight.detach().cpu().numpy().reshape(-1)
            b = float(model.fc.bias.detach().cpu().numpy().reshape(())) if model.fc.bias is not None else 0.0
            return acc, w, b
        return acc, None, None
else:
    device = None


# train/eval wrapper
def train_and_eval(backend: str,
                   mode: str,
                   leaves_tr: np.ndarray, y_tr: np.ndarray,
                   leaves_te: np.ndarray, y_te: np.ndarray,
                   tr_sl: slice, te_sl: slice, N_cur: int,
                   *,
                   # shared
                   fit_intercept: bool,
                   normalize_inputs: bool,
                   lambda_reg: float,
                   reg_convention: str,
                   # sklearn
                   max_iter: int,
                   tol: float,
                   # adamw
                   lr: float,
                   weight_decay: float,
                   epochs: int,
                   batch_size: int,
                   return_wb: bool = False) -> Tuple[float, Optional[np.ndarray], Optional[float]]:
    if backend == "sklearn_lbfgs":
        Xtr = dense_X_from_slice(mode, leaves_tr, tr_sl, N_cur, normalize_inputs=normalize_inputs)
        Xte = dense_X_from_slice(mode, leaves_te, te_sl, N_cur, normalize_inputs=normalize_inputs)
        acc, w, b = logreg_sklearn_train_and_eval(
            Xtr, y_tr[tr_sl], Xte, y_te[te_sl],
            fit_intercept=fit_intercept,
            lambda_reg=lambda_reg,
            reg_convention=reg_convention,
            max_iter=max_iter,
            tol=tol,
        )
        if return_wb:
            return acc, w, b
        return acc, None, None

    if backend == "adamw":
        if torch is None:
            raise RuntimeError("PyTorch is not available. Install torch or use --backend sklearn_lbfgs.")
        acc, w, b = adamw_train_and_eval(
            mode, leaves_tr, y_tr, leaves_te, y_te,
            tr_sl, te_sl, N_cur=N_cur,
            lr=lr, wd=weight_decay, eps=0.0, epochs=epochs, batch_size=batch_size,
            fit_intercept=fit_intercept, normalize_inputs=normalize_inputs,
            return_wb=return_wb,
        )
        return acc, w, b

    raise ValueError(f"Unknown backend: {backend}")


# eps_curve mode
@dataclass
class StatCurve:
    eps:  np.ndarray
    mean: np.ndarray
    sem:  np.ndarray


def averaged_curves_for_N(N_cur: int, eps_grid: np.ndarray,
                          data_root: str, R: int, epochs: int, batch_size: int,
                          seed_base: int, P_or_alpha,
                          args,
                          save_weight_stats: bool,
                          weight_rows_out: Optional[List[Dict[str, Any]]] = None) -> Tuple[Dict[str, StatCurve], Dict[str, np.ndarray]]:
    if P_or_alpha["P"] is not None:
        P_cur = int(P_or_alpha["P"])
        alpha_cur = P_cur / float(N_cur)
    else:
        alpha_cur = float(P_or_alpha["alpha"])
        P_cur = int(round(alpha_cur * N_cur))
    P_cur = max(1, P_cur)

    E = len(eps_grid)
    acc_p_runs = np.full((R, E), np.nan, dtype=np.float64)
    acc_c_runs = np.full((R, E), np.nan, dtype=np.float64)

    for e_idx, eps in enumerate(tqdm(eps_grid, leave=False, desc=f"N={N_cur} (avg up to {R} runs)") if USE_TQDM else eps_grid):
        mode, leaves_tr, y_tr, leaves_te, y_te, Pmax, _, _meta = load_precomputed(data_root, N_cur, float(eps))
        if P_cur > Pmax:
            raise ValueError(f"Requested P={P_cur} exceeds available Pmax={Pmax} for N={N_cur}, eps={eps:.3f}")

        R_eff = min(R, Pmax // P_cur)
        for r in range(R_eff):
            seed = (seed_base + 100000 * int(N_cur) + 1000 * int(round(float(eps)*100)) + r)
            seed_all(int(seed), deterministic_cuda=False)

            tr_sl = slice(r * P_cur, (r + 1) * P_cur)
            te_sl = slice(r * P_cur, (r + 1) * P_cur)

            acc_p, w, b = train_and_eval(
                args.backend, mode,
                leaves_tr, y_tr, leaves_te, y_te,
                tr_sl, te_sl, N_cur,
                fit_intercept=bool(args.fit_intercept),
                normalize_inputs=bool(args.normalize_inputs),
                lambda_reg=float(args.lambda_reg),
                reg_convention=str(args.reg_convention),
                max_iter=int(args.max_iter),
                tol=float(args.tol),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                epochs=int(epochs),
                batch_size=int(batch_size),
                return_wb=bool(save_weight_stats),
            )

            if save_weight_stats and (w is not None):
                stats = extract_weight_stats_from_wb(w, float(b), N=N_cur, eps=float(eps), L=int(round(math.log2(N_cur))))
                row = {"N": N_cur, "L": int(round(math.log2(N_cur))), "epsilon": float(eps), "alpha": float(alpha_cur),
                       "P": int(P_cur), "run": int(r), "test_acc": float(acc_p),
                       "backend": str(args.backend),
                       "fit_intercept": int(args.fit_intercept),
                       "normalize_inputs": int(args.normalize_inputs),
                       "lambda_reg": float(args.lambda_reg),
                       "reg_convention": str(args.reg_convention),
                       "lr": float(args.lr), "weight_decay": float(args.weight_decay),
                       "max_iter": int(args.max_iter), "tol": float(args.tol)}
                row.update(stats)
                if weight_rows_out is not None:
                    weight_rows_out.append(row)

            acc_c = census_acc(leaves_te, y_te, te_sl, mode, N_cur)
            acc_p_runs[r, e_idx] = acc_p
            acc_c_runs[r, e_idx] = acc_c

        log(f"[avg] N={N_cur:4d} | ε={eps:.2f} | α≈{alpha_cur:.3f} -> "
            f"LogReg mean={np.nanmean(acc_p_runs[:,e_idx]):.4f}, "
            f"Census mean={np.nanmean(acc_c_runs[:,e_idx]):.4f} | R_eff={R_eff}")

    def to_statcurve(A):
        valid = ~np.isnan(A)
        mean = np.array([np.nanmean(A[:,j]) for j in range(A.shape[1])])
        sem  = np.array([
            (np.nanstd(A[:,j], ddof=1) / math.sqrt(valid[:,j].sum())) if valid[:,j].sum() >= 2 else 0.0
            for j in range(A.shape[1])
        ])
        return StatCurve(eps=eps_grid.copy(), mean=mean, sem=sem)

    curves = {"perceptron": to_statcurve(acc_p_runs),
              "census":     to_statcurve(acc_c_runs)}
    raw = {"perceptron": acc_p_runs, "census": acc_c_runs}
    return curves, raw


def single_run_curve_for_N(N_cur: int, eps_grid: np.ndarray,
                           data_root: str, epochs: int, batch_size: int,
                           seed_base: int, P_or_alpha, args) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if P_or_alpha["P"] is not None:
        P_cur = int(P_or_alpha["P"])
    else:
        P_cur = int(round(float(P_or_alpha["alpha"]) * N_cur))
    P_cur = max(1, P_cur)

    eps_list, acc_p_list, acc_c_list = [], [], []
    for eps in (tqdm(eps_grid, leave=False, desc=f"N={N_cur} (single run)") if USE_TQDM else eps_grid):
        mode, leaves_tr, y_tr, leaves_te, y_te, Pmax, _, _meta = load_precomputed(data_root, N_cur, float(eps))
        if P_cur > Pmax:
            raise ValueError(f"Requested P={P_cur} exceeds available Pmax={Pmax} for N={N_cur}, eps={eps:.3f}")
        tr_sl = slice(0, P_cur)
        te_sl = slice(0, P_cur)
        seed = (seed_base + 100000 * int(N_cur) + 1000 * int(round(float(eps)*100)))
        seed_all(int(seed), deterministic_cuda=False)

        acc_p, _, _ = train_and_eval(
            args.backend, mode,
            leaves_tr, y_tr, leaves_te, y_te,
            tr_sl, te_sl, N_cur,
            fit_intercept=bool(args.fit_intercept),
            normalize_inputs=bool(args.normalize_inputs),
            lambda_reg=float(args.lambda_reg),
            reg_convention=str(args.reg_convention),
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            epochs=int(epochs),
            batch_size=int(batch_size),
            return_wb=False,
        )

        acc_c = census_acc(leaves_te, y_te, te_sl, mode, N_cur)

        eps_list.append(float(eps))
        acc_p_list.append(float(acc_p))
        acc_c_list.append(float(acc_c))

        log(f"[single] N={N_cur:4d} | ε={eps:.2f} -> LogReg acc={acc_p:.4f} | Census acc={acc_c:.4f}")

    return {"perceptron": (np.array(eps_list), np.array(acc_p_list)),
            "census":     (np.array(eps_list), np.array(acc_c_list))}


def write_results_csv(csv_path: str,
                      avg_curves: Dict[int, Dict[str, StatCurve]],
                      avg_raw: Dict[int, Dict[str, np.ndarray]],
                      singles: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]],
                      R_runs: int):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "model", "N", "epsilon", "value", "sem", "run", "R"])
        for N_cur, d in avg_curves.items():
            for model in ("perceptron", "census"):
                sc = d[model]
                for e, v, s in zip(sc.eps.tolist(), sc.mean.tolist(), sc.sem.tolist()):
                    w.writerow(["avg", model, N_cur, e, v, s, "", R_runs])
        for N_cur, d in avg_raw.items():
            eps_list = avg_curves[N_cur]["perceptron"].eps.tolist()
            for model in ("perceptron", "census"):
                A = d[model]
                R, E = A.shape
                for r in range(R):
                    for e_idx, e in enumerate(eps_list):
                        val = A[r, e_idx]
                        if np.isnan(val):
                            continue
                        w.writerow(["raw", model, N_cur, e, val, "", r, ""])
        for N_cur, d in singles.items():
            for model in ("perceptron", "census"):
                eps_arr, acc_arr = d[model]
                for e, v in zip(eps_arr.tolist(), acc_arr.tolist()):
                    w.writerow(["single", model, N_cur, e, v, "", "", ""])


def make_and_save_plot(out_png: str, out_pdf: str,
                       avg_curves, singles, R_runs: int,
                       alpha_val: float, P_value: Optional[int] = None,
                       overlay_gauss_approx: bool = False):
    plt.figure(figsize=(10.5, 6.8))
    for N_cur in sorted(avg_curves.keys()):
        sc_p = avg_curves[N_cur]["perceptron"]
        sc_c = avg_curves[N_cur]["census"]
        plt.errorbar(sc_p.eps, sc_p.mean, yerr=sc_p.sem, fmt='-o', capsize=3, linewidth=1.8,
                     label=f"LogReg avg (R≤{R_runs}) N={N_cur}")
        plt.errorbar(sc_c.eps, sc_c.mean, yerr=sc_c.sem, fmt='--x', capsize=3, linewidth=1.4,
                     label=f"Census avg (R≤{R_runs}) N={N_cur}")
    for N_cur in sorted(singles.keys()):
        eps_p, acc_p = singles[N_cur]["perceptron"]; eps_c, acc_c = singles[N_cur]["census"]
        plt.plot(eps_p, acc_p, '-', linewidth=2.3, label=f"LogReg N={N_cur} (single)")
        plt.plot(eps_c, acc_c, '--', linewidth=2.0, label=f"Census N={N_cur} (single)")

    if overlay_gauss_approx:
        Ns_all = list(avg_curves.keys()) + list(singles.keys())
        if Ns_all:
            Nmax = max(Ns_all)
            Lmax = int(round(math.log2(Nmax)))
            if Nmax in singles:
                eps_grid = singles[Nmax]["census"][0]
            else:
                eps_grid = avg_curves[Nmax]["census"].eps
            gauss_approx = np.array([acc_gauss_approx_gaussian_census(Lmax, float(eps)) for eps in eps_grid], dtype=float)
            plt.plot(eps_grid, gauss_approx, ':', linewidth=2.2, label=f"census Gaussian approximation (N={Nmax})")

    plt.axhline(0.5, linestyle='--', linewidth=1, color='gray', label="Chance")
    plt.axvline(eps_KS, color='k', linestyle=':', linewidth=1.2, label=f'ε_KS≈{eps_KS:.3f}')
    plt.xlabel("ε"); plt.ylabel("Test accuracy")
    if P_value is None:
        plt.title(f"Ising (q=2 spins) — α={alpha_val:.3f}")
    else:
        plt.title(f"Ising (q=2 spins) — P={P_value} (α=P/N varies)")
    plt.legend(ncol=2, fontsize=9); plt.grid(True, alpha=0.25); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.savefig(out_pdf)
    plt.close()


# learning from samples plots
def run_alpha_curve(data_root: str, outdir: str, args):
    """Accuracy vs alpha at fixed epsilon (few eps). Adds Bayes reference line + chance."""
    os.makedirs(outdir, exist_ok=True)
    alpha_grid = [float(a) for a in args.alpha_list]
    eps_list = [float(e) for e in args.eps_list]
    N_list = [int(n) for n in args.N_list]

    rows = []
    for N_cur in N_list:
        L_cur = int(round(math.log2(N_cur)))
        for eps in eps_list:
            mode, leaves_tr, y_tr, leaves_te, y_te, Pmax, _, _meta = load_precomputed(data_root, N_cur, float(eps))
            # baseline (estimated on up to eval_Pmax samples)
            P_eval = min(int(args.eval_Pmax), int(Pmax))
            te_eval = slice(0, P_eval)
            if args.bayes_ref == "bp":
                bayes_acc = bp_acc(leaves_te, y_te, te_eval, mode, N_cur, eps)
            elif args.bayes_ref == "census":
                bayes_acc = census_acc(leaves_te, y_te, te_eval, mode, N_cur)
            else:
                bayes_acc = float('nan')

            accs = []
            for alpha in alpha_grid:
                P_cur = max(1, int(round(alpha * N_cur)))
                if P_cur > Pmax:
                    acc = float('nan')
                else:
                    tr_sl = slice(0, P_cur)
                    te_sl = slice(0, P_cur)
                    seed = int(args.seed + 100000 * N_cur + 1000 * int(round(eps * 100)) + int(round(alpha * 100)))
                    seed_all(seed, deterministic_cuda=False)
                    acc, _, _ = train_and_eval(
                        args.backend, mode,
                        leaves_tr, y_tr, leaves_te, y_te,
                        tr_sl, te_sl, N_cur,
                        fit_intercept=bool(args.fit_intercept),
                        normalize_inputs=bool(args.normalize_inputs),
                        lambda_reg=float(args.lambda_reg),
                        reg_convention=str(args.reg_convention),
                        max_iter=int(args.max_iter),
                        tol=float(args.tol),
                        lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        epochs=int(args.epochs if args.epochs_large is None else args.epochs_large),
                        batch_size=int(args.batch_size),
                        return_wb=False,
                    )
                accs.append(acc)
                rows.append({
                    "N": N_cur, "L": L_cur, "epsilon": eps,
                    "alpha": float(alpha), "P": int(round(alpha * N_cur)),
                    "learner_acc": float(acc),
                    "bayes_ref": str(args.bayes_ref), "bayes_acc": float(bayes_acc),
                    "backend": str(args.backend),
                })

            # plot per (N, eps)
            plt.figure(figsize=(7.5, 4.8))
            plt.plot(alpha_grid, accs, marker='o', linewidth=1.8, label="Learner (logreg)")
            if args.bayes_ref != "none":
                plt.axhline(bayes_acc, linestyle='-', linewidth=1.8, label=f"{args.bayes_ref.upper()} reference")
            plt.axhline(0.5, linestyle='--', linewidth=1.0, color='gray', label="Chance")
            plt.xlabel(r"$\alpha=P/N$")
            plt.ylabel("Test accuracy")
            plt.title(f"Accuracy vs α | N={N_cur} (L={L_cur}) | ε={eps:.3f}")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=9)
            plt.tight_layout()
            fn = os.path.join(outdir, f"alpha_curve_N{N_cur}_eps{eps:.3f}.png")
            plt.savefig(fn, dpi=200)
            plt.close()

    # write CSV
    csv_path = os.path.join(outdir, "alpha_curve.csv")
    cols = ["N", "L", "epsilon", "alpha", "P", "learner_acc", "bayes_ref", "bayes_acc", "backend"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    log(f"[io] wrote {csv_path}")


def run_eps_curve_multi_alpha(data_root: str, outdir: str, args):
    """Accuracy vs epsilon for a few alpha values. Adds Bayes reference curve + chance."""
    os.makedirs(outdir, exist_ok=True)
    alpha_list = [float(a) for a in args.alpha_list]
    N_list = [int(n) for n in args.N_list]
    eps_values = np.round(np.arange(args.eps_from, args.eps_to + 1e-12, args.eps_step), 2)

    rows = []
    for N_cur in N_list:
        L_cur = int(round(math.log2(N_cur)))
        for alpha in alpha_list:
            P_cur = max(1, int(round(alpha * N_cur)))
            learner_accs = []
            bayes_accs = []

            for eps in eps_values:
                mode, leaves_tr, y_tr, leaves_te, y_te, Pmax, _, _meta = load_precomputed(data_root, N_cur, float(eps))
                if P_cur > Pmax:
                    learner_accs.append(float('nan'))
                    bayes_accs.append(float('nan'))
                    continue

                tr_sl = slice(0, P_cur)
                te_sl = slice(0, P_cur)
                seed = int(args.seed + 100000 * N_cur + 1000 * int(round(float(eps) * 100)) + int(round(alpha * 100)))
                seed_all(seed, deterministic_cuda=False)
                acc, _, _ = train_and_eval(
                    args.backend, mode,
                    leaves_tr, y_tr, leaves_te, y_te,
                    tr_sl, te_sl, N_cur,
                    fit_intercept=bool(args.fit_intercept),
                    normalize_inputs=bool(args.normalize_inputs),
                    lambda_reg=float(args.lambda_reg),
                    reg_convention=str(args.reg_convention),
                    max_iter=int(args.max_iter),
                    tol=float(args.tol),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    epochs=int(args.epochs if args.epochs_large is None else args.epochs_large),
                    batch_size=int(args.batch_size),
                    return_wb=False,
                )
                learner_accs.append(float(acc))

                # Bayes reference curve estimated on up to eval_Pmax samples
                P_eval = min(int(args.eval_Pmax), int(Pmax))
                te_eval = slice(0, P_eval)
                if args.bayes_ref == "bp":
                    bacc = bp_acc(leaves_te, y_te, te_eval, mode, N_cur, float(eps))
                elif args.bayes_ref == "census":
                    bacc = census_acc(leaves_te, y_te, te_eval, mode, N_cur)
                else:
                    bacc = float('nan')
                bayes_accs.append(float(bacc))

                rows.append({
                    "N": N_cur, "L": L_cur, "alpha": float(alpha), "P": int(P_cur),
                    "epsilon": float(eps), "learner_acc": float(acc),
                    "bayes_ref": str(args.bayes_ref), "bayes_acc": float(bacc),
                    "backend": str(args.backend),
                })

            # plot per (N, alpha)
            plt.figure(figsize=(8.2, 5.0))
            plt.plot(eps_values, learner_accs, marker='o', linewidth=1.8, label="Learner (logreg)")
            if args.bayes_ref != "none":
                plt.plot(eps_values, bayes_accs, linestyle='-', linewidth=2.0, label=f"{args.bayes_ref.upper()} reference")
            plt.axhline(0.5, linestyle='--', linewidth=1.0, color='gray', label="Chance")
            plt.axvline(eps_KS, color='k', linestyle=':', linewidth=1.2, label=f'ε_KS≈{eps_KS:.3f}')
            plt.xlabel(r"$\varepsilon$")
            plt.ylabel("Test accuracy")
            plt.title(f"Accuracy vs ε | N={N_cur} (L={L_cur}) | α={alpha:.2f} (P={P_cur})")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=9)
            plt.tight_layout()
            fn = os.path.join(outdir, f"eps_curve_N{N_cur}_alpha{alpha:.2f}.png")
            plt.savefig(fn, dpi=200)
            plt.close()

    # write CSV
    csv_path = os.path.join(outdir, "eps_curve_multi_alpha.csv")
    cols = ["N", "L", "alpha", "P", "epsilon", "learner_acc", "bayes_ref", "bayes_acc", "backend"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    log(f"[io] wrote {csv_path}")


# w_scaling mode
def run_weight_scaling(data_root: str, outdir: str,
                       eps_fixed: float, Ls: List[int], alphas: List[float],
                       runs: int, epochs: int, batch_size: int,
                       args,
                       save_weights: bool):
    rows: List[Dict[str, Any]] = []
    eps = float(eps_fixed)
    for L in Ls:
        N = 1 << int(L)
        for alpha in alphas:
            P = max(1, int(round(float(alpha) * N)))
            mode, leaves_tr, y_tr, leaves_te, y_te, Pmax, _, _meta = load_precomputed(data_root, N, eps)
            if P > Pmax:
                raise ValueError(f"Requested P={P} exceeds Pmax={Pmax} for N={N}, eps={eps:.3f}")
            R_eff = min(runs, Pmax // P)
            if R_eff < 1:
                raise ValueError(f"Not enough samples for any run: P={P}, Pmax={Pmax}")
            for r in range(R_eff):
                seed = (int(args.seed) + 100000 * int(N) + 1000 * int(round(eps*100)) + 17 * int(alpha*10) + r)
                seed_all(int(seed), deterministic_cuda=False)

                tr_sl = slice(r * P, (r + 1) * P)
                te_sl = slice(r * P, (r + 1) * P)

                acc, w, b = train_and_eval(
                    args.backend, mode,
                    leaves_tr, y_tr, leaves_te, y_te,
                    tr_sl, te_sl, N,
                    fit_intercept=bool(args.fit_intercept),
                    normalize_inputs=bool(args.normalize_inputs),
                    lambda_reg=float(args.lambda_reg),
                    reg_convention=str(args.reg_convention),
                    max_iter=int(args.max_iter),
                    tol=float(args.tol),
                    lr=float(args.lr),
                    weight_decay=float(args.weight_decay),
                    epochs=int(epochs),
                    batch_size=int(batch_size),
                    return_wb=True,
                )
                stats = extract_weight_stats_from_wb(np.asarray(w), float(b), N=N, eps=eps, L=L)
                row = {
                    "N": int(N), "L": int(L), "epsilon": float(eps),
                    "alpha": float(alpha), "P": int(P), "run": int(r),
                    "test_acc": float(acc),
                    "backend": str(args.backend),
                    "fit_intercept": int(args.fit_intercept),
                    "normalize_inputs": int(args.normalize_inputs),
                    "lambda_reg": float(args.lambda_reg),
                    "reg_convention": str(args.reg_convention),
                    "lr": float(args.lr), "weight_decay": float(args.weight_decay),
                    "max_iter": int(args.max_iter), "tol": float(args.tol),
                }
                row.update(stats)
                rows.append(row)

                if save_weights:
                    np.savez_compressed(
                        os.path.join(outdir, f"w_N{N}_L{L}_eps{eps:.3f}_alpha{alpha:.3f}_run{r}.npz"),
                        w=np.asarray(w), b=float(b),
                        N=N, L=L, eps=eps, alpha=float(alpha), P=P,
                        backend=str(args.backend),
                        lambda_reg=float(args.lambda_reg),
                        reg_convention=str(args.reg_convention),
                        fit_intercept=bool(args.fit_intercept),
                        normalize_inputs=bool(args.normalize_inputs),
                        lr=float(args.lr), weight_decay=float(args.weight_decay),
                        max_iter=int(args.max_iter), tol=float(args.tol),
                    )
                log(f"[w_scaling] L={L:2d} N={N:5d} eps={eps:.3f} alpha={alpha:.1f} P={P:6d} run={r} "
                    f"acc={acc:.4f} ||w||={stats['w_norm']:.4f} w_mean={stats['w_mean']:.4e}")

    csv_path = os.path.join(outdir, "w_scaling.csv")
    cols = [
        "N","L","epsilon","alpha","P","run","test_acc",
        "backend","fit_intercept","normalize_inputs","lambda_reg","reg_convention",
        "lr","weight_decay","max_iter","tol",
        "w_norm","w_norm_over_sqrtN","w_mean","cos_ones","bias",
        "sqrt_cmu","m_eff_from_w",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    log(f"[io] wrote {csv_path}")

    # plots
    agg: Dict[tuple, Dict[str, list]] = {}
    for r in rows:
        key = (float(r["alpha"]), int(r["L"]))
        agg.setdefault(key, {"w_norm": [], "w_norm_over_sqrtN": [], "test_acc": []})
        agg[key]["w_norm"].append(float(r["w_norm"]))
        agg[key]["w_norm_over_sqrtN"].append(float(r["w_norm_over_sqrtN"]))
        agg[key]["test_acc"].append(float(r["test_acc"]))

    alphas_sorted = sorted({k[0] for k in agg.keys()})
    Ls_sorted = sorted({k[1] for k in agg.keys()})

    def mean_or_nan(xs):
        return float(np.mean(xs)) if len(xs) else float("nan")

    plt.figure(figsize=(7.5, 4.8))
    for alpha in alphas_sorted:
        ys = [mean_or_nan(agg.get((alpha, L), {"w_norm": []})["w_norm"]) for L in Ls_sorted]
        plt.plot(Ls_sorted, ys, marker='o', linewidth=1.6, label=f"α={alpha:g}")
    plt.xlabel("L"); plt.ylabel(r"$\|w\|_2$")
    plt.title(f"Weight norm vs depth (eps={eps:.3f})")
    plt.grid(True, alpha=0.25); plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "w_norm_vs_L.png"), dpi=200); plt.close()

    plt.figure(figsize=(7.5, 4.8))
    for alpha in alphas_sorted:
        ys = [mean_or_nan(agg.get((alpha, L), {"w_norm_over_sqrtN": []})["w_norm_over_sqrtN"]) for L in Ls_sorted]
        plt.plot(Ls_sorted, ys, marker='o', linewidth=1.6, label=f"α={alpha:g}")
    plt.xlabel("L"); plt.ylabel(r"$\|w\|_2 / \sqrt{N}$")
    plt.title(f"(||w||/sqrt(N)) vs depth (eps={eps:.3f})")
    plt.grid(True, alpha=0.25); plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "w_norm_over_sqrtN_vs_L.png"), dpi=200); plt.close()

    plt.figure(figsize=(7.5, 4.8))
    for alpha in alphas_sorted:
        ys = [mean_or_nan(agg.get((alpha, L), {"test_acc": []})["test_acc"]) for L in Ls_sorted]
        plt.plot(Ls_sorted, ys, marker='o', linewidth=1.6, label=f"α={alpha:g}")
    plt.xlabel("L"); plt.ylabel("Test accuracy")
    plt.title(f"Accuracy vs depth (eps={eps:.3f})")
    plt.grid(True, alpha=0.25); plt.legend(ncol=2, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "acc_vs_L.png"), dpi=200); plt.close()
    log("[io] wrote plots: w_norm_vs_L.png, w_norm_over_sqrtN_vs_L.png, acc_vs_L.png")


# main
if __name__ == "__main__":
    args = get_args()
    USE_TQDM = not args.no_tqdm
    seed_all(int(args.seed), deterministic_cuda=True)
    os.makedirs(args.outdir, exist_ok=True)

    log(f"Computed ε_KS for q=2, k={k}: {eps_KS:.8f}")
    log(f"[cfg] backend={args.backend} fit_intercept={args.fit_intercept} normalize_inputs={args.normalize_inputs} "
        f"lambda_reg={args.lambda_reg} reg_convention={args.reg_convention}")

    if args.backend == "sklearn_lbfgs":
        if linear_model is None:
            raise RuntimeError("scikit-learn not available in this environment.")
        log(f"[cfg] sklearn_lbfgs: max_iter={args.max_iter} tol={args.tol}")
    else:
        if torch is None:
            raise RuntimeError("PyTorch not available in this environment.")
        log(f"[cfg] adamw: lr={args.lr:.3e} weight_decay={args.weight_decay:.3e}")

    if args.mode == "w_scaling":
        run_weight_scaling(
            data_root=args.data_root,
            outdir=args.outdir,
            eps_fixed=float(args.eps_fixed),
            Ls=[int(x) for x in args.Ls],
            alphas=[float(a) for a in args.alphas],
            runs=int(args.scaling_runs),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            args=args,
            save_weights=bool(args.save_weights),
        )
        log("[done] w_scaling finished")
        raise SystemExit(0)

    if args.mode == "alpha_curve":
        out = os.path.join(args.outdir, "alpha_curve")
        run_alpha_curve(data_root=args.data_root, outdir=out, args=args)
        log("[done] alpha_curve finished")
        raise SystemExit(0)

    if args.mode == "learning_summary":
        out_alpha = os.path.join(args.outdir, "alpha_curve")
        out_eps = os.path.join(args.outdir, "eps_curve_multi_alpha")
        run_alpha_curve(data_root=args.data_root, outdir=out_alpha, args=args)
        run_eps_curve_multi_alpha(data_root=args.data_root, outdir=out_eps, args=args)
        log("[done] learning_summary finished")
        raise SystemExit(0)

    # eps_curve
    default_Ns_small = [64, 128, 256, 512]
    default_Ns_large = [1024, 2048, 4096, 8192]
    if args.eps_Ls is not None:
        Ns_selected = sorted({1 << int(L) for L in args.eps_Ls})
        Ns_small = [N for N in Ns_selected if N <= 512]
        Ns_large = [N for N in Ns_selected if N > 512]
    else:
        Ns_small = default_Ns_small
        Ns_large = default_Ns_large

    # for alpha=0.05, only run N >= 256
    if args.P is None and args.alpha is not None and abs(float(args.alpha) - 0.05) < 1e-12:
        N_min = 256
        Ns_small = [N for N in Ns_small if N >= N_min]
        Ns_large = [N for N in Ns_large if N >= N_min]
        log(f"[cfg] alpha={args.alpha:.3f}: enforcing N >= {N_min} for eps_curve")

    eps_values = np.round(np.arange(args.eps_from, args.eps_to + 1e-12, args.eps_step), 2)
    P_or_alpha = {"P": args.P, "alpha": args.alpha if args.alpha is not None else 1.0}

    avg_curves: Dict[int, Dict[str, StatCurve]] = {}
    avg_raw: Dict[int, Dict[str, np.ndarray]] = {}
    weight_rows: List[Dict[str, Any]] = []

    epochs_small = int(args.epochs)
    epochs_large = int(args.epochs if args.epochs_large is None else args.epochs_large)

    for N_cur in (tqdm(Ns_small, desc="Averaging (small N)") if USE_TQDM else Ns_small):
        curves, raw = averaged_curves_for_N(
            N_cur, eps_values,
            data_root=args.data_root, R=int(args.runs),
            epochs=epochs_small, batch_size=int(args.batch_size),
            seed_base=int(args.seed), P_or_alpha=P_or_alpha,
            args=args,
            save_weight_stats=bool(args.save_weight_stats),
            weight_rows_out=weight_rows,
        )
        avg_curves[N_cur] = curves
        avg_raw[N_cur] = raw

    singles: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for N_cur in (tqdm(Ns_large, desc="Single runs (large N)") if USE_TQDM else Ns_large):
        singles[N_cur] = single_run_curve_for_N(
            N_cur, eps_values,
            data_root=args.data_root, epochs=epochs_large,
            batch_size=int(args.batch_size),
            seed_base=int(args.seed), P_or_alpha=P_or_alpha,
            args=args,
        )

    csv_path = os.path.join(args.outdir, "results.csv")
    write_results_csv(csv_path, avg_curves, avg_raw, singles, R_runs=int(args.runs))
    log(f"[io] wrote {csv_path}")

    if args.save_weight_stats and weight_rows:
        stats_csv = os.path.join(args.outdir, "weight_stats.csv")
        cols = sorted({k for r in weight_rows for k in r.keys()})
        with open(stats_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in weight_rows:
                w.writerow(r)
        log(f"[io] wrote {stats_csv}")

    if args.write_gauss_approx:
        gauss_approx_csv = os.path.join(args.outdir, "gauss_approx_gaussian_census.csv")
        with open(gauss_approx_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["N", "L", "epsilon", "acc_gauss_approx"])
            for N_cur in sorted(set(list(avg_curves.keys()) + list(singles.keys()))):
                L_cur = int(round(math.log2(N_cur)))
                for eps in eps_values.tolist():
                    w.writerow([N_cur, L_cur, float(eps), acc_gauss_approx_gaussian_census(L_cur, float(eps))])
        log(f"[io] wrote {gauss_approx_csv}")

    png_path = os.path.join(args.outdir, "results.png")
    pdf_path = os.path.join(args.outdir, "results.pdf")
    alpha_val = (args.P / 1024.0) if args.P is not None else (args.alpha if args.alpha is not None else 1.0)
    make_and_save_plot(png_path, pdf_path, avg_curves, singles,
                       R_runs=int(args.runs), alpha_val=float(alpha_val), P_value=args.P,
                       overlay_gauss_approx=bool(args.overlay_gauss_approx))
    log("[done] eps_curve finished")

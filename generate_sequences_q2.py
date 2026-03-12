# generates and saves spins leaves and 0/1 labels for each (N,ε). 
# writes separate .npy files for train and test, each of size Pmax

import os, json, argparse
import numpy as np
from numba import njit

@njit
def _broadcast_once_spins(L: int, eps: float, root_spin: int, out_leaves: np.ndarray):
    # root_spin in {-1, +1}; out_leaves shape (2**L,)
    level = np.empty(1, np.int8)
    level[0] = root_spin
    for _ in range(L):
        nxt = np.empty(level.size * 2, np.int8)
        for i in range(level.size):
            a = level[i]
            # child 1
            if np.random.rand() < eps:
                nxt[2*i] = -a
            else:
                nxt[2*i] = a
            # child 2
            if np.random.rand() < eps:
                nxt[2*i+1] = -a
            else:
                nxt[2*i+1] = a
        level = nxt
    # write leaves
    for j in range(level.size):
        out_leaves[j] = level[j]

@njit
def _generate_block(L: int, eps: float, P: int, leaves: np.ndarray, y01: np.ndarray):
    # leaves: (P, 2**L) int8 in {-1,+1}; y01: (P,) uint8 in {0,1}
    N = leaves.shape[1]
    for p in range(P):
        root_spin = 1 if np.random.rand() < 0.5 else -1
        y01[p] = 1 if root_spin == 1 else 0
        _broadcast_once_spins(L, eps, root_spin, leaves[p])

def pack_spins_to_bytes(leaves_int8: np.ndarray) -> np.ndarray:
    # leaves_int8: (P, N) values in {-1, +1}
    bits = (leaves_int8 > 0).astype(np.uint8)              # {-1,+1} -> {0,1}
    packed = np.packbits(bits, axis=1)                     # shape: (P, ceil(N/8))
    return packed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--N", type=int, required=True, help="Number of leaves (must be 2**L)")
    ap.add_argument("--L", type=int, required=True, help="Depth so that N=2**L")
    ap.add_argument("--eps", type=float, nargs="+", required=True, help="List of epsilons")
    ap.add_argument("--Pmax", type=int, required=True, help="Max per split (train/test) so total = 2*Pmax")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--packbits", action="store_true", default=True,
                    help="Store leaves bit-packed (1 bit/spin) to save space")
    args = ap.parse_args()

    assert args.N == (1 << args.L), "N must equal 2**L"
    os.makedirs(args.outdir, exist_ok=True)

    for eps in args.eps:
        base = os.path.join(args.outdir, f"N{args.N}", f"eps_{eps:.3f}")
        os.makedirs(base, exist_ok=True)

        # train
        np.random.seed(args.seed ^ 0xA5A5A5)
        leaves_tr = np.empty((args.Pmax, args.N), dtype=np.int8)
        y_tr = np.empty(args.Pmax, dtype=np.uint8)
        _generate_block(args.L, float(eps), args.Pmax, leaves_tr, y_tr)

        # test (different rng stream)
        np.random.seed((args.seed + 1) ^ 0x5A5A5A)
        leaves_te = np.empty((args.Pmax, args.N), dtype=np.int8)
        y_te = np.empty(args.Pmax, dtype=np.uint8)
        _generate_block(args.L, float(eps), args.Pmax, leaves_te, y_te)

    if args.packbits:
        packed_tr = pack_spins_to_bytes(leaves_tr)
        packed_te = pack_spins_to_bytes(leaves_te)
        np.save(os.path.join(base, "leaves_train_packed.npy"), packed_tr)
        np.save(os.path.join(base, "leaves_test_packed.npy"),  packed_te)
        np.save(os.path.join(base, "y_train.npy"), y_tr)
        np.save(os.path.join(base, "y_test.npy"),  y_te)

        meta = {
            "q": 2, "N": args.N, "L": args.L, "epsilon": float(eps),
            "Pmax_train": int(args.Pmax), "Pmax_test": int(args.Pmax),
            "labels": "0/1",
            "leaves": "packed bits row-major (uint8)",
            "packed_cols": int(packed_tr.shape[1]),
            "packbits": True
        }
        with open(os.path.join(base, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
    else:
        pass

        print(f"[ok] N={args.N} eps={eps:.3f} -> saved to {base}")

if __name__ == "__main__":
    main()
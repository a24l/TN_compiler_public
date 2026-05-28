"""
generate_M_and_dataset.py
=========================
Step 3 of the NFL pipeline.

  1. Generate a target unitary M of size (2^n × 2^n)  — Haar-random
     from U(d^n), the full n-qubit unitary group.
  2. Load the pool from build_pool.py output.
  3. Sample t training input states |x_j⟩ from the pool (MPS construction).
  4. Apply M to every |x_j⟩  →  label  |φ_j⟩ = M |x_j⟩
  5. Save the complete dataset  (X, Phi, M, indices)  to .npz

NFL paper reference  (arXiv:2412.05674):
  M  ∈  U(d^n)     Haar-random global unitary
  |φ_j⟩ = M |x_j⟩  training labels
  Loss = (1/t) Σ_j  ‖ Ψ^PS(|x_j⟩) - M|x_j⟩ ‖²

Usage
-----
  python build_pool.py              # run first — generates pool
  python generate_M_and_dataset.py  # then run this

All parameters are in the USER PARAMETERS section below.
"""

import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from build_pool_dataset_new import load_pool, sample_from_pool

os.makedirs("output", exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  HAAR UNITARY SAMPLER
# ══════════════════════════════════════════════════════════════════

def haar_unitary(size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample one Haar-random unitary of shape (size, size).

    M lives in U(d^n):
      n=1 qubit  → size =  2   (2×2   matrix)
      n=2 qubits → size =  4   (4×4   matrix)
      n=3 qubits → size =  8   (8×8   matrix)
      n=4 qubits → size = 16   (16×16 matrix)   ← NFL paper default
      n=5 qubits → size = 32   (32×32 matrix)
      n=6 qubits → size = 64   (64×64 matrix)

    Algorithm: QR decomposition of a Ginibre matrix + phase fix
    guarantees exact Haar measure.
    """
    Z = rng.standard_normal((size, size)) + \
        1j * rng.standard_normal((size, size))
    Q, R  = np.linalg.qr(Z)
    phase = np.diag(R) / np.abs(np.diag(R))
    return (Q * phase).astype(np.complex128)


def verify_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True if U†U = I to within tol."""
    err = np.max(np.abs(U.conj().T @ U - np.eye(U.shape[0])))
    return bool(err < tol)


# ══════════════════════════════════════════════════════════════════
#  GENERATE TARGET UNITARY M
# ══════════════════════════════════════════════════════════════════

def generate_M_composite(
    n_qubits: int = 4,
    d: int = 2,
    seed: int = 99,
    n_layers: int = 3,          # ← number of M_k factors
    mode: str = "sequential",   # "sequential" | "brickwork" | "blocklocal"
    verbose: bool = True,
) -> np.ndarray:
    """
    Build an aggregate target unitary M = M_K · ... · M_2 · M_1
    from multiple Haar-random factors, each drawn independently.

    Modes
    -----
    "sequential"  : M = M_K @ ... @ M_1,  all M_k ∈ U(2^n)
                    → generic highly-entangled target
    "brickwork"   : each layer alternates even/odd 2Q Haar gates
                    (matches your P_S circuit structure — good baseline)
    "blocklocal"  : M = M_left ⊗ M_right on disjoint qubit halves
                    → structured low-entanglement target, χ ≤ 4

    Parameters
    ----------
    n_layers : number of independent unitary factors K
    """
    rng  = np.random.default_rng(seed)
    dim  = d ** n_qubits

    if verbose:
        print(f"Composite M: mode={mode}, n_layers={n_layers}, n_qubits={n_qubits}")
        print(f"  Each factor M_k ∈ U({dim}), composed as M = M_{n_layers}·...·M_1")

    if mode == "sequential":
        # ── M = M_K @ ... @ M_1, all global unitaries ──────────────
        M = np.eye(dim, dtype=np.complex128)
        for k in range(n_layers):
            Mk = haar_unitary(dim, rng)
            M  = Mk @ M          # left-multiply: M_k acts after M_{k-1}
            if verbose:
                ok = verify_unitary(M)
                print(f"  Layer {k+1}: M_{k+1} applied, M unitary ✓ = {ok}")

    elif mode == "brickwork":
        # ── Each layer = alternating Haar 2-qubit gates ─────────────
        # Matches your P_S brickwork ansatz exactly
        # Layer ℓ even bonds: (0,1), (2,3), ...
        # Layer ℓ odd  bonds: (1,2), (3,4), ...
        M = np.eye(dim, dtype=np.complex128)
        for ell in range(n_layers):
            start = ell % 2          # alternate even/odd
            for k in range(start, n_qubits - 1, 2):
                # Haar U(4) on qubits (k, k+1)
                U2q = haar_unitary(d * d, rng)
                # Embed into full dim×dim space
                I_L = np.eye(d**k,           dtype=np.complex128)
                I_R = np.eye(d**(n_qubits-k-2), dtype=np.complex128)
                gate = np.kron(np.kron(I_L, U2q), I_R)
                M = gate @ M
            if verbose:
                ok = verify_unitary(M)
                print(f"  Brickwork layer {ell+1} (start bond={start}): M unitary ✓ = {ok}")

    elif mode == "blocklocal":
        # ── M = M_left ⊗ M_right on disjoint qubit halves ───────────
        # n_left  = n_qubits // 2  qubits
        # n_right = n_qubits - n_left qubits
        assert n_qubits >= 2, "Need at least 2 qubits for blocklocal"
        n_left  = n_qubits // 2
        n_right = n_qubits - n_left
        M = np.eye(dim, dtype=np.complex128)
        for k in range(n_layers):
            ML = haar_unitary(d**n_left,  rng)
            MR = haar_unitary(d**n_right, rng)
            Mk = np.kron(ML, MR)     # block-local, χ_MPO = 1 at centre cut
            M  = Mk @ M
            if verbose:
                ok = verify_unitary(M)
                print(f"  Block layer {k+1}: M_L∈U({d**n_left}), M_R∈U({d**n_right}), M unitary ✓ = {ok}")
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: sequential, brickwork, blocklocal")

    # Final unitarity check
    residual = np.max(np.abs(M.conj().T @ M - np.eye(dim)))
    if verbose:
        print(f"\nFinal M ∈ U({dim}): ‖M†M - I‖_max = {residual:.2e}  "
              f"{'✓' if residual < 1e-10 else '✗'}")
    return M


# ══════════════════════════════════════════════════════════════════
#  BUILD DATASET:  apply M to training states → labels
# ══════════════════════════════════════════════════════════════════

def build_dataset(
        M:         np.ndarray,
        pool:      np.ndarray,
        n_qubits:  int   = 4,
        t_states:  int   = 50,
        D:         int   = None,
        d:         int   = 2,
        pool_seed: int   = 0,
        save_path: str   = "output/dataset",
        verbose:   bool  = True,
) -> dict:
    """
    Build the complete NFL training dataset.

    For each training sample j:
      |x_j⟩  =  MPS state built by sampling n unitaries from pool
      |φ_j⟩  =  M |x_j⟩                        (the label)

    The loss the variational model Ψ^PS will minimise is:
      L = (1/t) Σ_j  ‖ Ψ^PS(|x_j⟩) - |φ_j⟩ ‖²

    Parameters
    ----------
    M         : np.ndarray  shape (d^n, d^n)  — target unitary
    pool      : np.ndarray  shape (N, Dd, Dd) — pool from build_pool.py
    n_qubits  : must match M size: d^n_qubits == M.shape[0]
    t_states  : training set size (x-axis in NFL Fig. 2)
    D         : bond dimension (inferred from pool if None)
    d         : physical dimension
    pool_seed : RNG seed for sampling from pool
    save_path : output path (without .npz)
    verbose   : print progress

    Returns
    -------
    dict:
      "X"       : (t, d^n)       input states   |x_j⟩
      "Phi"     : (t, d^n)       label states   |φ_j⟩ = M|x_j⟩
      "M"       : (d^n, d^n)     target unitary
      "indices" : (t, n)  int    pool sample indices
      "local_unitaries": (t, n, Dd, Dd)  actual matrices used
    """
    dim = d ** n_qubits
    assert M.shape == (dim, dim), \
        f"M shape {M.shape} does not match d^n = {dim}×{dim}"

    if verbose:
        print("=" * 58)
        print("  Building Training Dataset")
        print("=" * 58)
        print(f"  n_qubits             : {n_qubits}")
        print(f"  t_states (training)  : {t_states}")
        print(f"  Hilbert space dim    : {dim}")
        print(f"  M shape              : {M.shape}")
        print(f"  Pool shape           : {pool.shape}")
        print("-" * 58)

    # ── Sample input states |x_j⟩ from pool ──────────────────────
    samples = sample_from_pool(
        pool      = pool,
        n_qubits  = n_qubits,
        t_states  = t_states,
        D         = D,
        d         = d,
        seed      = pool_seed,
        verbose   = False,
    )
    X               = samples["state_vectors"]    # (t, dim), normalized by build_pool_dataset_new
    indices         = samples["indices"]          # (t, n)
    local_unitaries = samples["local_unitaries"]  # (t, n, Dd, Dd)

    # ── Apply M to each |x_j⟩ → label |φ_j⟩ ─────────────────────
    # M @ x_j  for every j simultaneously via matrix multiply
    # X.T shape: (dim, t)  →  M @ X.T shape: (dim, t)  →  Phi shape: (t, dim)
    x_norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(x_norms <= 1e-14):
        raise ValueError("Cannot build dataset from near-zero sampled input states.")
    X = X / x_norms
    Phi = (M @ X.T).T    # shape (t, dim)

    if verbose:
        # Verify labels are normalised relative to inputs
        x_norms   = np.linalg.norm(X,   axis=1)
        phi_norms = np.linalg.norm(Phi, axis=1)
        print(f"  Input  ‖x_j‖   — min: {x_norms.min():.6f}  "
              f"max: {x_norms.max():.6f}")
        print(f"  Label  ‖φ_j‖   — min: {phi_norms.min():.6f}  "
              f"max: {phi_norms.max():.6f}")
        print(f"  ‖φ_j‖ / ‖x_j‖  — min: {(phi_norms/x_norms).min():.6f}  "
              f"max: {(phi_norms/x_norms).max():.6f}")
        print(f"  (ratio = 1.0 exactly when M is unitary and state is normalised)")

        # Verify M applied correctly: check M†M|x_j⟩ = |x_j⟩
        X_recovered = (M.conj().T @ Phi.T).T
        recovery_err = np.max(np.abs(X_recovered - X))
        print(f"  Recovery check ‖M†φ_j - x_j‖ max: {recovery_err:.2e}  "
              f"{'✓' if recovery_err < 1e-10 else '⚠'}")

    # ── Save dataset ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez(
        save_path,
        X               = X,
        Phi             = Phi,
        M               = M,
        indices         = indices,
        local_unitaries = local_unitaries,
        n_qubits        = np.array(n_qubits),
        t_states        = np.array(t_states),
        d               = np.array(d),
        D               = np.array(D if D is not None else pool.shape[1] // d),
        dim             = np.array(dim),
    )

    if verbose:
        fsize = os.path.getsize(save_path + ".npz") / 1e6
        print("-" * 58)
        print(f"  Saved dataset        : {save_path}.npz  ({fsize:.3f} MB)")
        print(f"  X   shape            : {X.shape}   (inputs)")
        print(f"  Phi shape            : {Phi.shape}   (labels)")
        print(f"  M   shape            : {M.shape}    (target unitary)")
        print(f"  indices shape        : {indices.shape}   (pool sample log)")
        print("=" * 58)

    return {
        "X":               X,
        "Phi":             Phi,
        "M":               M,
        "indices":         indices,
        "local_unitaries": local_unitaries,
    }


def load_dataset(path: str) -> dict:
    """
    Load a dataset saved by build_dataset().

    Usage
    -----
    ds = load_dataset("output/dataset")
    X   = ds["X"]    # (t, dim)  input states
    Phi = ds["Phi"]  # (t, dim)  label states
    M   = ds["M"]    # (dim, dim) target unitary
    """
    p    = path if path.endswith(".npz") else path + ".npz"
    arch = np.load(p)
    return {k: arch[k] for k in arch.files}


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── USER PARAMETERS ────────────────────────────────────────────
    N_QUBITS   = 4      # ← controls M size: 2^N_QUBITS × 2^N_QUBITS
                        #   n=2 →  4×4,  n=3 →  8×8,  n=4 → 16×16
                        #   n=5 → 32×32, n=6 → 64×64
    D          = 2      # MPS bond dimension of training states
    D_PHYS     = 2      # physical dimension (2 = qubits)
    T_STATES   = 6000     # training set size  (t in NFL paper)
    M_SEED     = 99     # seed for M  (change to get a different M)
    POOL_SEED  = 0      # seed for sampling from pool
    POOL_PATH  = "output/pool_mps1_D2_N10k.npz"
    SAVE_PATH  = "output/dataset_M4_D2_t6000_seq"  # output path (without .npz)
    # ── END PARAMETERS ─────────────────────────────────────────────

    # Step 1 — generate M
    M = generate_M_composite(n_qubits=N_QUBITS, d=D_PHYS, seed=M_SEED, verbose=True, mode="sequential")

    # Step 2 — load pool (generated by build_pool.py)
    pool_data = load_pool(POOL_PATH)
    pool      = pool_data["pool"]
    print(f"\nLoaded pool: {pool.shape}")

    # Step 3 — build dataset: sample |x_j⟩ from pool, apply M → |φ_j⟩
    print()
    dataset = build_dataset(
        M         = M,
        pool      = pool,
        n_qubits  = N_QUBITS,
        t_states  = T_STATES,
        D         = D,
        d         = D_PHYS,
        pool_seed = POOL_SEED,
        save_path = SAVE_PATH,
        verbose   = True,
    )

    # Quick dataset summary
    X, Phi = dataset["X"], dataset["Phi"]
    print(f"\nDataset ready:")
    print(f"  X   : {X.shape}   dtype={X.dtype}")
    print(f"  Phi : {Phi.shape}   dtype={Phi.dtype}")
    print(f"  M   : {M.shape}   dtype={M.dtype}")
    print(f"\nExample  j=0:")
    print(f"  |x_0⟩  = {X[0, :4].round(4)} ...")
    print(f"  |φ_0⟩  = {Phi[0, :4].round(4)} ...")

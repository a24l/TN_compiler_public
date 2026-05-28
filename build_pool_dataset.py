"""
build_pool.py
=============
Step 2 of the TN Compiler / NFL project.

Generates a POOL of N Haar-random local unitaries of size (Dd × Dd),
then samples n_qubits unitaries from that pool to build MPS training states.

This exactly follows the NFL paper (arXiv:2412.05674) Fig. 2 construction:
  - Pool of N=100 local unitaries  U^0, U^1, ..., U^{N-1}
  - Each U^k  is a  (Dd × Dd) = (D*d × D*d)  Haar-random unitary
  - To build one training state |x_j>:
      draw n indices  k[j, 0..n-1] from {0,...,N-1}
      |x_j> = contract( U^{k[j,0]}, U^{k[j,1]}, ..., U^{k[j,n-1]} ) |0...0>
  - Repeat t times to get a training set of size t

Usage
-----
from build_pool import generate_pool, sample_from_pool, load_pool

# Generate pool
result = generate_pool(N=100, D=2, d=2, seed=42)
pool   = result["pool"]          # shape (100, 4, 4)

# Sample training states for 4 qubits
samples = sample_from_pool(pool, n_qubits=4, t_states=50)
X = samples["state_vectors"]     # shape (50, 16)  — training inputs
k = samples["indices"]           # shape (50, 4)   — which pool entry per site
"""

import numpy as np
import os


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — Haar-random unitary sampler
# ══════════════════════════════════════════════════════════════════

def haar_unitary(size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample one Haar-random unitary of shape (size, size).

    Algorithm:
      1. Fill matrix Z with i.i.d. complex Gaussians
      2. QR decompose Z = Q · R
      3. Fix column phases using diag(R) / |diag(R)|
         → guarantees exact Haar measure (not biased by QR convention)

    Parameters
    ----------
    size : matrix dimension  (use Dd = D*d for NFL paper)
    rng  : numpy Generator

    Returns
    -------
    U : np.ndarray  shape (size, size)  complex128
    """
    Z = rng.standard_normal((size, size)) + \
        1j * rng.standard_normal((size, size))
    Q, R  = np.linalg.qr(Z)
    phase = np.diag(R) / np.abs(np.diag(R))
    return (Q * phase).astype(np.complex128)


def verify_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True if U†U = I to within tol."""
    return bool(np.max(np.abs(U.conj().T @ U - np.eye(U.shape[0]))) < tol)


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — Generate pool
# ══════════════════════════════════════════════════════════════════

def generate_pool(
        N:         int   = 100,
        D:         int   = 2,
        d:         int   = 2,
        seed:      int   = 42,
        save_path: str   = "output/pool_D2",
        verbose:   bool  = True,
) -> dict:
    """
    Generate N Haar-random local unitaries of size (Dd × Dd).

    Parameters
    ----------
    N         : pool size.  N=100 is the NFL paper default.
    D         : MPS bond dimension.
                D=1 → local unitaries are (d × d), product states.
                D=2 → local unitaries are (Dd × Dd), entangled MPS.
    d         : physical dimension per site. d=2 for qubits.
    seed      : RNG seed for reproducibility.
    save_path : output file path, without .npz extension.
    verbose   : print progress.

    Returns
    -------
    dict:
      "pool"     : np.ndarray  shape (N, Dd, Dd)  complex128
      "metadata" : dict with N, D, d, Dd, seed
    """
    rng = np.random.default_rng(seed)
    Dd  = D * d

    if verbose:
        print("=" * 58)
        print("  Generating Local Unitary Pool")
        print("=" * 58)
        print(f"  N  (pool size)       : {N}")
        print(f"  D  (bond dim)        : {D}")
        print(f"  d  (physical dim)    : {d}  {'← qubits' if d==2 else ''}")
        print(f"  Dd (unitary size)    : {Dd} × {Dd}")
        print(f"  Seed                 : {seed}")
        print(f"  Save                 : {save_path}.npz")
        print("-" * 58)

    pool       = np.zeros((N, Dd, Dd), dtype=np.complex128)
    n_verified = 0

    for k in range(N):
        U = haar_unitary(Dd, rng)
        if verify_unitary(U):
            n_verified += 1
        else:
            print(f"  ⚠ U^{k} failed unitarity check!")
        pool[k] = U
        if verbose and (k + 1) % max(1, N // 10) == 0:
            print(f"  Generated {k+1:>5d} / {N}   verified: {n_verified}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez(save_path, pool=pool,
             N=np.array(N), D=np.array(D), d=np.array(d),
             Dd=np.array(Dd),
             seed=np.array(seed if seed is not None else -1))

    if verbose:
        fsize = os.path.getsize(save_path + ".npz") / 1e6
        print("-" * 58)
        print(f"  Verified             : {n_verified} / {N}")
        print(f"  Saved                : {save_path}.npz  ({fsize:.3f} MB)")
        print("=" * 58)

    return {"pool": pool,
            "metadata": dict(N=N, D=D, d=d, Dd=Dd, seed=seed)}


def load_pool(path: str) -> dict:
    """
    Load a pool saved by generate_pool().

    Usage
    -----
    result = load_pool("pool")
    pool   = result["pool"]      # shape (N, Dd, Dd)
    meta   = result["metadata"]
    """
    p    = path if path.endswith(".npz") else path + ".npz"
    arch = np.load(p)
    return {
        "pool": arch["pool"],
        "metadata": {k: int(arch[k]) for k in ["N", "D", "d", "Dd", "seed"]},
    }


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — MPS contraction helpers
# ══════════════════════════════════════════════════════════════════

def _pool_unitary_to_mps_tensor(U: np.ndarray, D: int, d: int) -> np.ndarray:
    """
    Reshape a (Dd × Dd) pool unitary into an MPS tensor (D, d, D).

    Rows index (left_bond α, physical σ): row = α*d + σ
    Columns index right_bond β.
    We keep only the first D columns as the right bond.
    """
    Dd = D * d
    A  = U.reshape(D, d, Dd)   # (D, d, Dd)
    return A[:, :, :D]          # (D, d, D)


def _contract_mps_chain(tensors: list, d: int) -> np.ndarray:
    """
    Contract a list of MPS tensors (D_l, d, D_r) into a state vector.
    Left boundary: left bond = 0.  Right boundary: right bond = 0.
    """
    # Start: left bond = 0 → take first slice → (d, D_r)
    state = tensors[0][0, :, :]           # (d, D_r)
    for A in tensors[1:]:
        dim_so_far = state.shape[0]        # d^site
        # state: (dim_so_far, D_l)
        # A    : (D_l, d, D_r)
        state = np.einsum("ia,asr->isr", state, A)          # (dim_so_far, d, D_r)
        state = state.reshape(dim_so_far * d, A.shape[2])   # (dim_so_far*d, D_r)
    return state[:, 0]   # right bond = 0  → shape (d^n,)


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — Sample from pool → build training states
# ══════════════════════════════════════════════════════════════════

def sample_from_pool(
        pool:     np.ndarray,
        n_qubits: int  = 4,
        t_states: int  = 20,
        D:        int  = None,
        d:        int  = 2,
        seed:     int  = 0,
        verbose:  bool = True,
) -> dict:
    """
    Sample t_states MPS training states from the pool.

    For each sample j in {0,...,t-1}:
      1. Draw n_qubits indices  k[j,q] from {0,...,N-1}
      2. Fetch  U^{k[j,q]}  from the pool  for each site q
      3. Compute |x_j> by contracting the MPS chain

    Parameters
    ----------
    pool     : np.ndarray  shape (N, Dd, Dd)
    n_qubits : number of qubits / MPS sites         default 4
    t_states : number of training states to generate
    D        : bond dimension (inferred from pool if None)
    d        : physical dimension                    default 2
    seed     : RNG seed for the sampling step
    verbose  : print progress

    Returns
    -------
    dict:
      "indices"          (t, n)         int      — pool index per site per sample
      "local_unitaries"  (t, n, Dd, Dd) complex  — actual matrices used
      "state_vectors"    (t, d^n)       complex  — full state |x_j>
      "metadata"         dict
    """
    N  = pool.shape[0]
    Dd = pool.shape[1]
    if D is None:
        D = Dd // d
    assert D * d == Dd, f"D*d={D*d} ≠ Dd={Dd}"
    dim = d ** n_qubits
    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 58)
        print("  Sampling Training States from Pool")
        print("=" * 58)
        print(f"  Pool size N          : {N}")
        print(f"  n_qubits             : {n_qubits}")
        print(f"  t_states (training)  : {t_states}")
        print(f"  D  (bond dim)        : {D}")
        print(f"  d  (physical dim)    : {d}")
        print(f"  Local unitary size   : {Dd} × {Dd}")
        print(f"  State vector dim     : {d}^{n_qubits} = {dim}")
        print("-" * 58)

    indices         = rng.integers(0, N, size=(t_states, n_qubits))
    local_unitaries = np.zeros((t_states, n_qubits, Dd, Dd), dtype=np.complex128)
    state_vectors   = np.zeros((t_states, dim),               dtype=np.complex128)

    ket0_d = np.zeros(d, dtype=np.complex128)
    ket0_d[0] = 1.0

    for j in range(t_states):
        for q in range(n_qubits):
            local_unitaries[j, q] = pool[indices[j, q]]

        if D == 1:
            # Product state: kron of (U^k @ |0>)
            state = None
            for q in range(n_qubits):
                ls = local_unitaries[j, q] @ ket0_d
                state = ls if state is None else np.kron(state, ls)
        else:
            # MPS contraction
            tensors = [_pool_unitary_to_mps_tensor(local_unitaries[j, q], D, d)
                       for q in range(n_qubits)]
            state = _contract_mps_chain(tensors, d)

        state_vectors[j] = state

        if verbose and (j + 1) % max(1, t_states // 5) == 0:
            norm = np.linalg.norm(state)
            print(f"  Sample {j+1:>4d}/{t_states}  "
                  f"k={indices[j].tolist()}  ‖ψ‖={norm:.6f}")

    if verbose:
        norms = np.linalg.norm(state_vectors, axis=1)
        print("-" * 58)
        print(f"  Norm stats:  min={norms.min():.6f}  "
              f"max={norms.max():.6f}  mean={norms.mean():.6f}")
        print("=" * 58)

    return {
        "indices":          indices,
        "local_unitaries":  local_unitaries,
        "state_vectors":    state_vectors,
        "metadata": dict(N=N, D=D, d=d, n_qubits=n_qubits,
                         t_states=t_states, seed=seed),
    }


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── USER PARAMETERS ────────────────────────────────────────────
    N        = 10000    # pool size  (NFL paper uses 100)
    D        = 2      # bond dim   (NFL paper uses D=2)
    d        = 2      # qubits
    n_qubits = 4      # system sizemp
    t_states = 50     # training set size (x-axis in NFL Fig. 2)
    seed     = 42
    # ───────────────────────────────────────────────────────────────

    # Step 1: generate pool
    result  = generate_pool(N=N, D=D, d=d, seed=seed, save_path="output/pool_D2_n10k")
    pool    = result["pool"]

    # Step 2: sample training states
    samples = sample_from_pool(pool, n_qubits=n_qubits,
                                t_states=t_states, D=D, d=d, seed=0)

    print("\nTraining input states X:", samples["state_vectors"].shape)
    print("Sampling index table  k:", samples["indices"].shape)
    print("\nNext step → apply target unitary M to get labels y_j = M |x_j>")

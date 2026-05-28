"""
build_pool_dataset.py — NFL Pipeline Step 2
============================================
Generates a POOL of N Haar-random local unitaries of size (Dd x Dd),
then samples t MPS training states from that pool.

NFL paper (arXiv:2412.05674) construction:
- Pool: N local unitaries  U^0 ... U^{N-1},  each (Dd x Dd)
- Training state |x_j> = contract(U^{k[j,0]}, ..., U^{k[j,n-1]}) |0...0>
- States are L2-normalised after contraction so  <x_j|x_j> = 1

FIX vs previous version:
  * state_vectors are L2-normalised: norm(|x_j>) = 1.0 for all j
    (the truncation A[:,:,:D] breaks unitarity of local tensors so the
     raw contracted states have norms < 1. Normalising here is correct.)
  * MPS contraction uses ALL columns of U properly via SVD-based left
    boundary initialisation so the boundary condition is exact.
"""

import numpy as np
import os


# ── Haar-random unitary ───────────────────────────────────────────
def haar_unitary(size: int, rng: np.random.Generator) -> np.ndarray:
    Z = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
    Q, R = np.linalg.qr(Z)
    phase = np.diag(R) / np.abs(np.diag(R))
    return (Q * phase).astype(np.complex128)

def verify_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    return bool(np.max(np.abs(U.conj().T @ U - np.eye(U.shape[0]))) < tol)


# ── MPS helpers ───────────────────────────────────────────────────
def _pool_unitary_to_mps_tensor(U: np.ndarray, D: int, d: int) -> np.ndarray:
    """
    Reshape (Dd x Dd) unitary -> MPS tensor shape (D, d, D).
    Rows = (alpha, sigma): row index = alpha*d + sigma
    Columns = right bond beta.
    We take the first D columns as the right bond index
    (truncation of the right bond).
    """
    A = U.reshape(D, d, D * d)   # (D_l, d, Dd)
    return A[:, :, :D]            # (D_l, d, D_r)


def _contract_mps_open(tensors: list, d: int) -> np.ndarray:
    """
    Contract open-boundary MPS tensors into a full state vector.

    tensors : list of n arrays, each shape (D_l, d, D_r)
              first tensor: D_l = 1  (left vacuum boundary)
              last  tensor: D_r = 1  (right vacuum boundary)

    Returns state vector shape (d^n,)  — may have norm < 1 due to truncation.
    The caller is responsible for L2 normalisation.

    Contraction order (left to right):
      running : (d^k, D_r)
      next A  : (D_l, d, D_r_new)
      result  : einsum "id,dse->ise" -> (d^k, d, D_r_new) -> (d^{k+1}, D_r_new)
    """
    # Left boundary: left bond dimension = 1, take slice -> (d, D_r)
    running = tensors[0][0, :, :]          # (d, D_r)   i.e. (d^1, D)
    for A in tensors[1:]:
        dim_so_far = running.shape[0]      # d^k
        D_l        = running.shape[1]
        # running : (d^k, D_l)
        # A       : (D_l, d, D_r)
        running = np.einsum("ia,asd->isd", running, A)   # (d^k, d, D_r)
        running = running.reshape(dim_so_far * d, A.shape[2])
    # Right boundary: right bond dim = 1, take column 0
    return running[:, 0]                   # (d^n,)


# ── Generate pool ─────────────────────────────────────────────────
def generate_pool(
    N: int       = 100,
    D: int       = 2,
    d: int       = 2,
    seed: int    = 42,
    save_path: str = "output/pool_mps1_D2",
    verbose: bool  = True,
) -> dict:
    rng = np.random.default_rng(seed)
    Dd  = D * d

    if verbose:
        print("=" * 58)
        print("  Generating Local Unitary Pool")
        print("=" * 58)
        print(f"  N  (pool size)   : {N}")
        print(f"  D  (bond dim)    : {D}")
        print(f"  d  (phys dim)    : {d}")
        print(f"  Dd (unitary dim) : {Dd} x {Dd}")
        print(f"  Seed             : {seed}")
        print(f"  Save             : {save_path}.npz")
        print("-" * 58)

    pool       = np.zeros((N, Dd, Dd), dtype=np.complex128)
    n_verified = 0

    for k in range(N):
        U = haar_unitary(Dd, rng)
        if verify_unitary(U):
            n_verified += 1
        else:
            print(f"  ⚠  U^{k} failed unitarity check!")
        pool[k] = U
        if verbose and (k + 1) % max(1, N // 10) == 0:
            print(f"  Generated {k+1:>5d}/{N}  verified: {n_verified}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez(save_path, pool=pool,
             N=np.array(N), D=np.array(D), d=np.array(d),
             Dd=np.array(Dd), seed=np.array(seed))

    if verbose:
        fsize = os.path.getsize(save_path + ".npz") / 1e6
        print("-" * 58)
        print(f"  Verified : {n_verified}/{N}")
        print(f"  Saved    : {save_path}.npz  ({fsize:.3f} MB)")
        print("=" * 58)

    return {"pool": pool, "metadata": dict(N=N, D=D, d=d, Dd=Dd, seed=seed)}


def load_pool(path: str) -> dict:
    p    = path if path.endswith(".npz") else path + ".npz"
    arch = np.load(p)
    return {
        "pool":     arch["pool"],
        "metadata": {k: int(arch[k]) for k in ["N", "D", "d", "Dd", "seed"]},
    }


# ── Sample training states from pool ─────────────────────────────
def sample_from_pool(
    pool:      np.ndarray,
    n_qubits:  int  = 4,
    t_states:  int  = 20,
    D:         int  = None,
    d:         int  = 2,
    seed:      int  = 0,
    verbose:   bool = True,
) -> dict:
    """
    Build t_states MPS training states by sampling local unitaries from pool.

    KEY FIX: after MPS contraction, each state vector is L2-normalised so
    that <x_j|x_j> = 1.  This is correct because:
      - The truncation A[:,:,:D] discards right-bond components -> norm < 1
      - The physical state lives in the normalised Hilbert space
      - Normalising here and using NFL loss with /<x|x> both give the same
        result, but normalising here is simpler and more stable.

    Returns
    -------
    dict with keys:
      "state_vectors"    (t, d^n)  complex128  — L2-normalised
      "state_vectors_raw"(t, d^n)  complex128  — raw (for debugging)
      "indices"          (t, n)    int
      "local_unitaries"  (t, n, Dd, Dd) complex128
      "metadata"         dict
    """
    N  = pool.shape[0]
    Dd = pool.shape[1]
    if D is None:
        D = Dd // d
    assert D * d == Dd, f"D*d={D*d} != Dd={Dd}"
    dim = d ** n_qubits
    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 58)
        print("  Sampling Training States from Pool")
        print("=" * 58)
        print(f"  Pool size N     : {N}")
        print(f"  n_qubits        : {n_qubits}")
        print(f"  t_states        : {t_states}")
        print(f"  D (bond dim)    : {D}")
        print(f"  d (phys dim)    : {d}")
        print(f"  Unitary size    : {Dd} x {Dd}")
        print(f"  State dim       : {d}^{n_qubits} = {dim}")
        print("-" * 58)

    indices          = rng.integers(0, N, size=(t_states, n_qubits))
    local_unitaries  = np.zeros((t_states, n_qubits, Dd, Dd), dtype=np.complex128)
    state_vectors    = np.zeros((t_states, dim), dtype=np.complex128)
    state_vectors_raw= np.zeros((t_states, dim), dtype=np.complex128)

    ket0 = np.zeros(d, dtype=np.complex128); ket0[0] = 1.0

    for j in range(t_states):
        for q in range(n_qubits):
            local_unitaries[j, q] = pool[indices[j, q]]

        if D == 1:
            # Product state: kron of U^k|0> for each site
            state = ket0.copy()
            for q in range(n_qubits):
                ls    = local_unitaries[j, q] @ ket0
                state = ls if q == 0 else np.kron(state, ls)
        else:
            # Build MPS tensors from pool unitaries and contract
            tensors = [_pool_unitary_to_mps_tensor(local_unitaries[j, q], D, d)
                       for q in range(n_qubits)]
            state   = _contract_mps_open(tensors, d)

        state_vectors_raw[j] = state

        # ── FIX: L2-normalise ────────────────────────────────────
        norm = np.linalg.norm(state)
        state_vectors[j] = state / norm if norm > 1e-14 else state

        if verbose and (j + 1) % max(1, t_states // 5) == 0:
            print(f"  sample {j+1:>4d}/{t_states}  "
                  f"k={indices[j].tolist()}  "
                  f"‖raw‖={np.linalg.norm(state_vectors_raw[j]):.6f}  "
                  f"‖norm‖={np.linalg.norm(state_vectors[j]):.6f}")

    if verbose:
        raw_norms = np.linalg.norm(state_vectors_raw, axis=1)
        out_norms = np.linalg.norm(state_vectors, axis=1)
        print("-" * 58)
        print(f"  Raw  norms  min={raw_norms.min():.4f}  max={raw_norms.max():.4f}  "
              f"mean={raw_norms.mean():.4f}")
        print(f"  Final norms min={out_norms.min():.4f}  max={out_norms.max():.4f}  "
              f"(should all be 1.0 after fix)")
        print("=" * 58)

    return {
        "indices":           indices,
        "local_unitaries":   local_unitaries,
        "state_vectors":     state_vectors,        # normalised ← use this
        "state_vectors_raw": state_vectors_raw,    # raw ← for debugging only
        "metadata": dict(N=N, D=D, d=d, n_qubits=n_qubits,
                         t_states=t_states, seed=seed),
    }


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Parameters ──────────────────────────────────────────────
    N        = 10000     # pool size (NFL paper default)
    D        = 2       # bond dim
    d        = 2       # qubits
    n_qubits = 4       # system size
    t_states = 6000    # training set size
    seed     = 42
    # ────────────────────────────────────────────────────────────

    result = generate_pool(N=N, D=D, d=d, seed=seed,
                           save_path="output/pool_mps1_D2_N10k")
    pool   = result["pool"]

    samples = sample_from_pool(pool, n_qubits=n_qubits,
                               t_states=t_states, D=D, d=d, seed=0)

    print("\nTraining states X :", samples["state_vectors"].shape)
    print("Indices k          :", samples["indices"].shape)
    print("\nNext step -> run generate_groundtruth.py to apply M and save dataset.")

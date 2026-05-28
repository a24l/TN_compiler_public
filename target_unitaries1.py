"""
target_unitaries.py
===================
Step 1 of the TN Compiler project.

Generate and store N target unitaries built from random Ising chain circuits.
Each unitary M_k is an n-qubit circuit of:
  - Single-site rotations : Rz·Ry·Rz  (ZYZ), or  Rx·Ry·Rz  (XYZ), or Haar random
  - Two-site couplings    : ZZ / XX / XY / CNOT between nearest neighbours

User-facing function: generate_target_unitaries(...)
Loader function:      load_target_unitaries(path)

Example
-------
from target_unitaries import generate_target_unitaries, load_target_unitaries

data = generate_target_unitaries(
    n_unitaries = 100,
    n_qubits    = 4,
    n_layers    = 3,
    gate_type   = 'ZYZ',    # 'ZYZ' | 'XYZ' | 'haar'
    coupling    = 'ZZ',     # 'ZZ'  | 'XX'  | 'XY'  | 'CNOT'
    seed        = 42,
)

U_0 = data['unitaries'][0]   # first 16×16 unitary matrix
"""

import numpy as np
import os


# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — Single-qubit gate primitives  (2×2 matrices)
# ══════════════════════════════════════════════════════════════════

def gate_Rx(theta: float) -> np.ndarray:
    """Rotation around X axis: exp(-i theta/2 · X)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c,       -1j * s],
                     [-1j * s,      c]], dtype=complex)


def gate_Ry(theta: float) -> np.ndarray:
    """Rotation around Y axis: exp(-i theta/2 · Y)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c,  -s],
                     [s,   c]], dtype=complex)


def gate_Rz(theta: float) -> np.ndarray:
    """Rotation around Z axis: exp(-i theta/2 · Z)"""
    return np.array([[np.exp(-1j * theta / 2),              0],
                     [0,             np.exp(1j * theta / 2)]], dtype=complex)


def gate_haar_single(rng: np.random.Generator) -> np.ndarray:
    """
    Haar-random single-qubit unitary via QR decomposition.
    Samples UNIFORMLY from all possible 2×2 unitaries.
    """
    Z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    Q, R = np.linalg.qr(Z)
    phase = np.diag(R) / np.abs(np.diag(R))
    return Q * phase


def single_site_unitary(gate_type: str,
                         angles: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    """
    Build one 2×2 single-site unitary.

    Parameters
    ----------
    gate_type : 'ZYZ' | 'XYZ' | 'haar'
    angles    : array of 3 floats  (ignored for 'haar')
    rng       : numpy Generator    (used only for 'haar')
    """
    if gate_type == "ZYZ":
        return gate_Rz(angles[0]) @ gate_Ry(angles[1]) @ gate_Rz(angles[2])
    elif gate_type == "XYZ":
        return gate_Rx(angles[0]) @ gate_Ry(angles[1]) @ gate_Rz(angles[2])
    elif gate_type == "haar":
        return gate_haar_single(rng)
    else:
        raise ValueError(f"Unknown gate_type '{gate_type}'. "
                         "Choose from: 'ZYZ', 'XYZ', 'haar'")


# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — Two-qubit coupling gates  (4×4 matrices)
# ══════════════════════════════════════════════════════════════════

def gate_ZZ(theta: float) -> np.ndarray:
    """Ising ZZ: exp(-i theta/2 · Z⊗Z)"""
    p = theta / 2
    return np.diag([np.exp(-1j*p), np.exp(1j*p),
                    np.exp( 1j*p), np.exp(-1j*p)]).astype(complex)


def gate_XX(theta: float) -> np.ndarray:
    """Ising XX: exp(-i theta/2 · X⊗X)"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c,     0,     0, -1j*s],
                     [0,     c, -1j*s,     0],
                     [0, -1j*s,     c,     0],
                     [-1j*s, 0,     0,     c]], dtype=complex)


def gate_XY(theta: float) -> np.ndarray:
    """XY model: exp(-i theta/2 · (X⊗X + Y⊗Y))"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[1,     0,     0, 0],
                     [0,     c, -1j*s, 0],
                     [0, -1j*s,     c, 0],
                     [0,     0,     0, 1]], dtype=complex)


def gate_CNOT() -> np.ndarray:
    """Standard CNOT (discrete — no angle)."""
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)


def two_site_gate(coupling: str, theta: float) -> np.ndarray:
    """Dispatch to the correct two-qubit gate."""
    if coupling == "ZZ":    return gate_ZZ(theta)
    elif coupling == "XX":  return gate_XX(theta)
    elif coupling == "XY":  return gate_XY(theta)
    elif coupling == "CNOT": return gate_CNOT()
    else:
        raise ValueError(f"Unknown coupling '{coupling}'. "
                         "Choose from: 'ZZ', 'XX', 'XY', 'CNOT'")


# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — Embed gates into the full n-qubit Hilbert space
# ══════════════════════════════════════════════════════════════════

def embed_1q_gate(gate_2x2: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """
    Tensor-product the 2×2 gate onto `qubit`, identity on all others.
    e.g. qubit=1, n=3  →  I ⊗ gate ⊗ I   (8×8 matrix)
    """
    ops = [np.eye(2, dtype=complex)] * n_qubits
    ops[qubit] = gate_2x2
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def embed_2q_gate(gate_4x4: np.ndarray, qubit_a: int,
                  qubit_b: int, n_qubits: int) -> np.ndarray:
    """
    Embed a 4×4 two-qubit gate on adjacent sites (qubit_a, qubit_b=qubit_a+1).
    Builds:  I^⊗qubit_a  ⊗  gate  ⊗  I^⊗(n-qubit_b-1)
    """
    assert qubit_b == qubit_a + 1, "Only nearest-neighbour gates supported"
    left  = np.eye(2 ** qubit_a,                  dtype=complex)
    right = np.eye(2 ** (n_qubits - qubit_b - 1), dtype=complex)

    if qubit_a == 0 and qubit_b == n_qubits - 1:
        return gate_4x4
    elif qubit_a == 0:
        return np.kron(gate_4x4, right)
    elif qubit_b == n_qubits - 1:
        return np.kron(left, gate_4x4)
    else:
        return np.kron(np.kron(left, gate_4x4), right)


# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — Build ONE n-qubit unitary
# ══════════════════════════════════════════════════════════════════

def build_one_unitary(n_qubits: int, n_layers: int, gate_type: str,
                      coupling: str, single_angles: np.ndarray,
                      zz_angles: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Construct one n-qubit Ising chain unitary matrix M.

    Circuit per layer:
        [single-site Rz·Ry·Rz on every qubit]
        [ZZ (or chosen coupling) on every nearest-neighbour pair]

    Returns
    -------
    U : np.ndarray  shape (2^n, 2^n)  complex128
    """
    dim = 2 ** n_qubits
    U   = np.eye(dim, dtype=complex)

    for layer in range(n_layers):
        # Single-site layer
        for q in range(n_qubits):
            local_u = single_site_unitary(gate_type,
                                          single_angles[layer, q], rng)
            U = embed_1q_gate(local_u, q, n_qubits) @ U

        # Two-site coupling layer
        for bond in range(n_qubits - 1):
            coupler = two_site_gate(coupling, zz_angles[layer, bond])
            U = embed_2q_gate(coupler, bond, bond + 1, n_qubits) @ U

    return U


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — Unitarity check
# ══════════════════════════════════════════════════════════════════

def verify_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True if U†U = I to within tolerance."""
    residual = np.max(np.abs(U.conj().T @ U - np.eye(U.shape[0])))
    return bool(residual < tol)


# ══════════════════════════════════════════════════════════════════
#  SECTION 6 — Main user-facing generator function
# ══════════════════════════════════════════════════════════════════

def generate_target_unitaries(
        n_unitaries:         int   = 100,
        n_qubits:            int   = 4,
        n_layers:            int   = 3,
        gate_type:           str   = "ZYZ",
        coupling:            str   = "ZZ",
        angle_range_single:  tuple = (0.0, 2 * np.pi),
        angle_range_couple:  tuple = (0.0, np.pi),
        seed:                int   = 42,
        save_path:           str   = "target_unitaries",
        verbose:             bool  = True,
) -> dict:
    """
    Generate N random Ising-chain target unitaries and save to disk.

    Parameters
    ----------
    n_unitaries         : how many unitaries to create          [REQUIRED]
    n_qubits            : number of qubits (matrix = 2^n × 2^n)[REQUIRED]
    n_layers            : circuit depth                         [REQUIRED]
    gate_type           : single-site gate  'ZYZ'|'XYZ'|'haar' [default ZYZ]
    coupling            : two-site gate     'ZZ'|'XX'|'XY'|'CNOT' [default ZZ]
    angle_range_single  : (min, max) in radians for 1q angles  [default 0..2π]
    angle_range_couple  : (min, max) in radians for 2q angles  [default 0..π]
    seed                : RNG seed (int) or None               [default 42]
    save_path           : output file path (no extension)      [default current dir]
    verbose             : print progress                        [default True]

    Returns
    -------
    dict with keys:
      'unitaries'      np.ndarray  (N, 2^n, 2^n)  complex128
      'single_angles'  np.ndarray  (N, L, n, 3)
      'zz_angles'      np.ndarray  (N, L, n-1)
      'metadata'       dict
    """
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits

    if verbose:
        print("=" * 60)
        print("  Generating Target Unitaries")
        print("=" * 60)
        print(f"  N unitaries  : {n_unitaries}")
        print(f"  Qubits n     : {n_qubits}  →  matrix {dim}×{dim}")
        print(f"  Layers L     : {n_layers}")
        print(f"  Gate type    : {gate_type}")
        print(f"  Coupling     : {coupling}")
        print(f"  Angle single : {angle_range_single}")
        print(f"  Angle couple : {angle_range_couple}")
        print(f"  Seed         : {seed}")
        print(f"  Save path    : {save_path}.npz")
        print("-" * 60)

    unitaries     = np.zeros((n_unitaries, dim, dim), dtype=np.complex128)
    single_angles = np.zeros((n_unitaries, n_layers, n_qubits, 3))
    zz_angles     = np.zeros((n_unitaries, n_layers, n_qubits - 1))
    n_verified    = 0

    for k in range(n_unitaries):
        s_ang = rng.uniform(*angle_range_single, (n_layers, n_qubits, 3))
        c_ang = rng.uniform(*angle_range_couple, (n_layers, n_qubits - 1))

        U = build_one_unitary(n_qubits, n_layers, gate_type,
                               coupling, s_ang, c_ang, rng)

        if verify_unitary(U):
            n_verified += 1
        else:
            print(f"  WARNING: U[{k}] failed unitarity check!")

        unitaries[k]     = U
        single_angles[k] = s_ang
        zz_angles[k]     = c_ang

        if verbose and (k + 1) % max(1, n_unitaries // 10) == 0:
            print(f"  [{k+1:>5d}/{n_unitaries}]  verified so far: {n_verified}")

    metadata = dict(n_unitaries=n_unitaries, n_qubits=n_qubits,
                    n_layers=n_layers, hilbert_dim=dim, gate_type=gate_type,
                    coupling=coupling, seed=seed,
                    n_verified_unitary=n_verified)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez(save_path, unitaries=unitaries,
             single_angles=single_angles, zz_angles=zz_angles,
             n_unitaries=np.array(n_unitaries), n_qubits=np.array(n_qubits),
             n_layers=np.array(n_layers), hilbert_dim=np.array(dim),
             seed=np.array(seed if seed is not None else -1))

    if verbose:
        fsize = os.path.getsize(save_path + ".npz") / 1e6
        print("-" * 60)
        print(f"  Verified: {n_verified}/{n_unitaries}")
        print(f"  Saved  : {save_path}.npz  ({fsize:.2f} MB)")
        print("=" * 60)

    return dict(unitaries=unitaries, single_angles=single_angles,
                zz_angles=zz_angles, metadata=metadata)


# ══════════════════════════════════════════════════════════════════
#  SECTION 7 — Loader
# ══════════════════════════════════════════════════════════════════

def load_target_unitaries(path: str) -> dict:
    """
    Load unitaries saved by generate_target_unitaries().

    Usage
    -----
    data = load_target_unitaries("target_unitaries")
    U_0  = data["unitaries"][0]      # shape (16, 16)
    ang  = data["single_angles"][0]  # shape (L, n, 3)
    """
    load_path = path if path.endswith(".npz") else path + ".npz"
    arch = np.load(load_path)
    return dict(
        unitaries     = arch["unitaries"],
        single_angles = arch["single_angles"],
        zz_angles     = arch["zz_angles"],
        metadata      = dict(
            n_unitaries = int(arch["n_unitaries"]),
            n_qubits    = int(arch["n_qubits"]),
            n_layers    = int(arch["n_layers"]),
            hilbert_dim = int(arch["hilbert_dim"]),
            seed        = int(arch["seed"]),
        ),
    )


# ══════════════════════════════════════════════════════════════════
#  QUICK DEMO  (runs when you execute this file directly)
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    data = generate_target_unitaries(
        n_unitaries = 100,    # ← change this to however many you need
        n_qubits    = 4,      # ← 4 qubits → 16×16 matrices
        n_layers    = 3,
        gate_type   = "ZYZ",
        coupling    = "ZZ",
        seed        = 42,
        save_path   = "target_unitaries",
    )

    # Reload and inspect
    loaded = load_target_unitaries("target_unitaries")
    print("\nLoaded shapes:")
    print("  unitaries    :", loaded["unitaries"].shape)
    print("  single_angles:", loaded["single_angles"].shape)
    print("  zz_angles    :", loaded["zz_angles"].shape)
    print("  metadata     :", loaded["metadata"])

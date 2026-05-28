"""
kak_su4.py — KAK1 SU(4) circuit in Qiskit
Implements: U = (A1⊗A0) · exp(i*(kx·XX + ky·YY + kz·ZZ)) · (B1⊗B0)
Reference:  Tucci (2005), arXiv:quant-ph/0507171, Corollary 6
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RXXGate, RYYGate, RZZGate
from qiskit.quantum_info import Operator
from scipy.linalg import expm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from qiskit.quantum_info import Statevector


@dataclass
class MPSResult:
    A: np.ndarray          # (2, χ)  — left site tensor  (qubit 0)
    S: np.ndarray          # (χ,)    — Schmidt / singular values
    B: np.ndarray          # (χ, 2)  — right site tensor (qubit 1)
    chi: int               # effective bond dimension
    truncation_error: float
    entanglement_entropy: float   # von Neumann entropy (nats)
    is_product_state: bool
    norm: float

def statevector_to_mps(sv: np.ndarray, chi_max: int = None, tol: float = 1e-12) -> MPSResult:
    """
    Decompose a 2-qubit statevector into MPS via SVD.

    Reshape (4,) → (2, 2), apply SVD, optionally truncate bond dim.
    Reconstruction: sv ≈ (A @ diag(S) @ B).flatten()
    """
    sv = np.asarray(sv, dtype=complex)
    assert sv.shape == (4,), f"Expected (4,), got {sv.shape}"
    assert np.isclose(np.linalg.norm(sv), 1.0, atol=1e-8), "Statevector must be normalised"

    # Step 1 — reshape to matrix
    M = sv.reshape(2, 2)                       # M[σ₀, σ₁]

    # Step 2 — SVD
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    # U:(2,2), S:(2,), Vh:(2,2)   →   M = U @ diag(S) @ Vh  exactly

    # Step 3 — truncate
    keep = S > tol
    if chi_max is not None:
        idx  = np.argsort(S)[::-1][:chi_max]
        keep = np.zeros(len(S), dtype=bool); keep[idx] = True
    chi    = int(keep.sum())
    S_k    = S[keep];  A = U[:, keep];  B = Vh[keep, :]

    # Step 4 — diagnostics
    lam2    = S_k**2
    entropy = float(-np.sum(lam2 * np.log(np.where(lam2 > 0, lam2, 1.0))))
    return MPSResult(A=A, S=S_k, B=B, chi=chi,
                     truncation_error=float(1 - lam2.sum()),
                     entanglement_entropy=entropy,
                     is_product_state=(chi == 1),
                     norm=float(lam2.sum()))

def mps_reconstruct(mps: MPSResult) -> np.ndarray:
    """Rebuild statevector from MPS tensors."""
    return (mps.A @ np.diag(mps.S) @ mps.B).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Numeric reference (mirrors your PyTorch kak1_su4_gate exactly)
# ─────────────────────────────────────────────────────────────────────────────
def _weyl_generators():
    XX = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=complex)
    YY = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]], dtype=complex)
    ZZ = np.diag([1,-1,-1,1]).astype(complex)
    return XX, YY, ZZ

def zyz_matrix(alpha, beta, gamma):
    """SU(2): Rz(alpha) @ Ry(beta) @ Rz(gamma)"""
    Rz = lambda t: np.array([[np.exp(-1j*t/2), 0], [0, np.exp(1j*t/2)]])
    Ry = lambda t: np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]])
    return Rz(alpha) @ Ry(beta) @ Rz(gamma)

def kak1_su4_matrix(params_left, kxyz, params_right):
    """
    Numeric 4×4 reference matrix — direct translation of the PyTorch version.
    Useful for validation and as a target for decomposition.
    """
    XX, YY, ZZ = _weyl_generators()
    H_weyl = kxyz[0]*XX + kxyz[1]*YY + kxyz[2]*ZZ
    W = expm(1j * H_weyl)

    A1 = zyz_matrix(*params_left[:3]);  A0 = zyz_matrix(*params_left[3:])
    B1 = zyz_matrix(*params_right[:3]); B0 = zyz_matrix(*params_right[3:])
    return np.kron(A1, A0) @ W @ np.kron(B1, B0)


# ─────────────────────────────────────────────────────────────────────────────
# Core circuit builder  (concrete numeric parameters)
# ─────────────────────────────────────────────────────────────────────────────
def kak1_su4_circuit(params_left, kxyz, params_right, barriers=True):
    """
    Build a 2-qubit KAK (Tucci 2005 Corollary 6) QuantumCircuit.

    U = (A1⊗A0) · exp(i*(kx·XX + ky·YY + kz·ZZ)) · (B1⊗B0)

    ┌─────────────────────────────────────────────────────────┐
    │  Qiskit qubit ordering                                  │
    │    q[0]  ↔  "qubit 0"  — A0/B0 (right Kronecker slot)  │
    │    q[1]  ↔  "qubit 1"  — A1/B1 (left  Kronecker slot)  │
    └─────────────────────────────────────────────────────────┘

    ⚠  Gate-ordering note
    Qiskit builds the unitary as U = G_last ··· G_1 (rightmost gate in circuit
    is leftmost in the matrix product).  To realise Rz(α)·Ry(β)·Rz(γ) we
    therefore append them in REVERSE order: rz(γ), ry(β), rz(α).

    Weyl entangler decomposition
    ────────────────────────────
    XX, YY, ZZ mutually commute on the Weyl subspace, so

        exp(i*(kx·XX + ky·YY + kz·ZZ))
            = exp(i·kx·XX) · exp(i·ky·YY) · exp(i·kz·ZZ)

    Qiskit's RXXGate(θ) = exp(−i·θ/2·XX), so exp(i·kx·XX) ≡ RXX(−2·kx).
    Same logic applies to RYY and RZZ.

    Parameters
    ----------
    params_left  : array-like (6,)
        ZYZ angles [alpha1, beta1, gamma1, alpha0, beta0, gamma0]
        for A1 (q[1]) and A0 (q[0]).
    kxyz         : array-like (3,)
        Weyl canonical coordinates [kx, ky, kz].
        Physical range for SU(4) Weyl chamber: 0 ≤ kz ≤ ky ≤ kx ≤ π/4.
    params_right : array-like (6,)
        ZYZ angles [alpha1, beta1, gamma1, alpha0, beta0, gamma0]
        for B1 (q[1]) and B0 (q[0]).
    barriers     : bool, default True
        Insert visual barriers between the three blocks.

    Returns
    -------
    qc : QuantumCircuit   — 2-qubit circuit, 15 free real DOF = dim(su(4))
    """
    pl = list(params_left)
    k  = list(kxyz)
    pr = list(params_right)

    qc = QuantumCircuit(2, name="KAK1_SU4")

    # ── Block 1 — Right local: B1⊗B0 ─────────────────────────────────────
    # Circuit order (reversed): rz(γ), ry(β), rz(α)  →  matrix: Rz(α)·Ry(β)·Rz(γ)
    qc.rz(pr[2], 1); qc.ry(pr[1], 1); qc.rz(pr[0], 1)   # B1 on q[1]
    qc.rz(pr[5], 0); qc.ry(pr[4], 0); qc.rz(pr[3], 0)   # B0 on q[0]
    if barriers: qc.barrier(label="B1⊗B0")

    # ── Block 2 — Weyl entangler ──────────────────────────────────────────
    qc.append(RXXGate(-2 * k[0]), [0, 1])
    qc.append(RYYGate(-2 * k[1]), [0, 1])
    qc.append(RZZGate(-2 * k[2]), [0, 1])
    if barriers: qc.barrier(label="Weyl")

    # ── Block 3 — Left local: A1⊗A0 ──────────────────────────────────────
    qc.rz(pl[2], 1); qc.ry(pl[1], 1); qc.rz(pl[0], 1)   # A1 on q[1]
    qc.rz(pl[5], 0); qc.ry(pl[4], 0); qc.rz(pl[3], 0)   # A0 on q[0]

    return qc


# ─────────────────────────────────────────────────────────────────────────────
# Fully parameterized version  (for variational / MPS training)
# ─────────────────────────────────────────────────────────────────────────────
def kak1_su4_circuit_parametric(prefix="kak"):
    """
    Return a 2-qubit KAK circuit with 15 symbolic Qiskit Parameters.

    Bind with:
        bound_qc = qc.assign_parameters({
            **dict(zip(pL, params_left)),
            **dict(zip(pk, kxyz)),
            **dict(zip(pR, params_right))
        })

    Parameters
    ----------
    prefix : str
        Prefix for parameter names — change when stacking multiple KAK layers
        to avoid name collisions (e.g. prefix="kak0", "kak1", ...).

    Returns
    -------
    qc  : QuantumCircuit         — unbound parametric circuit
    pL  : ParameterVector (6,)   — left  ZYZ angles  [a1,b1,g1, a0,b0,g0]
    pk  : ParameterVector (3,)   — Weyl coordinates  [kx, ky, kz]
    pR  : ParameterVector (6,)   — right ZYZ angles  [a1,b1,g1, a0,b0,g0]
    """
    pL = ParameterVector(f"{prefix}_L", 6)   # left  ZYZ
    pk = ParameterVector(f"{prefix}_k", 3)   # Weyl
    pR = ParameterVector(f"{prefix}_R", 6)   # right ZYZ

    qc = QuantumCircuit(2, name=f"KAK1_SU4_{prefix}")

    # Right local block
    qc.rz(pR[2], 1); qc.ry(pR[1], 1); qc.rz(pR[0], 1)
    qc.rz(pR[5], 0); qc.ry(pR[4], 0); qc.rz(pR[3], 0)
    qc.barrier()

    # Weyl entangler
    qc.append(RXXGate(-2 * pk[0]), [0, 1])
    qc.append(RYYGate(-2 * pk[1]), [0, 1])
    qc.append(RZZGate(-2 * pk[2]), [0, 1])
    qc.barrier()

    # Left local block
    qc.rz(pL[2], 1); qc.ry(pL[1], 1); qc.rz(pL[0], 1)
    qc.rz(pL[5], 0); qc.ry(pL[4], 0); qc.rz(pL[3], 0)

    return qc, pL, pk, pR


# conver the SU4 circuit to gate function
def kak1_su4_gate(params_left, kxyz, params_right):
    qc = kak1_su4_circuit(params_left, kxyz, params_right, barriers=False)
    kak_gate = qc.to_gate(label="KAK1_SU4")
    new_qc = QuantumCircuit(2)
    new_qc.append(kak_gate, [0, 1]) 
    return kak_gate, new_qc


#extract the statevector from a QuantumCircuit as a NumPy array
def get_statevector(qc: QuantumCircuit,
                    initial_state: np.ndarray = None,
                    decimals: int = 6,
                    verbose: bool = True) -> np.ndarray:
    """
    Extract the statevector from a Qiskit QuantumCircuit as a NumPy array.

    Parameters
    ----------
    qc            : QuantumCircuit  — circuit to simulate (no measurements)
    initial_state : np.ndarray or None
                    Optional complex initial state of shape (2**n_qubits,).
                    Defaults to |0...0⟩ if None.
    decimals      : int   — decimal places for printed display (default 6)
    verbose       : bool  — print amplitude table to stdout (default True)

    Returns
    -------
    sv_np : np.ndarray, shape (2**n_qubits,), dtype complex128
    """
    n = qc.num_qubits

    if initial_state is not None:
        sv_init = np.asarray(initial_state, dtype=complex)
        if sv_init.shape != (2**n,):
            raise ValueError(f"initial_state shape {sv_init.shape} != (2^{n}={2**n},)")
        if not np.isclose(np.linalg.norm(sv_init), 1.0, atol=1e-8):
            raise ValueError(f"initial_state not normalised (norm={np.linalg.norm(sv_init):.6f})")
        sv_obj = Statevector(sv_init).evolve(qc)
    else:
        sv_obj = Statevector(qc)          # evolves from |0...0⟩

    sv_np = sv_obj.data                   # np.ndarray complex128

    if verbose:
        dim = len(sv_np)
        print(f"{'─'*58}")
        print(f"  Statevector  |  {n} qubit(s), dim = {dim}")
        print(f"{'─'*58}")
        print(f"  {'Basis':>8}  {'Real':>14}  {'Imag':>14}  {'|amp|²':>10}")
        print(f"  {'─'*8}  {'─'*14}  {'─'*14}  {'─'*10}")
        for idx, amp in enumerate(sv_np):
            prob   = abs(amp)**2
            marker = "  ◀" if prob > 0.01 else ""
            print(f"  {f'|{idx:0{n}b}⟩':>8}  {amp.real:>14.{decimals}f}  "
                  f"{amp.imag:>14.{decimals}f}  {prob:>10.{decimals}f}{marker}")
        print(f"{'─'*58}")
        total = np.sum(np.abs(sv_np)**2)
        print(f"  Σ|amp|²  = {total:.{decimals}f}  "
              f"{'✓ normalised' if np.isclose(total, 1.0) else '✗ NOT normalised'}")
        print(f"{'─'*58}")
        print(f"\n  sv_np:\n  {np.round(sv_np, decimals)}\n")

    return sv_np


def mps_to_circuit(mps: MPSResult, verbose: bool = True) -> QuantumCircuit:
    """
    Build a Qiskit state-prep circuit from MPSResult tensors A, S, B.

    Steps:
      1. Rebuild sv = (A @ diag(S) @ B).flatten()  — exact inversion of SVD
      2. Build 4×4 unitary Q with Q[:,0] = sv via QR decomposition
      3. Embed Q as a Qiskit UnitaryGate on 2 qubits
    """
    # ── Step 1: reconstruct statevector from MPS tensors ─────────────────
    sv_target = (mps.A @ np.diag(mps.S) @ mps.B).flatten()
    sv_target /= np.linalg.norm(sv_target)

    # ── Step 2: build 4×4 unitary Q whose first column = sv_target ───────
    aug     = np.column_stack([sv_target.reshape(4,1), np.eye(4, dtype=complex)[:, :3]])
    Q, R    = np.linalg.qr(aug, mode='complete')

    # Fix sign: enforce Q[:,0] = sv_target (QR can flip it)
    phase   = np.dot(Q[:, 0].conj(), sv_target)
    Q[:, 0] *= np.conj(phase) / abs(phase)

    # Fix global phase: ensure det(Q) = +1
    det = np.linalg.det(Q)
    Q[:, -1] *= np.conj(det) / abs(det)

    # ── Step 3: wrap as Qiskit circuit ───────────────────────────────────
    qc = QuantumCircuit(2, name="MPS_prep")
    qc.unitary(Q, [0, 1], label="U_mps")
    return qc

# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    p_left  = rng.uniform(-np.pi, np.pi, 6)
    kxyz    = rng.uniform(-np.pi/4, np.pi/4, 3)
    p_right = rng.uniform(-np.pi, np.pi, 6)

    # ── Concrete circuit ──────────────────────────────────────────────────
    qc_kak = kak1_su4_circuit(p_left, kxyz, p_right)
    sv_kak = get_statevector(qc_kak, verbose=False)
    #print("KAK1_SU4 circuit:")
    #print(qc.draw('mpl'))
    #print()

    # ── Decompose into MPS (A, S, B from SVD)
    mps_kak = statevector_to_mps(sv_kak, chi_max=1)

    #build state -prep circuit from mps tensors and extract statevector
    qc_prep = mps_to_circuit(mps_kak, verbose=False)

    # ── Gate version ──────────────────────────────────────────────────────
    kak_gate, gate_qc = kak1_su4_gate(p_left, kxyz, p_right)
    #print("KAK1_SU4 as a single gate:")
    #print(gate_qc.draw('mpl'))
    #print()
    
    
    sv_out   = Statevector(qc_prep).data
    fidelity = abs(np.dot(sv_kak.conj(), sv_out))**2
    print(f"Fidelity: {fidelity:.10f}")   # should be ~1.0

    # ── Statevector extraction ───────────────────────────────────────────
    #sv = get_statevector(gate_qc, verbose=True)
    
    # ── MPS decomposition and reconstruction ───────────────────────────
    #mps = statevector_to_mps(sv, chi_max=2)
    #print(f"MPS decomposition:\n{mps}\n")

    #sv_recon = mps_reconstruct(mps)
    #print(f"Reconstructed statevector:\n{sv_recon}\n")
    #print(f"Fidelity with original: {np.abs(np.dot(sv.conj(), sv_recon)):.6f}")
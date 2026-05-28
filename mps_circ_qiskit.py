"""
mps_circuit_qiskit.py
=====================
Build a variable bond-dimension MPS from a random statevector (via SVD),
convert each left-canonical core to a Qiskit UnitaryGate, and draw the
resulting staircase preparation circuit.

Variables:
  N_SITES  -- number of physical qubits
  BOND_DIM -- D (truncated bond dimension; D=4 is exact for n=4,d=2)
  D_PHYS   -- d = 2 (qubits)
  SEED     -- reproducibility

Architecture (left-canonical staircase):
  bond register : 1 qubit  (= log2(D) qubits for D=2)
  phys register : N_SITES qubits

  |0>_b |0>_q0 |0>_q1 |0>_q2 |0>_q3
         [U0]──[U1]────[U2]────[U3]
  bond──────────────────────────────

  Gate U_k acts on [bond, phys_k] for k=1..n-1
  Gate U_0 acts only on phys_0  (D_l=1 so size=d, 1-qubit gate)
"""

import os, numpy as np
from scipy.linalg import qr as scipy_qr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

os.makedirs("output", exist_ok=True)

# ─── USER PARAMETERS ──────────────────────────────────────
N_SITES  = 4
BOND_DIM = 2      # D=1 → product state, D=4 → exact for n=4
D_PHYS   = 2
SEED     = 42
# ──────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════
#  1.  Random statevector → left-canonical MPS via SVD
# ══════════════════════════════════════════════════════════

def statevector_to_lc_mps(psi, n, d, D):
    """
    Left-canonical MPS decomposition via sequential SVD.
    Returns list of n tensors A[k] shape (D_l, d, D_r).
    Left-canonicality: Σ_s A[k]_s† A[k]_s = I_{D_r}
    """
    cores = []
    M = psi.copy().reshape(1, d**n)
    D_l = 1
    for k in range(n - 1):
        M = M.reshape(D_l * d, -1)
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        chi = min(len(S), D)
        U, S, Vh = U[:, :chi], S[:chi], Vh[:chi, :]
        cores.append(U.reshape(D_l, d, chi).astype(np.complex128))
        M  = np.diag(S) @ Vh
        D_l = chi
    cores.append(M.reshape(D_l, d, 1).astype(np.complex128))
    return cores


def contract_mps(cores, d):
    """Contract left-canonical MPS to statevector."""
    C = cores[0][0, :, :]            # (d, D_r)
    for A in cores[1:]:
        C = np.einsum("sd,dtr->str", C, A)
        C = C.reshape(C.shape[0] * C.shape[1], C.shape[2])
    return C[:, 0]


# ══════════════════════════════════════════════════════════
#  2.  Left-canonical core → guaranteed unitary gate
# ══════════════════════════════════════════════════════════

def core_to_unitary(A):
    """
    Convert left-canonical core A (D_l, d, D_r) to a unitary
    matrix of size (D_l*d) × (D_l*d).

    The isometry V: C^{D_r} → C^{D_l*d}  has columns
        V[:, r] = A[:, :, r].flatten()
    and satisfies V†V = I_{D_r}.

    We complete V to a full unitary via Haar-distributed
    complementary columns using scipy QR with column pivoting,
    then fix diagonal phases so U†U = I to 1e-14.
    """
    D_l, d, D_r = A.shape
    size = D_l * d

    # Build isometry columns
    V = np.column_stack([A[:, :, r].flatten(order='C') for r in range(D_r)])
    # V : (size, D_r)

    if size == D_r:
        # Already square — just re-orthogonalise
        Q, R = np.linalg.qr(V)
        phase = np.diag(R) / np.abs(np.diag(R))
        return (Q * phase).astype(np.complex128)

    # Re-orthogonalise the isometry columns
    Q_iso, R_iso = np.linalg.qr(V)          # (size, D_r) orthonormal
    Q_iso = Q_iso * (np.diag(R_iso) / np.abs(np.diag(R_iso)))

    # Build random complement for remaining (size - D_r) columns
    rng = np.random.default_rng(abs(hash(str(A.shape))) % 2**31)
    Z = rng.standard_normal((size, size - D_r)) + \
        1j * rng.standard_normal((size, size - D_r))
    # Project out span(Q_iso)
    Z = Z - Q_iso @ (Q_iso.conj().T @ Z)
    Q_comp, R_comp = scipy_qr(Z, mode='economic', pivoting=False)
    Q_comp = Q_comp * (np.diag(R_comp) / np.abs(np.diag(R_comp)))

    U = np.hstack([Q_iso, Q_comp])          # (size, size)

    # Final polish: guarantee ‖U†U − I‖ < 1e-12
    # One Newton step: U ← U (I + (I − U†U)/2)
    err = np.eye(size) - U.conj().T @ U
    U = U @ (np.eye(size) + err / 2)

    unitarity_err = np.max(np.abs(U.conj().T @ U - np.eye(size)))
    assert unitarity_err < 1e-6, \
        f"Unitarity error {unitarity_err:.2e} too large for Qiskit"
    return U.astype(np.complex128)


# ══════════════════════════════════════════════════════════
#  3.  Build Qiskit staircase circuit
# ══════════════════════════════════════════════════════════

def build_circuit(cores, D, d, n):
    """
    Left-canonical staircase MPS preparation circuit.

    Register layout:
        bond : log2(D) qubits  — virtual bond register (starts/ends |0>)
        phys : n qubits        — physical output register

    Site 0: gate size = 1*d = d  → acts on phys[0] only (no bond needed)
    Site k: gate size = D*d      → acts on bond[0..log2D-1] + phys[k]
    """
    n_bond = int(np.ceil(np.log2(D))) if D > 1 else 0
    phys = QuantumRegister(n, name="q")

    if n_bond > 0:
        bond = QuantumRegister(n_bond, name="b")
        qc = QuantumCircuit(bond, phys, name=f"MPS  n={n}  D={D}")
    else:
        bond = None
        qc = QuantumCircuit(phys, name=f"MPS  n={n}  D={D} (product)")

    COLOURS = ["#01696f","#457b9d","#e76f51","#9b5de5",
               "#e9c46a","#264653","#a8dadc","#f4a261"]

    for k, A in enumerate(cores):
        U = core_to_unitary(A)
        size = U.shape[0]
        n_gate_q = int(np.round(np.log2(size)))

        gate = UnitaryGate(U, label=f"A[{k+1}]")

        if bond is not None and n_gate_q > 1:
            target = (list(bond) + [phys[k]])[:n_gate_q]
        else:
            target = [phys[k]]

        qc.append(gate, target)
        if k < n - 1:
            qc.barrier()

    return qc, bond, phys, n_bond


# ══════════════════════════════════════════════════════════
#  4.  Verification
# ══════════════════════════════════════════════════════════

def verify(qc, psi_mps, n, n_bond):
    """
    Simulate circuit statevector, select bond=|0> sector,
    compare to classical MPS contraction.
    Returns fidelity ∈ [0,1].
    """
    sv = Statevector(qc).data   # shape (2^(n_bond+n),)
    total_q = n_bond + n

    # In Qiskit, qubit 0 is the LSB of the state index.
    # The bond register is declared first → its qubits are at bit positions 0..n_bond-1.
    # Select bond=|0>: keep only indices where bits 0..n_bond-1 are all 0.
    if n_bond > 0:
        bond_mask = (1 << n_bond) - 1
        indices = [i for i in range(2**total_q) if (i & bond_mask) == 0]
        sv_phys = sv[indices]
    else:
        sv_phys = sv

    sv_phys = sv_phys / (np.linalg.norm(sv_phys) + 1e-15)
    psi_norm = psi_mps / (np.linalg.norm(psi_mps) + 1e-15)

    fid = abs(np.dot(psi_norm.conj(), sv_phys))**2
    return fid


# ══════════════════════════════════════════════════════════
#  5.  Visualisation
# ══════════════════════════════════════════════════════════

GATE_COLOURS = ["#01696f","#457b9d","#e76f51","#9b5de5",
                "#e9c46a","#264653","#a8dadc","#f4a261"]

STYLE = {
    "backgroundcolor": "#f7f6f2",
    "gatefacecolor":   "#f9f9f9",
    "fontsize": 10,
    "subfontsize": 8,
}


def draw_full_circuit(qc, cores, n, D, fidelity, path):
    style = {
        **STYLE,
        "displaycolor": {
            f"A[{k+1}]": (GATE_COLOURS[k % len(GATE_COLOURS)], "#ffffff")
            for k in range(n)
        },
    }
    fig = qc.draw(output="mpl", style=style, fold=60, scale=0.9,
                  plot_barriers=True, initial_state=True)

    n_bond = int(np.ceil(np.log2(D))) if D > 1 else 0
    n_params = sum(A.size * 2 for A in cores)
    fig.suptitle(
        f"MPS Preparation Circuit  (n={n} sites, D={D}, d=2)\n"
        f"Qubits: {qc.num_qubits} ({n_bond} bond + {n} physical) | "
        f"Depth: {qc.depth()} | "
        f"Circuit–MPS fidelity: {fidelity:.6f}",
        fontsize=11, fontweight="bold", color="#264653", y=1.05,
    )
    patches = [
        mpatches.Patch(
            color=GATE_COLOURS[k % len(GATE_COLOURS)],
            label=f"A[{k+1}]  shape {cores[k].shape}"
        )
        for k in range(n)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=n,
               fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.10), edgecolor="#dcd9d5")
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="#f7f6f2")
    plt.close(fig)
    print(f"  Saved: {path}")


def draw_layerwise(cores, D, d, fidelity, path):
    n = len(cores)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 6.0), facecolor="#f7f6f2")
    if n == 1:
        axes = [axes]

    for k, (A, ax) in enumerate(zip(cores, axes)):
        U = core_to_unitary(A)
        size = U.shape[0]
        n_q = int(np.round(np.log2(size)))
        qc_k = QuantumCircuit(n_q, name=f"A[{k+1}]")
        gate = UnitaryGate(U, label=f"A[{k+1}]")
        qc_k.append(gate, list(range(n_q)))

        style_k = {
            **STYLE,
            "fontsize": 12,
            "displaycolor": {
                f"A[{k+1}]": (GATE_COLOURS[k % len(GATE_COLOURS)], "#ffffff"),
            },
        }
        qc_k.draw(output="mpl", style=style_k, ax=ax, scale=0.9,
                  plot_barriers=False, initial_state=True)
        D_l, _, D_r = A.shape
        ax.set_title(
            f"Site {k+1}\nCore A[{k+1}]  ({D_l}×{d}×{D_r})\n"
            f"Gate: {size}×{size}  ({n_q}-qubit)",
            fontsize=10, color="#264653", pad=8,
        )
        ax.patch.set_facecolor("#f7f6f2")

    n_bond = int(np.ceil(np.log2(D))) if D > 1 else 0
    fig.suptitle(
        f"MPS Circuit — Site-by-Site Gate View\n"
        f"n={n}  D={D}  d={d}  |  "
        f"{n_bond} bond qubit(s)  |  Fidelity = {fidelity:.6f}",
        fontsize=13, fontweight="bold", color="#264653", y=1.06,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="#f7f6f2")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    n, D, d = N_SITES, BOND_DIM, D_PHYS

    print(f"\n{'═'*55}")
    print(f"  MPS Circuit Builder")
    print(f"  n={n}  D={D}  d={d}  seed={SEED}")
    n_bond = int(np.ceil(np.log2(D))) if D > 1 else 0
    print(f"  Total circuit qubits: {n_bond + n} ({n_bond} bond + {n} physical)")
    print(f"{'═'*55}")

    # ── 1. Target statevector ──────────────────────────────
    rng = np.random.default_rng(SEED)
    raw = rng.standard_normal(d**n) + 1j * rng.standard_normal(d**n)
    psi_target = raw / np.linalg.norm(raw)

    # ── 2. MPS decomposition ───────────────────────────────
    print(f"\nDecomposing into left-canonical MPS (D={D})...")
    cores = statevector_to_lc_mps(psi_target, n, d, D)
    psi_mps = contract_mps(cores, d)
    trunc_err = 1 - abs(np.dot(psi_target.conj(),
                                psi_mps / np.linalg.norm(psi_mps)))**2
    print(f"  Truncation error: {trunc_err:.4e}  "
          f"({'exact' if trunc_err < 1e-10 else f'D={D} approx'})")

    for k, A in enumerate(cores):
        Dl, _, Dr = A.shape
        # verify left-canonical
        lc_err = max(np.max(np.abs(
            sum(A[:,s,:].conj().T @ A[:,s,:] for s in range(d)) - np.eye(Dr)
        )), 0)
        print(f"  A[{k+1}]  shape {A.shape}  "
              f"left-canonical err: {lc_err:.2e}")

    # ── 3. Build circuit ────────────────────────────────────
    print(f"\nBuilding Qiskit circuit...")
    qc, bond, phys, n_bond = build_circuit(cores, D, d, n)
    print(f"  Depth   : {qc.depth()}")
    print(f"  Gates   : {dict(qc.count_ops())}")
    print(f"  Qubits  : {qc.num_qubits}")

    # ── 4. Verify ───────────────────────────────────────────
    print(f"\nVerifying circuit vs MPS contraction...")
    fid = verify(qc, psi_mps, n, n_bond)
    print(f"  Fidelity |<ψ_MPS|ψ_circuit>|² = {fid:.8f}  "
          f"{'✓' if fid > 0.99 else '✗'}")

    # ── 5. Save cores ───────────────────────────────────────
    np.savez("output/mps_cores.npz",
             **{f"core_{k}": A for k, A in enumerate(cores)},
             psi_target=psi_target, psi_mps=psi_mps,
             n=n, D=D, d=d, seed=SEED, fidelity=fid)
    print(f"\n  Cores saved: output/mps_cores.npz")

    # ── 6. Draw ─────────────────────────────────────────────
    print(f"\nDrawing circuits...")
    draw_full_circuit(qc, cores, n, D, fid, "output/mps_circuit_qiskit.png")
    draw_layerwise(cores, D, d, fid, "output/mps_circuit_layerwise.png")

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  {'Site':<6}  {'Shape':^18}  {'Gate':^8}  {'n_q'}")
    print(f"{'─'*55}")
    for k, A in enumerate(cores):
        U = core_to_unitary(A)
        size = U.shape[0]
        nq = int(np.round(np.log2(size)))
        print(f"  A[{k+1}]    {str(A.shape):^18}  {size}×{size:<4}   {nq}")
    print(f"{'─'*55}")
    print(f"  Bond register: {n_bond} qubit(s), shared across all site gates")
    print(f"  Truncation err: {trunc_err:.2e}")
    print(f"  Circuit fidelity: {fid:.8f}")
    print(f"\n  Done.")
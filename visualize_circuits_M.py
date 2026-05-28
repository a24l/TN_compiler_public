"""
visualize_circuits.py
=====================
Visualize MPS training states as Qiskit circuits with target unitary M.

Each circuit shows two sections separated by a barrier:
  LEFT  — a Qiskit statevector preparation circuit for the sampled |x_j⟩
  RIGHT — M gate applies the 16×16 target unitary → |φ_j⟩ = M|x_j⟩

Workflow:
  1. Load pool from output/pool_mps1_D2.npz
  2. Sample 5 states (same seed=7 as before)
  3. Generate M (same seed=99 as dataset)
  4. Build circuit: statevector prep → barrier → M
  5. Composite figure with per-sample labels and norms

Run:
  python build_pool_dataset_new.py  # step 1: generate pool
  python generate_M_and_dataset.py  # step 2: generate M + dataset
  python visualize_circuits_M.py    # step 3: visualize

Requires: qiskit, matplotlib, pillow, pylatexenc
"""

import numpy as np
import sys, os, io
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from build_pool_dataset_new import load_pool, sample_from_pool
from generate_M_and_dataset import generate_M

os.makedirs("output/architecture", exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  USER PARAMETERS  — keep consistent with build_pool.py / generate_M
# ══════════════════════════════════════════════════════════════════
POOL_PATH   = "output/pool_mps1_D2_N10k.npz"
N_QUBITS    = 4
N_SAMPLES   = 5
SAMPLE_SEED = 7      # same as previous circuit visualization
M_SEED      = 99     # same as generate_M_and_dataset.py
OUT_PATH    = "output/architecture/mps_circuits_with_M.png"
# ══════════════════════════════════════════════════════════════════

STYLE = {
    "backgroundcolor": "#ffffff",
    "linecolor":       "#1a1a2e",
    "textcolor":       "#1a1a2e",
    "gatefacecolor":   "#2a9d8f",
    "gatetextcolor":   "#ffffff",
    "margin":          [0.8, 0.4, 0.4, 0.4],
    "fontsize":        14,
    "subfontsize":     11,
}
BG='#f7f6f2'; TEAL='#2a9d8f'; DARK='#264653'; MUTED='#6c757d'
ORANGE='#e76f51'; GOLD='#e9c46a'


def statevector_to_lc_mps(psi, n_qubits, d=2, max_bond=None):
    """Left-canonical MPS decomposition of a statevector via sequential SVD."""
    cores = []
    remainder = psi.reshape(1, d**n_qubits)
    bond_left = 1

    for _site in range(n_qubits - 1):
        remainder = remainder.reshape(bond_left * d, -1)
        U, S, Vh = np.linalg.svd(remainder, full_matrices=False)
        keep = len(S) if max_bond is None else min(len(S), max_bond)
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]
        cores.append(U.reshape(bond_left, d, keep).astype(np.complex128))
        remainder = np.diag(S) @ Vh
        bond_left = keep

    cores.append(remainder.reshape(bond_left, d, 1).astype(np.complex128))
    return cores


def contract_mps(cores):
    """Contract left-canonical MPS cores back to a statevector."""
    contracted = cores[0][0, :, :]
    for core in cores[1:]:
        contracted = np.einsum("ia,asr->isr", contracted, core)
        contracted = contracted.reshape(contracted.shape[0] * contracted.shape[1], contracted.shape[2])
    return contracted[:, 0]


def core_to_unitary(core):
    """Complete a left-canonical MPS core isometry to a square unitary."""
    bond_left, d, bond_right = core.shape
    size = bond_left * d
    iso = np.column_stack([core[:, :, r].reshape(-1) for r in range(bond_right)])

    q_iso, r_iso = np.linalg.qr(iso)
    phase = np.diag(r_iso) / np.abs(np.diag(r_iso))
    q_iso = q_iso * phase

    if bond_right == size:
        return q_iso.astype(np.complex128)

    rng = np.random.default_rng(size * 1009 + bond_right)
    complement = rng.standard_normal((size, size - bond_right)) + \
        1j * rng.standard_normal((size, size - bond_right))
    complement = complement - q_iso @ (q_iso.conj().T @ complement)
    q_comp, r_comp = np.linalg.qr(complement)
    comp_phase = np.diag(r_comp) / np.abs(np.diag(r_comp))
    q_comp = q_comp * comp_phase

    unitary = np.hstack([q_iso, q_comp])
    err = np.max(np.abs(unitary.conj().T @ unitary - np.eye(size)))
    if err > 1e-8:
        raise ValueError(f"MPS core unitary completion failed: max error {err:.2e}")
    return unitary.astype(np.complex128)


def build_mps_preparation_circuit(state_vector, n_qubits=4, bond_dim=2, d=2):
    """Build a Qiskit circuit that prepares state_vector from MPS cores."""
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import UnitaryGate

    cores = statevector_to_lc_mps(state_vector, n_qubits, d=d, max_bond=bond_dim)
    max_right_bond = max(core.shape[2] for core in cores)
    n_bond = int(np.ceil(np.log2(max(1, max_right_bond)))) if max_right_bond > 1 else 0

    phys = QuantumRegister(n_qubits, name="q")
    if n_bond:
        bond = QuantumRegister(n_bond, name="b")
        qc = QuantumCircuit(bond, phys, name="MPS prep")
    else:
        bond = None
        qc = QuantumCircuit(phys, name="MPS prep")

    for site, core in enumerate(cores):
        gate = UnitaryGate(core_to_unitary(core), label=f"A[{site}]")
        if bond is not None and gate.num_qubits > 1:
            target = list(bond) + [phys[site]]
        else:
            target = [phys[site]]
        qc.append(gate, target)
        if site < n_qubits - 1:
            qc.barrier()

    return qc, cores, n_bond, bond, phys


def prepared_physical_state(qc, n_qubits, n_bond):
    """Simulate an MPS prep circuit and select the bond-|0> physical sector."""
    from qiskit.quantum_info import Statevector

    sv = Statevector(qc).data
    if n_bond == 0:
        return sv

    bond_mask = (1 << n_bond) - 1
    indices = [idx for idx in range(2 ** (n_bond + n_qubits)) if (idx & bond_mask) == 0]
    state = sv[indices]
    norm = np.linalg.norm(state)
    return state / norm if norm > 1e-14 else state


def reorder_statevector_to_qiskit_basis(state_vector, n_qubits):
    """
    Convert a state in |q0 q1 ... q{n-1}> ordering to Qiskit's basis ordering.

    sample_from_pool builds vectors in left-to-right qubit order, while Qiskit
    statevector amplitudes use little-endian indexing.
    """
    tensor = np.asarray(state_vector, dtype=np.complex128).reshape((2,) * n_qubits)
    return np.transpose(tensor, axes=tuple(range(n_qubits - 1, -1, -1))).reshape(-1)


def build_statevector_preparation_circuit(state_vector, n_qubits=4, fidelity_tol=1e-8):
    """Build a prep circuit from a sampled state vector using Qiskit initialize."""
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.quantum_info import Statevector

    target_state = np.asarray(state_vector, dtype=np.complex128)
    norm = np.linalg.norm(target_state)
    if norm <= 1e-14:
        raise ValueError("Statevector preparation requires a non-zero state.")
    target_state = target_state / norm
    qiskit_target = reorder_statevector_to_qiskit_basis(target_state, n_qubits)

    phys = QuantumRegister(n_qubits, name="q")
    qc = QuantumCircuit(phys, name="State prep")
    qc.initialize(qiskit_target, list(phys))

    prepared = Statevector(qc).data
    prep_fidelity = abs(np.vdot(qiskit_target, prepared)) ** 2
    if prep_fidelity < 1.0 - fidelity_tol:
        raise ValueError(
            f"Statevector prep fidelity too low: {prep_fidelity:.12f}"
        )

    return qc, phys, prep_fidelity


def build_circuit(j, indices, state_vectors, M, n_qubits=4, bond_dim=2, d=2,
                  fidelity_tol=1e-8):
    """
    Circuit for M|x_j⟩:

      |0⟩ ... --[Qiskit statevector prep for actual x_j]-- ░ -- M

    - Prep gate is built directly from sampled state_vectors[j]
    - M is the 16×16 Haar-random target unitary
    - Barrier (░) separates the two sections visually
    """
    from qiskit.circuit.library import UnitaryGate

    _ = (indices, bond_dim, d)  # retained for API compatibility
    qc, phys, prep_fidelity = build_statevector_preparation_circuit(
        state_vectors[j], n_qubits=n_qubits, fidelity_tol=fidelity_tol)

    # Separator before M
    qc.barrier()

    # M gate on all qubits → produces M|x_j⟩
    qc.append(UnitaryGate(M, label="M"), list(phys))

    return qc, prep_fidelity


def qc_to_image(qc):
    """Render circuit with |0⟩ labels visible on the left."""
    fig = qc.draw(
        output        = "mpl",
        style         = STYLE,
        fold          = -1,
        plot_barriers = True,
        initial_state = True,      # ← |0⟩ visible on every wire
        reverse_bits  = False,     # q_0 top → q_3 bottom
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor="#ffffff", pad_inches=0.15)
    plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGBA")


def build_composite(indices, state_vectors, Phi, circuit_imgs, prep_fidelities, n_samples):
    fig = plt.figure(figsize=(20, 30), facecolor=BG)

    fig.text(0.5, 0.988,
             "NFL Dataset 1 — MPS Training States with Target Unitary M",
             ha="center", va="top", fontsize=17, fontweight="bold", color=DARK)
    fig.text(0.5, 0.977,
             "Exact Qiskit statevector prep   |   "
             "M ∈ U(16): Haar-random 16×16 target unitary   |   n=4 qubits",
             ha="center", va="top", fontsize=11, color=MUTED)
    fig.text(0.03, 0.966,
             "◀  State prep section: initialize(|x_j⟩) from sampled vector  "
             "│  barrier  │  "
             "M gate: applies target unitary → |φ_j⟩ = M|x_j⟩  ▶",
             va="top", fontsize=10, color=DARK,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f4f3",
                       edgecolor=TEAL, linewidth=0.8))
    fig.add_artist(plt.Line2D([0.03,0.97],[0.958,0.958],
                   color=DARK,lw=0.7,alpha=0.25,transform=fig.transFigure))

    panel_h, panel_gap, top_start = 0.155, 0.014, 0.950

    for j in range(n_samples):
        top    = top_start - j*(panel_h+panel_gap)
        bottom = top - panel_h
        sample_indices = indices[j].tolist()

        fig.text(0.03, top-0.004,
                 f"|x_{j}⟩  →  M|x_{j}⟩",
                 fontsize=13, fontweight="bold", color=TEAL,
                 va="top", fontfamily="monospace")
        fig.text(0.20, top-0.004,
                 f"pool indices={sample_indices}   "
                 f"prep fidelity={prep_fidelities[j]:.8f}",
                 fontsize=10, color=DARK, va="top")
        fig.text(0.65, top-0.004,
                 f"‖|x_{j}⟩‖={np.linalg.norm(state_vectors[j]):.4f}",
                 fontsize=10, color=ORANGE, va="top")
        fig.text(0.80, top-0.004,
                 f"‖M|x_{j}⟩‖={np.linalg.norm(Phi[j]):.4f}",
                 fontsize=10, color=GOLD, va="top")

        ax = fig.add_axes([0.03, bottom+0.008, 0.94, panel_h-0.028],
                          frameon=True)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#dcd9d5"); sp.set_linewidth(0.9)
        ax.set_facecolor("#ffffff")
        ax.imshow(np.array(circuit_imgs[j]), aspect="auto")

        if j < n_samples-1:
            fig.add_artist(plt.Line2D(
                [0.03,0.97],[bottom+0.003,bottom+0.003],
                color=DARK,lw=0.5,alpha=0.18,transform=fig.transFigure))

    fig.text(0.5, 0.017,
             "■ initialize(|x_j⟩) = Qiskit state preparation from sampled vector   "
             "■ M = 16×16 target unitary acting on all 4 qubits   "
             "| barrier separates state preparation from M application",
             ha="center", fontsize=9.5, color=MUTED)
    fig.text(0.5, 0.006,
             "|φ_j⟩ = M|x_j⟩, with |x_j⟩ verified against the displayed statevector preparation circuit",
             ha="center", fontsize=10, color=DARK, style="italic")
    return fig


if __name__ == "__main__":

    # Load pool
    pool_data = load_pool(POOL_PATH)
    pool, meta = pool_data["pool"], pool_data["metadata"]
    print(f"Loaded pool: {pool.shape}  D={meta['D']}  N={meta['N']}")

    # Sample (same seed as before)
    samples = sample_from_pool(
        pool=pool, n_qubits=N_QUBITS, t_states=N_SAMPLES,
        D=meta["D"], d=meta["d"], seed=SAMPLE_SEED, verbose=True,
    )
    indices         = samples["indices"]
    state_vectors   = samples["state_vectors"]

    # Generate M (same seed as dataset)
    M   = generate_M(n_qubits=N_QUBITS, d=meta["d"], seed=M_SEED, verbose=True)
    Phi = (M @ state_vectors.T).T

    # Build and render circuits
    circuit_imgs = []
    prep_fidelities = []
    for j in range(N_SAMPLES):
        qc, prep_fid = build_circuit(
            j, indices, state_vectors, M,
            n_qubits=N_QUBITS, bond_dim=meta["D"], d=meta["d"])
        img = qc_to_image(qc)
        circuit_imgs.append(img)
        prep_fidelities.append(prep_fid)
        print(f"  Rendered |x_{j}>  →  M|x_{j}>   "
              f"k={indices[j].tolist()}  prep_fidelity={prep_fid:.12f}")

    # Composite figure
    fig = build_composite(indices, state_vectors, Phi, circuit_imgs, prep_fidelities, N_SAMPLES)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")

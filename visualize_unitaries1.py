"""
visualize_unitaries.py
======================
Standalone visualizer for target_unitaries.npz.

Reconstructs 5 randomly sampled Ising-chain unitaries as Qiskit
QuantumCircuit objects and saves them as PNG diagrams.

Dependencies: numpy, qiskit, matplotlib, Pillow
Install:      pip install qiskit qiskit-aer pylatexenc pillow matplotlib

Usage:
    python visualize_unitaries.py

    # or change the parameters at the bottom of this file:
    #   SAMPLE_IDS   = [3, 17, 42, 66, 99]  ← fixed indices, or None for random
    #   SEED         = 7                     ← controls random sampling
    #   N_SAMPLES    = 5                     ← how many to visualise
    #   INPUT_FILE   = "target_unitaries"    ← path to .npz (no extension)
    #   OUTPUT_DIR   = "output"
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from qiskit import QuantumCircuit
from PIL import Image
import os


# ──────────────────────────────────────────────────────────────────
#  Gate colour scheme used in Qiskit mpl style
# ──────────────────────────────────────────────────────────────────
#  Dark teal  #264653  →  RZ  (Rz rotation around Z axis)
#  Teal       #2a9d8f  →  RY  (Ry rotation around Y axis)
#  Orange     #e76f51  →  RZZ (Ising ZZ coupling between neighbours)
#  Light grey #dee2e6  →  Barrier (visual layer separator)
# ──────────────────────────────────────────────────────────────────

QISKIT_STYLE = {
    "backgroundcolor": "#ffffff",
    "linecolor":       "#1a1a2e",
    "textcolor":       "#1a1a2e",
    "gatefacecolor":   "#2a9d8f",
    "gatetextcolor":   "#ffffff",
    "barrierfacecolor":"#f1f3f5",
    "margin":          [0.8, 0.3, 0.2, 0.3],
    "fontsize":        12,
    "subfontsize":     9,
    "displaycolor": {
        "rz":  ("#264653", "#ffffff"),   # dark teal
        "ry":  ("#2a9d8f", "#ffffff"),   # teal
        "rzz": ("#e76f51", "#ffffff"),   # orange
    },
}


# ══════════════════════════════════════════════════════════════════
#  CORE: build a Qiskit QuantumCircuit from saved angles
# ══════════════════════════════════════════════════════════════════

def build_qiskit_circuit(idx: int,
                          single_angles: np.ndarray,
                          zz_angles: np.ndarray,
                          n_qubits: int,
                          n_layers: int) -> QuantumCircuit:
    """
    Reconstruct the Qiskit QuantumCircuit for target unitary M_{idx}.

    The circuit implements the Ising chain in three steps per layer:

      Layer k:
        For every qubit q:
          Rz(c) → Ry(b) → Rz(a)        (ZYZ single-site rotation)
        For every bond (q, q+1):
          RZZ(theta)                    (Ising nearest-neighbour coupling)
        Barrier                         (visual layer separator)

    Parameters
    ----------
    idx           : index of the unitary in the dataset (0 to N-1)
    single_angles : array shape (N, L, n, 3)
    zz_angles     : array shape (N, L, n-1)
    n_qubits      : number of qubits
    n_layers      : number of layers L

    Returns
    -------
    qc : QuantumCircuit with name f'M_{idx}'
    """
    qc = QuantumCircuit(n_qubits, name=f"M_{idx}")

    for layer in range(n_layers):

        # ── ZYZ single-site gates ──────────────────────────────────
        # Matrix notation: U = Rz(a) · Ry(b) · Rz(c)
        # Circuit order (left = applied first in time):
        #   Rz(c)  then  Ry(b)  then  Rz(a)
        for q in range(n_qubits):
            a = float(single_angles[idx, layer, q, 0])
            b = float(single_angles[idx, layer, q, 1])
            c = float(single_angles[idx, layer, q, 2])
            qc.rz(c, q)   # first in time
            qc.ry(b, q)
            qc.rz(a, q)   # last in time

        # ── ZZ coupling ─────────────────────────────────────────────
        # rzz(theta) = exp(-i theta/2 · Z⊗Z)
        for bond in range(n_qubits - 1):
            theta = float(zz_angles[idx, layer, bond])
            qc.rzz(theta, bond, bond + 1)

        # ── Barrier between layers ────────────────────────────────
        if layer < n_layers - 1:
            qc.barrier()

    return qc


# ══════════════════════════════════════════════════════════════════
#  VISUALIZER: draw one circuit and return a matplotlib Figure
# ══════════════════════════════════════════════════════════════════

def draw_circuit(qc: QuantumCircuit, idx: int,
                 n_qubits: int, n_layers: int):
    """
    Draw a single QuantumCircuit as a matplotlib Figure using Qiskit mpl.

    Parameters
    ----------
    qc       : QuantumCircuit to draw
    idx      : unitary index (used in title)
    n_qubits : for title annotation
    n_layers : for title annotation

    Returns
    -------
    fig : matplotlib Figure  (caller is responsible for closing it)
    """
    counts = {}
    for inst in qc.data:
        name = inst.operation.name
        if name != "barrier":
            counts[name] = counts.get(name, 0) + 1
    gate_str = "  |  ".join(f"{v}× {k.upper()}" for k, v in counts.items())

    fig = qc.draw(
        output        = "mpl",
        style         = QISKIT_STYLE,
        fold          = -1,       # never wrap the circuit
        plot_barriers = True,
        initial_state = True,     # shows |0⟩ on each qubit wire
    )

    fig.suptitle(
        f"Target Unitary  M_{{ {idx} }}   "
        f"n={n_qubits} qubits  ·  L={n_layers} layers  ·  {gate_str}",
        fontsize=10.5, fontweight="bold", color="#1a1a2e", y=1.04,
    )
    fig.patch.set_facecolor("#ffffff")
    return fig


# ══════════════════════════════════════════════════════════════════
#  MAIN: load → sample → draw → composite
# ══════════════════════════════════════════════════════════════════

def visualize_unitaries(
        input_file:  str        = "target_unitaries",
        output_dir:  str        = "output",
        n_samples:   int        = 5,
        sample_ids:  list | None = None,
        seed:        int        = 7,
) -> None:
    """
    Load unitaries, draw n_samples circuits, save PNGs.

    Parameters
    ----------
    input_file  : path to .npz without extension
    output_dir  : directory for PNG output
    n_samples   : how many circuits to draw  (ignored if sample_ids given)
    sample_ids  : explicit list of indices   (overrides n_samples / seed)
    seed        : RNG seed for random sampling
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────
    path = input_file if input_file.endswith(".npz") else input_file + ".npz"
    arch          = np.load(path)
    single_angles = arch["single_angles"]
    zz_angles     = arch["zz_angles"]
    n_qubits      = int(arch["n_qubits"])
    n_layers      = int(arch["n_layers"])
    n_total       = int(arch["n_unitaries"])
    print(f"Loaded {n_total} unitaries  ({n_qubits} qubits, {n_layers} layers)")

    # ── Choose which indices to visualise ─────────────────────────
    if sample_ids is None:
        rng        = np.random.default_rng(seed)
        sample_ids = sorted(rng.choice(n_total, size=n_samples,
                                        replace=False).tolist())
    print(f"Visualising indices: {sample_ids}")

    # ── Draw individual circuits ───────────────────────────────────
    individual_paths = []
    for idx in sample_ids:
        qc  = build_qiskit_circuit(idx, single_angles, zz_angles,
                                    n_qubits, n_layers)
        fig = draw_circuit(qc, idx, n_qubits, n_layers)
        out = os.path.join(output_dir, f"circuit_U{idx:03d}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        individual_paths.append(out)
        print(f"  Saved: {out}")

    # ── Composite: stack all circuits vertically ───────────────────
    images = [Image.open(p).convert("RGBA") for p in individual_paths]
    max_w  = max(im.width  for im in images)
    gap    = 24
    header = 110

    total_h = header + sum(im.height for im in images) + gap * (len(images) - 1)
    canvas  = Image.new("RGBA", (max_w, total_h), (255, 255, 255, 255))

    # Header with legend
    fig_h, ax_h = plt.subplots(figsize=(max_w / 140, header / 140))
    fig_h.patch.set_facecolor("#ffffff")
    ax_h.axis("off")
    ax_h.text(0.5, 0.82,
        f"5 Sampled Ising Chain Target Unitaries  "
        f"[n={n_qubits} qubits, L={n_layers} layers]  "
        f"— indices {sample_ids}",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color="#1a1a2e", transform=ax_h.transAxes)
    legend_handles = [
        mpatches.Patch(color="#264653", label="RZ — rotation around Z axis"),
        mpatches.Patch(color="#2a9d8f", label="RY — rotation around Y axis"),
        mpatches.Patch(color="#e76f51", label="RZZ — Ising ZZ coupling"),
        mpatches.Patch(color="#dee2e6", label="Barrier — layer separator"),
    ]
    ax_h.legend(handles=legend_handles, loc="lower center", ncol=4,
                fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.1))
    fig_h.tight_layout(pad=0.3)
    tmp_hdr = os.path.join(output_dir, "_hdr_tmp.png")
    fig_h.savefig(tmp_hdr, dpi=140, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig_h)

    hdr = Image.open(tmp_hdr).convert("RGBA")
    hdr = hdr.resize((max_w, header), Image.LANCZOS)
    canvas.paste(hdr, (0, 0))
    os.remove(tmp_hdr)

    y = header
    for im in images:
        canvas.paste(im, ((max_w - im.width) // 2, y))
        y += im.height + gap

    composite = os.path.join(output_dir, "all_5_circuits.png")
    canvas.save(composite)
    print(f"\n✓ Composite diagram: {composite}")
    print(f"  Image size: {canvas.width} × {canvas.height} px")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT  —  edit these parameters to customise
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    visualize_unitaries(
        input_file  = "target_unitaries",   # .npz file from Step 1
        output_dir  = "output",
        n_samples   = 5,                    # how many circuits to visualise
        sample_ids  = None,                 # set to e.g. [3,17,42] for fixed
        seed        = 7,                    # RNG seed for random sampling
    )

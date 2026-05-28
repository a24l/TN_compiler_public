"""
visualize_mpsbrick_ZZising_qiskit.py
=====================================
Visualize the MPS Brickwork P_S circuit (ZYZ + ZZ Ising entanglers)
using Qiskit QuantumCircuit and draw(output='mpl').

Architecture per layer l:
  Step (a) - ZYZ single-qubit gates on every qubit k
             Rz(alpha) . Ry(beta) . Rz(gamma)
  Step (b) - ZZ Ising entanglers on bonds (k, k+1)
             ZZ(theta) = exp(-i*theta/2 * Z x Z)
             Decomposed as: CX . Rz(theta) . CX

Parameters match train_ps_brickwork.py (ZZ version):
  n_qubits = 4
  n_layers = 3   (representative; full training uses 6)
  bond_dim = 2   (D independent ZZ modes per bond per layer)

"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

os.makedirs("output/architecture/mps", exist_ok=True)

# =====================================================================
#  HYPERPARAMETERS  <- edit here to match your training run
# =====================================================================
N_QUBITS = 4      # n  - number of qubits
N_LAYERS = 3      # L  - number of brickwork layers to visualise
BOND_DIM = 2      # D  - number of independent ZZ modes per bond

OUT_FULL  = "output/architecture/mps/ps_brickwork_ZZising_qiskit.png"
OUT_LAYER = "output/architecture/mps/ps_brickwork_ZZising_qiskit_layerwise.png"
# =====================================================================


# ---------------------------------------------------------------------
#  GATE HELPERS
# ---------------------------------------------------------------------

def add_zyz(qc, k, l):
    """
    Append ZYZ decomposition on qubit k in layer l.
    Rz(alpha) . Ry(beta) . Rz(gamma)
    Three independent trainable parameters per qubit per layer.
    """
    qc.rz(Parameter(f"a_q{k}_l{l}"), k)   # alpha
    qc.ry(Parameter(f"b_q{k}_l{l}"), k)   # beta
    qc.rz(Parameter(f"g_q{k}_l{l}"), k)   # gamma


def add_zz_decomposed(qc, k, l, d):
    """
    Append one ZZ Ising entangler on bond (k, k+1), mode d of layer l.
    ZZ(theta) = exp(-i*theta/2 * Z x Z)
    Exact decomposition: CX(k -> k+1) . Rz(theta, k+1) . CX(k -> k+1)
    One trainable parameter per (bond, layer, mode).
    """
    qc.cx(k, k + 1)
    qc.rz(Parameter(f"t_b{k}_l{l}_d{d}"), k + 1)
    qc.cx(k, k + 1)

def add_zz(qc, k, l, d):
    """
    Append one ZZ Ising entangler on bond (k, k+1), mode d of layer l.
    ZZ(theta) = exp(-i*theta/2 * Z x Z)
    Represented as a single symbolic gate for visualisation purposes only.
    """
    qc.rzz(Parameter(f"t_b{k}_l{l}_d{d}"), k, k + 1)


# ---------------------------------------------------------------------
#  BUILD FULL P_S CIRCUIT
# ---------------------------------------------------------------------

def build_ps_circuit(n, L, D):
    """
    Build the full MPS brickwork P_S circuit.

    Per layer l:
      barrier  (visual separator)
      ZYZ local gates on all n qubits           [Step a]
      ZZ Ising entanglers on all (n-1) bonds,
        repeated D times per bond               [Step b]
    Final barrier marks circuit output.

    Returns
    -------
    qc : QuantumCircuit  with symbolic Parameter objects
    """
    qc = QuantumCircuit(n, name="P_S MPS Brickwork (ZZ Ising)")
    for l in range(L):
        qc.barrier()                           # layer boundary
        # Step (a): ZYZ local gates
        for k in range(n):
            add_zyz(qc, k, l)
        # Step (b): ZZ Ising entanglers
        for k in range(n - 1):                # bonds: (0,1),(1,2),(2,3)
            for d in range(D):                 # D independent modes
                add_zz(qc, k, l, d)
    qc.barrier()                               # output marker
    return qc


# ---------------------------------------------------------------------
#  DRAW 1: FULL CIRCUIT  (all layers, folded)
# ---------------------------------------------------------------------

def draw_full_circuit(qc, path, n, L, D):
    """
    Draw the complete P_S circuit using qiskit's mpl backend.
    Gates are colour-coded:
      Teal  (#2a9d8f) -> Rz  (ZYZ alpha and gamma steps)
      Blue  (#457b9d) -> Ry  (ZYZ beta step)
      Dark  (#264653) -> CX  (part of ZZ Ising decomposition)
    Fold every 52 columns to keep the image readable.
    """
    style = {
        "displaycolor": {
            "rz": ("#2a9d8f", "#ffffff"),
            "ry": ("#457b9d", "#ffffff"),
            "cx": ("#264653", "#ffffff"),
            "rzz": ("#9b5de5", "#ffffff")
        },
        "fontsize": 9,
        "subfontsize": 7,
        "gatefacecolor":     "#f7f6f2",
        "backgroundcolor":   "#f7f6f2",
    }

    fig = qc.draw(
        output="mpl",
        style=style,
        fold=52,                    # wrap after 52 gate columns
        scale=0.72,
        plot_barriers=True,
        initial_state=True,
        reverse_bits=False,
    )

    n_zyz   = L * n * 3
    n_zz    = L * (n - 1) * D
    n_total = n_zyz + n_zz

    fig.suptitle(
        f"MPS Brickwork P_S  -  ZYZ local gates + ZZ Ising entanglers\n"
        f"n={n} qubits  |  L={L} layers  |  D={D} ZZ modes per bond  "
        f"|  {n_total} trainable params",
        fontsize=11, fontweight="bold", color="#264653", y=1.03,
    )

    patches = [
        mpatches.Patch(color="#2a9d8f", label="Rz  (ZYZ steps alpha & gamma)"),
        mpatches.Patch(color="#457b9d", label="Ry  (ZYZ step beta)"),
        mpatches.Patch(color="#264653",
                       label="CX + Rz(theta) + CX  =  ZZ(theta) Ising mode"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.04),
               edgecolor="#dcd9d5")

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="#f7f6f2")
    plt.close(fig)
    print(f"  Full circuit saved : {path}")


# ---------------------------------------------------------------------
#  DRAW 2: LAYER-BY-LAYER  (one subplot per layer)
# ---------------------------------------------------------------------

def draw_layerwise(n, L, D, path):
    """
    Draw each brickwork layer in a separate Matplotlib subplot.
    Each subplot shows:
      - ZYZ gates on all n qubits
      - D ZZ Ising entanglers on each of the (n-1) bonds
    This makes it easy to inspect the structure of a single repetition
    unit of the brickwork before mentally stacking L of them.
    """
    fig, axes = plt.subplots(
        1, L,
        figsize=(6 * L, 5.5),
        facecolor="#f7f6f2",
    )
    if L == 1:
        axes = [axes]

    style = {
        "displaycolor": {
            "rz": ("#2a9d8f", "#ffffff"),
            "ry": ("#457b9d", "#ffffff"),
            "cx": ("#264653", "#ffffff"),
            "rzz": ("#9b5de5", "#ffffff")
        },
        "fontsize": 11,
        "subfontsize": 9,
        "gatefacecolor":   "#f9f9f9",
        "backgroundcolor": "#f7f6f2",
    }

    for l, ax in enumerate(axes):
        qc_l = QuantumCircuit(n, name=f"Layer {l + 1}")

        # Step (a): ZYZ on every qubit
        for k in range(n):
            add_zyz(qc_l, k, l)

        # Step (b): ZZ Ising entanglers on every bond
        for k in range(n - 1):
            for d in range(D):
                add_zz(qc_l, k, l, d)

        qc_l.draw(
            output="mpl",
            style=style,
            ax=ax,
            scale=0.82,
            plot_barriers=False,
            initial_state=True,
        )

        ax.set_title(
            f"Layer {l + 1}\n"
            f"ZYZ ({n} sites)  +  ZZ: {n - 1} bonds x D={D} modes",
            fontsize=10, color="#264653", pad=8,
        )
        ax.patch.set_facecolor("#f7f6f2")

    fig.suptitle(
        f"MPS Brickwork P_S - Layer-by-Layer View\n"
        f"n={n} qubits  |  L={L} layers  |  D={D} ZZ Ising modes per bond",
        fontsize=12, fontweight="bold", color="#264653", y=1.05,
    )

    patches = [
        mpatches.Patch(color="#2a9d8f", label="Rz  (ZYZ steps alpha & gamma)"),
        mpatches.Patch(color="#457b9d", label="Ry  (ZYZ step beta)"),
        mpatches.Patch(color="#264653",
                       label="CX + Rz(theta) + CX  =  ZZ Ising mode"),
        mpatches.Patch(color="#9b5de5",
                       label="Rzz(2cz)  Weyl entangler ZZ"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.06),
               edgecolor="#dcd9d5")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="#f7f6f2")
    plt.close(fig)
    print(f"  Layerwise view saved : {path}")


# ---------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    n, L, D = N_QUBITS, N_LAYERS, BOND_DIM

    n_zyz   = L * n * 3
    n_zz    = L * (n - 1) * D
    n_total = n_zyz + n_zz

    print(f"\nP_S MPS Brickwork Circuit  (ZZ Ising version)")
    print(f"  n_qubits = {n}")
    print(f"  n_layers = {L}")
    print(f"  bond_dim = {D}")
    print(f"")
    print(f"  ZYZ params : {L} layers x {n} qubits x 3 = {n_zyz}")
    print(f"  ZZ  params : {L} layers x {n-1} bonds x D={D} = {n_zz}")
    print(f"  Total      : {n_total} trainable parameters")

    # build symbolic circuit
    qc = build_ps_circuit(n, L, D)
    print(f"")
    print(f"  Circuit depth : {qc.depth()}")
    print(f"  Gate counts   : {dict(qc.count_ops())}")
    print(f"  Num params    : {qc.num_parameters}")

    # render both views
    print(f"")
    draw_full_circuit(qc, OUT_FULL, n, L, D)
    draw_layerwise(n, L, D, OUT_LAYER)
    print(f"\nDone.")

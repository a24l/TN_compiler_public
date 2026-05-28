"""
visualize_mpsbrick_SU4_qiskit.py
==================================
Visualize the MPS Brickwork P_S circuit with FULL SU(4) bond gates
using Qiskit QuantumCircuit and draw(output='mpl').

Architecture per layer l:
  Step (a) - ZYZ single-qubit gates on every qubit k
             Rz(alpha) . Ry(beta) . Rz(gamma)
  Step (b) - Full SU(4) bond gates on bonds (k, k+1)
             G = expm(i*H),  H = (A - A†)/2i,  A in C^{4x4}
             Decomposed via KAK: Rz/Ry/Rz local rotations + 3 Rxx/Ryy/Rzz
             entanglers (canonical form of any SU(4) gate).

WHY SU(4) INSTEAD OF ZZ:
  ZZ = exp(-i*theta/2 * ZxZ) has 1 real DOF.
  Full SU(4) has 15 real DOF per bond - enough to reach any U(16)
  target unitary M. With only ZZ, F̄ plateaus at ~1/dim = 0.06.

SU(4) KAK decomposition (circuit-level):
  Any SU(4) gate U on qubits (k, k+1) can be written as:
    U = (A1 x A2) . exp(i*(cx*XX + cy*YY + cz*ZZ)) . (A3 x A4)
  where A1..A4 are arbitrary SU(2) gates (ZYZ here) and
  cx, cy, cz are the 3 Weyl / canonical parameters.
  Implemented in Qiskit as: local ZYZ + Rxx(2cx) + Ryy(2cy) + Rzz(2cz)
  giving 4*3 + 3 = 15 real DOF per SU(4) gate.

Parameters match train_ps_brickwork.py (SU4 version):
  n_qubits = 4
  n_layers = 3   (representative; full training uses 6)
  bond_dim = 2   (D independent SU(4) gates stacked per bond)


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
N_LAYERS = 1     # L  - number of brickwork layers to visualise
BOND_DIM = 3      # D  - number of independent SU(4) gates per bond

OUT_FULL  = "output/architecture/mps/ps_brickwork_SU4_qiskit_l1_D3.png"
OUT_LAYER = "output/architecture/mps/ps_brickwork_SU4_qiskit_layerwise_l1_D3.png"
# =====================================================================


# ---------------------------------------------------------------------
#  GATE HELPERS
# ---------------------------------------------------------------------

def add_zyz(qc, k, l, tag=""):
    """
    ZYZ single-qubit gate on qubit k in layer l.
    Rz(alpha) . Ry(beta) . Rz(gamma)  — 3 trainable params.
    tag: extra label suffix to keep Parameter names unique within SU4.
    """
    qc.rz(Parameter(f"a{tag}_q{k}_l{l}"), k)
    qc.ry(Parameter(f"b{tag}_q{k}_l{l}"), k)
    qc.rz(Parameter(f"g{tag}_q{k}_l{l}"), k)


def add_su4(qc, k, l, d):
    """
    Full SU(4) gate on bond (k, k+1), mode d of layer l.

    KAK decomposition (universal for any 2-qubit unitary):
      1. Local ZYZ on qubit k        (left A1 block)
      2. Local ZYZ on qubit k+1      (left A2 block)
      3. Rxx(2*cx)                   (Weyl entangler 1)
      4. Ryy(2*cy)                   (Weyl entangler 2)
      5. Rzz(2*cz)                   (Weyl entangler 3)
      6. Local ZYZ on qubit k        (right A3 block)
      7. Local ZYZ on qubit k+1      (right A4 block)

    Total per SU(4): 4*3 + 3 = 15 real parameters.
    With D modes: 15*D params per bond per layer.

    Rxx(t) = exp(-i*t/2 * X x X)
    Ryy(t) = exp(-i*t/2 * Y x Y)
    Rzz(t) = exp(-i*t/2 * Z x Z)
    All three are natively supported in Qiskit.
    """
    # ── left local unitaries ────────────────────────────────────
    add_zyz(qc, k,     l, tag=f"_L0_b{k}_d{d}")
    add_zyz(qc, k + 1, l, tag=f"_L1_b{k}_d{d}")

    # ── Weyl / canonical entanglers ─────────────────────────────
    cx = Parameter(f"cx_b{k}_l{l}_d{d}")
    cy = Parameter(f"cy_b{k}_l{l}_d{d}")
    cz = Parameter(f"cz_b{k}_l{l}_d{d}")
    qc.rxx(cx, k, k + 1)
    qc.ryy(cy, k, k + 1)
    qc.rzz(cz, k, k + 1)

    # ── right local unitaries ───────────────────────────────────
    add_zyz(qc, k,     l, tag=f"_R0_b{k}_d{d}")
    add_zyz(qc, k + 1, l, tag=f"_R1_b{k}_d{d}")


# ---------------------------------------------------------------------
#  BUILD FULL P_S CIRCUIT
# ---------------------------------------------------------------------

def build_ps_su4_circuit(n, L, D):
    """
    Build the full MPS brickwork P_S circuit with SU(4) bond gates.

    Per layer l:
      barrier  (visual layer separator)
      ZYZ local gates on all n qubits                  [Step a]
      Full SU(4) gates on all (n-1) bonds,
        D independent gates stacked per bond            [Step b]
    Final barrier marks circuit output P_S|x>.

    Returns
    -------
    qc : QuantumCircuit  with symbolic Parameter objects
    """
    qc = QuantumCircuit(n, name="P_S MPS Brickwork (SU4)")
    for l in range(L):
        qc.barrier(label=f'layer_{l}')           # layer boundary
        # Step (a): Full SU(4) bond gates
        for k in range(n - 1):                    # bonds: (0,1),(1,2),(2,3)
            for d in range(D):                     # D independent SU(4) per bond
                add_su4(qc, k, l, d)
    qc.barrier(label="output_marker")                                   # output marker
    return qc


# ---------------------------------------------------------------------
#  DRAW 1: FULL CIRCUIT  (all layers, folded)
# ---------------------------------------------------------------------

def draw_full_circuit(qc, path, n, L, D):
    """
    Draw the complete SU(4) brickwork P_S circuit.
    Colour scheme:
      Teal  (#2a9d8f) -> Rz  (ZYZ alpha, gamma + KAK local rotations)
      Blue  (#457b9d) -> Ry  (ZYZ beta)
      Coral (#e76f51) -> Rxx (Weyl entangler XX)
      Gold  (#e9c46a) -> Ryy (Weyl entangler YY)
      Purple(#9b5de5) -> Rzz (Weyl entangler ZZ)
    """
    style = {
        "displaycolor": {
            "rz":  ("#2a9d8f", "#ffffff"),
            "ry":  ("#457b9d", "#ffffff"),
            "rxx": ("#e76f51", "#ffffff"),
            "ryy": ("#e9c46a", "#264653"),
            "rzz": ("#9b5de5", "#ffffff"),
        },
        "fontsize": 9,
        "subfontsize": 7,
        "gatefacecolor":   "#f7f6f2",
        "backgroundcolor": "#f7f6f2",
    }

    fig = qc.draw(
        output="mpl",
        style=style,
        fold=48,
        scale=0.65,
        plot_barriers=True,
        initial_state=True,
        reverse_bits=False,
    )

    n_loc   = L * n * 3
    n_su4   = L * (n - 1) * D * 15
    n_total = n_loc + n_su4

    fig.suptitle(
        f"MPS Brickwork P_S  -  ZYZ local gates + Full SU(4) bond gates\n"
        f"n={n} qubits  |  L={L} layers  |  D={D} SU(4) modes per bond  "
        f"|  {n_total} trainable params",
        fontsize=11, fontweight="bold", color="#264653", y=1.03,
    )

    patches = [
        mpatches.Patch(color="#2a9d8f",
                       label="Rz  (ZYZ alpha/gamma + KAK local)"),
        mpatches.Patch(color="#457b9d",
                       label="Ry  (ZYZ beta)"),
        mpatches.Patch(color="#e76f51",
                       label="Rxx(2cx)  Weyl entangler XX"),
        mpatches.Patch(color="#e9c46a",
                       label="Ryy(2cy)  Weyl entangler YY"),
        mpatches.Patch(color="#9b5de5",
                       label="Rzz(2cz)  Weyl entangler ZZ"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=8, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.06),
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
    Each subplot shows one full repetition unit:
      ZYZ on all n qubits + D SU(4) gates on each (n-1) bond.
    """
    fig, axes = plt.subplots(
        1, L,
        figsize=(7 * L, 6),
        facecolor="#f7f6f2",
    )
    if L == 1:
        axes = [axes]

    style = {
        "displaycolor": {
            "rz":  ("#2a9d8f", "#ffffff"),
            "ry":  ("#457b9d", "#ffffff"),
            "rxx": ("#e76f51", "#ffffff"),
            "ryy": ("#e9c46a", "#264653"),
            "rzz": ("#9b5de5", "#ffffff"),
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
            add_zyz(qc_l, k, l, tag=f"_loc_l{l}")

        # Step (b): SU(4) on every bond
        for k in range(n - 1):
            for d in range(D):
                add_su4(qc_l, k, l, d)

        qc_l.draw(
            output="mpl",
            style=style,
            ax=ax,
            scale=0.75,
            plot_barriers=False,
            initial_state=True,
        )

        ax.set_title(
            f"Layer {l + 1}\n"
            f"ZYZ ({n} sites)  +  SU(4): {n - 1} bonds x D={D} modes",
            fontsize=10, color="#264653", pad=8,
        )
        ax.patch.set_facecolor("#f7f6f2")

    fig.suptitle(
        f"MPS Brickwork P_S - Layer-by-Layer View  (SU(4) bonds)\n"
        f"n={n} qubits  |  L={L} layers  |  D={D} SU(4) modes per bond",
        fontsize=12, fontweight="bold", color="#264653", y=1.06,
    )

    patches = [
        mpatches.Patch(color="#2a9d8f",
                       label="Rz  (ZYZ alpha/gamma + KAK local)"),
        mpatches.Patch(color="#457b9d",
                       label="Ry  (ZYZ beta)"),
        mpatches.Patch(color="#e76f51",
                       label="Rxx(2cx)  Weyl entangler XX"),
        mpatches.Patch(color="#e9c46a",
                       label="Ryy(2cy)  Weyl entangler YY"),
        mpatches.Patch(color="#9b5de5",
                       label="Rzz(2cz)  Weyl entangler ZZ"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.08),
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

    n_loc   = L * n * 3
    n_su4   = L * (n - 1) * D * 15
    n_total = n_loc + n_su4

    print(f"\nP_S MPS Brickwork Circuit  (Full SU(4) bond gates)")
    print(f"  n_qubits = {n}")
    print(f"  n_layers = {L}")
    print(f"  bond_dim = {D}")
    print(f"")
    print(f"  ZYZ local params : {L} x {n} x 3          = {n_loc}")
    print(f"  SU(4) bond params: {L} x {n-1} x D={D} x 15 = {n_su4}")
    print(f"  Total            : {n_total} trainable params")
    print(f"  (U({2**n}) needs {(2**n)**2} DOF  ->  circuit is over-complete ✓)")

    qc = build_ps_su4_circuit(n, L, D)
    print(f"")
    print(f"  Circuit depth : {qc.depth()}")
    print(f"  Gate counts   : {dict(qc.count_ops())}")
    print(f"  Num params    : {qc.num_parameters}")

    print(f"")
    draw_full_circuit(qc, OUT_FULL, n, L, D)
    draw_layerwise(n, L, D, OUT_LAYER)
    print(f"\nDone.")

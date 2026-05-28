"""
quantum_unitary.py
==================
Most general n-qubit parameterized unitary circuit using NumPy.

A general SU(2^n) unitary has (2^n)^2 - 1 = 4^n - 1 real parameters.
This is constructed via:
  U = exp(i * H)
where H is a 2^n x 2^n Hermitian matrix built from the full Pauli tensor-product
basis { I, X, Y, Z }^⊗n  (excluding the identity to stay in SU(2^n)).

Usage
-----
  python quantum_unitary.py                  # demo: 2 qubits, random params
  python quantum_unitary.py --nqubits 3      # 3-qubit random unitary
  python quantum_unitary.py --nqubits 2 --params 0.1 0.2 ...  (4^n-1 values)
"""

import argparse
import itertools
import numpy as np
from numpy.linalg import matrix_power
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate 

# ---------------------------------------------------------------------------
# Pauli basis
# ---------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [_I, _X, _Y, _Z]
PAULI_LABELS = ["I", "X", "Y", "Z"]


def pauli_basis(n_qubits: int) -> list[tuple[str, np.ndarray]]:
    """
    Return all 4^n tensor-product Pauli operators for n qubits,
    excluding the all-identity (trivial global phase).

    Returns
    -------
    list of (label, matrix) — length 4^n - 1
    """
    basis = []
    for combo in itertools.product(range(4), repeat=n_qubits):
        label = "".join(PAULI_LABELS[i] for i in combo)
        mat = PAULIS[combo[0]]
        for i in combo[1:]:
            mat = np.kron(mat, PAULIS[i])
        basis.append((label, mat))
    # remove the all-I term (index 0) to fix global phase → SU(2^n)
    return basis[1:]


# ---------------------------------------------------------------------------
# Hermitian generator
# ---------------------------------------------------------------------------
def build_hamiltonian(params: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    H = Σ_k  θ_k * σ_k
    where σ_k are the (4^n - 1) traceless Pauli basis operators.
    H is Hermitian, and traceless ⟹ det(exp(iH)) = 1  (SU gate).

    Parameters
    ----------
    params    : real array of length 4^n - 1
    n_qubits  : number of qubits

    Returns
    -------
    H : (2^n, 2^n) complex Hermitian matrix
    """
    dim = 2 ** n_qubits
    n_params = 4 ** n_qubits - 1
    assert len(params) == n_params, (
        f"Expected {n_params} parameters for {n_qubits} qubits, got {len(params)}"
    )
    basis = pauli_basis(n_qubits)
    H = np.zeros((dim, dim), dtype=complex)
    for theta, (_, sigma) in zip(params, basis):
        H += theta * sigma
    return H


# ---------------------------------------------------------------------------
# Unitary via matrix exponential
# ---------------------------------------------------------------------------
def matrix_exp_hermitian(H: np.ndarray) -> np.ndarray:
    """
    Compute U = exp(i * H) for a Hermitian matrix H using
    eigendecomposition:  H = V D V†  →  exp(iH) = V exp(iD) V†

    This is numerically stable and exact for Hermitian H.
    """
    eigenvalues, V = np.linalg.eigh(H)          # H = V diag(λ) V†
    exp_D = np.exp(1j * eigenvalues)             # exp(i λ_k)
    return (V * exp_D) @ V.conj().T              # V exp(iD) V†


def general_unitary(n_qubits: int, params: np.ndarray | None = None,
                    seed: int | None = None) -> np.ndarray:
    """
    Build the most general n-qubit SU(2^n) unitary.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.  Gate dimension = 2^n × 2^n.
    params   : array-like of length (4^n - 1), optional
        Real rotation parameters (radians).  If None, drawn uniformly
        from [−π, π] using `seed`.
    seed     : int, optional
        RNG seed (used only when params is None).

    Returns
    -------
    U : (2^n, 2^n) complex unitary matrix in SU(2^n)
    """
    n_params = 4 ** n_qubits - 1
    if params is None:
        rng = np.random.default_rng(seed)
        params = rng.uniform(-np.pi, np.pi, size=n_params)
    else:
        params = np.asarray(params, dtype=float)

    H = build_hamiltonian(params, n_qubits)
    U = matrix_exp_hermitian(H)
    return U


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Check U† U ≈ I."""
    n = U.shape[0]
    residual = U.conj().T @ U - np.eye(n, dtype=complex)
    return np.linalg.norm(residual) < tol


def is_special_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Check det(U) ≈ 1."""
    return abs(np.linalg.det(U) - 1.0) < tol


def unitarity_error(U: np.ndarray) -> float:
    """||U†U - I||_F  (should be ~machine epsilon)."""
    n = U.shape[0]
    return float(np.linalg.norm(U.conj().T @ U - np.eye(n)))


# ---------------------------------------------------------------------------
# Gradient of U w.r.t. each parameter (parameter-shift rule)
# ---------------------------------------------------------------------------
def gradient_unitary(n_qubits: int, params: np.ndarray,
                     shift: float = np.pi / 2) -> list[np.ndarray]:
    """
    Compute ∂U/∂θ_k using the finite-difference parameter-shift rule:
        dU/dθ_k ≈ [U(θ_k + s) − U(θ_k − s)] / (2 sin(s))
    with s = π/2  →  exact for single-Pauli generators.

    Returns
    -------
    grads : list of (2^n, 2^n) complex matrices, one per parameter
    """
    params = np.asarray(params, dtype=float)
    grads = []
    for k in range(len(params)):
        p_plus  = params.copy(); p_plus[k]  += shift
        p_minus = params.copy(); p_minus[k] -= shift
        dU = (general_unitary(n_qubits, p_plus) -
              general_unitary(n_qubits, p_minus)) / (2 * np.sin(shift))
        grads.append(dU)
    return grads



# ─────────────────────────────────────────────────────────────────────────────
# 1.  Generate the random unitary matrix
# ─────────────────────────────────────────────────────────────────────────────
def make_random_unitary(n_qubits: int = 4, seed: int = 42) -> np.ndarray:
    """
    Build and validate a random SU(2^n) unitary from the Pauli exponential map.

    Parameters
    ----------
    n_qubits : int   — number of qubits
    seed     : int   — RNG seed for reproducibility

    Returns
    -------
    U : (2^n, 2^n) complex ndarray  ∈ SU(2^n)
    """
    U = general_unitary(n_qubits=n_qubits, seed=seed)

    err = unitarity_error(U)
    det = np.linalg.det(U)
    print(f"  Unitary error  ||U†U - I||  = {err:.2e}")
    print(f"  Determinant    det(U)        = {det.real:+.8f}{det.imag:+.2e}j")
    assert is_unitary(U),         "Matrix is NOT unitary!"
    assert is_special_unitary(U), "det(U) ≠ 1 — not special unitary!"
    return U


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Wrap the matrix as a Qiskit UnitaryGate
# ─────────────────────────────────────────────────────────────────────────────
def make_unitary_gate(U: np.ndarray, label: str = "U_rand") -> UnitaryGate:
    """
    Convert a 2^n × 2^n NumPy unitary into a named Qiskit UnitaryGate.

    Qiskit uses *little-endian* (LSB = qubit 0) qubit ordering while
    quantum_unitary.py uses big-endian (MSB = qubit 0). We permute the
    tensor axes so the gate matrix conventions match.

    Parameters
    ----------
    U     : (2^n, 2^n) complex ndarray — big-endian unitary
    label : gate label shown in circuit diagrams

    Returns
    -------
    UnitaryGate
    """
    n = int(np.log2(U.shape[0]))
    perm = list(range(n - 1, -1, -1))
    U_le = U.reshape([2] * (2 * n))
    U_le = np.transpose(U_le, perm + [p + n for p in perm])
    U_le = U_le.reshape(2**n, 2**n)
    return UnitaryGate(U_le, label=label)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build the quantum circuit
# ─────────────────────────────────────────────────────────────────────────────
def build_circuit(n_qubits: int = 4,
                  seed: int = 42,
                  add_measurement: bool = True) -> tuple[QuantumCircuit, np.ndarray]:
    """
    Construct the parameterised unitary circuit.

    Circuit layout
    --------------
      q_0 ──[H]──┐
      q_1 ──[H]──┤
      q_2 ──[H]──┤  U_rand (2^n × 2^n)  ──[measure]
      q_3 ──[H]──┘

    Layer 0 : H⊗n  — prepare equal superposition |+⟩^⊗n
    Layer 1 : U    — custom n-qubit SU(2^n) gate
    Layer 2 : Measure all (optional)

    Parameters
    ----------
    n_qubits        : number of qubits
    seed            : RNG seed
    add_measurement : include classical register + measurements

    Returns
    -------
    qc : QuantumCircuit
    U  : underlying NumPy unitary matrix (for plotting)
    """
    print(f"\n{'='*55}")
    print(f"  Building {n_qubits}-qubit circuit  (seed={seed})")
    print(f"{'='*55}")
    print(f"  Generating random SU({2**n_qubits}) unitary ...")

    U    = make_random_unitary(n_qubits=n_qubits, seed=seed)
    gate = make_unitary_gate(U, label="U_rand")

    qr = QuantumRegister(n_qubits, name="q")
    qc = QuantumCircuit(qr, name="Random Unitary Circuit")


    # Layer 1 — Custom unitary gate on all qubits
    qc.append(gate, qr[:])
    qc.barrier(label="U")

    # Layer 2 — Measurement (optional)
    if add_measurement:
        cr = ClassicalRegister(n_qubits, name="c")
        qc.add_register(cr)
        qc.measure(qr, cr)

    print(f"  Circuit depth  : {qc.depth()}")
    print(f"  Gate count     : {dict(qc.count_ops())}")
    return qc, U


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_circuit(qc: QuantumCircuit,
                 style: str = "mpl",
                 filename: str = "circuit.png",
                 fold: int = -1) -> None:
    """
    Draw and save the circuit diagram.

    Parameters
    ----------
    style    : "mpl" (matplotlib) | "text" (ASCII to stdout)
    filename : output path for PNG (used when style="mpl")
    fold     : columns per row (-1 = no wrapping)
    """
    if style == "text":
        print("\n" + str(qc.draw(output="text", fold=fold)))
        return

    fig = qc.draw(output="mpl",
                  fold=fold,
                  style={"backgroundcolor": "#FFFFFF"})
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Circuit diagram saved  → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public API — full pipeline in one call
# ─────────────────────────────────────────────────────────────────────────────
def build_and_visualise(n_qubits: int = 4,
                        seed: int = 42,
                        draw_style: str = "mpl",
                        add_measurement: bool = False,
                        circuit_file: str = "circuit.png") -> QuantumCircuit:
    """
    Full pipeline: generate unitary → build circuit → draw → heatmap → spectrum.

    Parameters
    ----------
    n_qubits        : number of qubits (default 4)
    seed            : RNG seed for the random unitary (default 42)
    draw_style      : "mpl" saves PNG; "text" prints ASCII to stdout
    add_measurement : include measurement layer (default True)
    circuit_file    : output path for circuit PNG
    heatmap_file    : output path for amplitude/phase heatmap PNG
    spectrum_file   : output path for eigenvalue spectrum PNG

    Returns
    -------
    qc : QuantumCircuit — the constructed circuit
    """
    qc, U = build_circuit(n_qubits=n_qubits, seed=seed,
                          add_measurement=add_measurement)

    print("\n  Generating visualisations ...")
    draw_circuit(qc, style=draw_style, filename=circuit_file)

    return qc



# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
def _print_matrix(M: np.ndarray, label: str = "Matrix", max_dim: int = 8):
    dim = M.shape[0]
    print(f"\n{label}  (dim={dim}x{dim})")
    if dim <= max_dim:
        for row in M:
            parts = [f"{v.real:+.4f}{v.imag:+.4f}j" for v in row]
            print("  [" + "  ".join(parts) + "]")
    else:
        print("  (matrix too large to display — use U directly)")


def demo(n_qubits: int, params=None, seed: int = 42):
    dim       = 2 ** n_qubits
    n_params  = 4 ** n_qubits - 1
    print("=" * 60)
    print(f"  General SU({dim}) Unitary  |  n_qubits = {n_qubits}")
    print(f"  Parameters   : {n_params}")
    print(f"  Matrix dim   : {dim} x {dim}")
    print("=" * 60)

    if params is None:
        print(f"  Sampling random parameters (seed={seed}) ...")
    U = general_unitary(n_qubits, params=params, seed=seed)

    _print_matrix(U, label="U = exp(i·H)")

    unitary_ok = is_unitary(U)
    su_ok      = is_special_unitary(U)
    err        = unitarity_error(U)

    print(f"\n  Checks:")
    print(f"    Unitary  (U†U = I)  : {'✓' if unitary_ok else '✗'}  (||U†U-I||={err:.2e})")
    print(f"    Special  (det=1)    : {'✓' if su_ok      else '✗'}  (det={np.linalg.det(U):.6f})")
    print()
    return U

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random n-qubit unitary circuit builder & visualiser (Qiskit 2.x)"
    )
    parser.add_argument("--nqubits",    type=int, default=4,
                        help="Number of qubits (default: 4)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="RNG seed (default: 42)")
    parser.add_argument("--draw",       type=str, default="mpl",
                        choices=["mpl", "text"],
                        help="Circuit draw style (default: mpl)")
    parser.add_argument("--no-measure", action="store_true",
                        help="Omit the measurement layer")
    args = parser.parse_args()

    build_and_visualise(
        n_qubits=args.nqubits,
        seed=args.seed,
        draw_style=args.draw,
        add_measurement=not args.no_measure,
    )

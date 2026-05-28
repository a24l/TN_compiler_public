"""
Haar-unitary helpers for MPO experiments.

The dense unitary returned by qr_haar() is already unitary. Do not divide it by
its Frobenius norm; that would scale U and break U†U = I.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import Operator
from qiskit.sysnthesis import TwoQubitBasisDecomposer
from scipy.stats import unitary_group
from qiskit.circuit.library import CXGate


def qr_haar_matrix(dim: int, seed: int) -> np.ndarray:
    """Return a Haar-random unitary matrix of shape (dim, dim)."""
    if dim < 1:
        raise ValueError("dim must be >= 1")

    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    Q, R = np.linalg.qr(Z)
    phase = np.diag(R) / np.abs(np.diag(R))
    return (Q * phase).astype(np.complex128)


def haar_random_1q(seed: int) -> np.ndarray:
    """Return a 2x2 Haar-random single-qubit unitary."""
    return qr_haar_matrix(2, seed)


def qr_haar(n_qubits: int, seed: int) -> np.ndarray:
    """Return a Haar-random n-qubit unitary of shape (2**n, 2**n)."""
    if n_qubits < 1:
        raise ValueError("n_qubits must be >= 1")
    return qr_haar_matrix(2**n_qubits, seed)


def verify_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True when U†U = I within max-entry tolerance."""
    ident = np.eye(U.shape[0], dtype=np.complex128)
    return bool(np.max(np.abs(U.conj().T @ U - ident)) < tol)


def unitary_tensor(U: np.ndarray, n_qubits: int) -> np.ndarray:
    """Tensorize a dense n-qubit operator as shape [2] * (2 * n_qubits)."""
    dim = 2**n_qubits
    if U.shape != (dim, dim):
        raise ValueError(f"Expected U shape {(dim, dim)}, got {U.shape}")
    return U.reshape([2] * (2 * n_qubits))


def qr_haar_circuit(n_qubits: int, seed: int):
    """Build a Qiskit circuit containing one dense n-qubit Haar unitary."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    U = qr_haar(n_qubits, seed)
    gate = UnitaryGate(U, label=f"U_haar\n({n_qubits} qubits)")
    qc = QuantumCircuit(n_qubits)
    qc.append(gate, range(n_qubits))
    return qc


def build_haar_circuit(n_qubits: int, base_seed: int = 42):
    """Create a product circuit with one independent Haar 1Q gate per qubit."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate

    qc = QuantumCircuit(n_qubits)
    for qubit in range(n_qubits):
        U_k = haar_random_1q(seed=base_seed + qubit)
        qc.append(UnitaryGate(U_k, label=f"U_haar\n(q{qubit})"), [qubit])
    return qc


def svd_haar(n_qubits: int, seed: int):
    """Return the SVD of a Haar-random n-qubit unitary without rescaling it."""
    U = qr_haar(n_qubits, seed)
    left, singular_values, right = np.linalg.svd(U, full_matrices=False)
    return singular_values, left, right, U

#build a SU4 KAK decomposition of a Haar-random 2-qubit unitary, without rescaling it
def su4_kak_haar():
    U = unitary_group.rvs(4, random_state=42)
    # Decompose into CX gates
    decomposer = TwoQubitBasisDecomposer(CXGate())
    circuit = decomposer(U)
    return circuit, U



def main() -> None:
    n_qubits = 4
    seed = 42
    U = qr_haar(n_qubits, seed)
    U_tensor = unitary_tensor(U, n_qubits)

    print(f"Haar unitary shape       : {U.shape}")
    print(f"U is unitary             : {verify_unitary(U)}")
    print(f"Tensorized operator shape: {U_tensor.shape}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        qc = qr_haar_circuit(n_qubits, seed)
        fig = qc.draw(output="mpl", style="clifford", fold=-1)
        fig.suptitle(
            f"Haar-Random {n_qubits}-Qubit Unitary | seed={seed}",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )
        fig.savefig("haar_circuit.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("Saved: haar_circuit.png")
    except ImportError as exc:
        print(f"Skipping Qiskit circuit drawing: {exc}")


if __name__ == "__main__":
    main()

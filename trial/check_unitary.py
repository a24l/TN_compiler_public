from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from .mps import kak1_su4_circuit
except ImportError:
    from mps import kak1_su4_circuit

@dataclass
class UnitarityReport:
    is_unitary: bool
    n_qubits: int
    matrix_dim: int
    frob_error: float        # ||U†U - I||_F  → 0 if unitary
    max_off_diag: float      # max |off-diagonal of U†U| → 0 if unitary
    min_diag: float          # min diagonal of U†U → 1 if unitary
    max_diag: float          # max diagonal of U†U → 1 if unitary
    det_magnitude: float     # |det(U)| → 1 if unitary
    condition_number: float  # sv_max/sv_min → 1 if unitary
    sv_min: float
    sv_max: float
    global_phase: Optional[complex]
    tol: float

    def __str__(self):
        status = "✓ UNITARY" if self.is_unitary else "✗ NOT UNITARY"
        lines = [
            f"{'─'*52}",
            f"  Unitarity check  [{status}]",
            f"{'─'*52}",
            f"  Qubits          : {self.n_qubits}",
            f"  Matrix dim      : {self.matrix_dim}×{self.matrix_dim}",
            f"  Tolerance       : {self.tol:.1e}",
            f"{'─'*52}",
            f"  ||U†U - I||_F   : {self.frob_error:.4e}  {'✓' if self.frob_error < self.tol else '✗'}",
            f"  Max off-diag    : {self.max_off_diag:.4e}  {'✓' if self.max_off_diag < self.tol else '✗'}",
            f"  Diag range      : [{self.min_diag:.6f}, {self.max_diag:.6f}]",
            f"  |det(U)|        : {self.det_magnitude:.8f}  {'✓' if abs(self.det_magnitude-1)<self.tol else '✗'}",
            f"  Singular values : [{self.sv_min:.6f}, {self.sv_max:.6f}]",
            f"  Condition number: {self.condition_number:.6f}",
            f"{'─'*52}",
        ]
        return "\n".join(lines)


def check_unitary(qc: QuantumCircuit, tol: float = 1e-10, verbose: bool = True) -> UnitarityReport:
    """
    Check whether a Qiskit QuantumCircuit implements a unitary operator.

    Parameters
    ----------
    qc      : QuantumCircuit — circuit without measurements/resets
    tol     : float          — absolute tolerance (default 1e-10)
    verbose : bool           — print report to stdout

    Returns
    -------
    UnitarityReport  (.is_unitary is the main boolean result)
    """
    try:
        U = Operator(qc).data
    except Exception as exc:
        raise ValueError(
            "Could not extract unitary. Remove measurements/resets first.\n"
            f"Original error: {exc}"
        ) from exc

    n = U.shape[0]
    residual     = U.conj().T @ U - np.eye(n)
    frob_error   = np.linalg.norm(residual, ord='fro')
    diag_vals    = np.abs(np.diag(U.conj().T @ U))
    off_diag_mat = residual.copy(); np.fill_diagonal(off_diag_mat, 0)
    max_off_diag = np.max(np.abs(off_diag_mat))

    sv           = np.linalg.svd(U, compute_uv=False)
    sv_min, sv_max = sv.min(), sv.max()
    cond         = sv_max / sv_min if sv_min > 0 else np.inf
    det          = np.linalg.det(U)

    is_unitary = (
        frob_error   < tol and
        max_off_diag < tol and
        abs(sv_min - 1) < tol and
        abs(sv_max - 1) < tol and
        abs(abs(det) - 1) < tol
    )

    report = UnitarityReport(
        is_unitary=is_unitary, n_qubits=qc.num_qubits, matrix_dim=n,
        frob_error=frob_error, max_off_diag=max_off_diag,
        min_diag=float(diag_vals.min()), max_diag=float(diag_vals.max()),
        det_magnitude=abs(det), condition_number=cond,
        sv_min=float(sv_min), sv_max=float(sv_max),
        global_phase=det, tol=tol,
    )
    if verbose:
        print(report)
    return report


def build_mps_circuit(seed: int = 42) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    params_left = rng.uniform(-np.pi, np.pi, 6)
    kxyz = rng.uniform(-np.pi / 4, np.pi / 4, 3)
    params_right = rng.uniform(-np.pi, np.pi, 6)
    return kak1_su4_circuit(params_left, kxyz, params_right)


def check_mps_circuit_unitary(
    seed: int = 42, tol: float = 1e-10, verbose: bool = True
) -> UnitarityReport:
    qc = build_mps_circuit(seed=seed)
    return check_unitary(qc, tol=tol, verbose=verbose)


if __name__ == "__main__":
    check_mps_circuit_unitary()

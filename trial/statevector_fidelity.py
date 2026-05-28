import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from mps import kak1_su4_circuit


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


#usage example
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    p_left  = rng.uniform(-np.pi, np.pi, 6)
    kxyz    = rng.uniform(-np.pi/4, np.pi/4, 3)
    p_right = rng.uniform(-np.pi, np.pi, 6)

    # ── Concrete circuit ──────────────────────────────────────────────────
    qc = kak1_su4_circuit(p_left, kxyz, p_right)
    sv = get_statevector(qc, verbose=True)
    print(f"Statevector shape: {sv.shape}, dtype: {sv.dtype}")
    print(sv)
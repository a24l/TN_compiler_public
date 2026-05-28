"""
MPO reconstruction diagnostic.

This module decomposes a dense n-qubit unitary into MPO tensors with a
sequential SVD and contracts those tensors back to a dense operator. It does
not attempt to turn MPO cores into local physical gates; doing that by summing
virtual bonds and projecting to the nearest unitary is not equivalent to the
original operator.
"""

from __future__ import annotations

import numpy as np


def qr_haar(dim: int, seed: int) -> np.ndarray:
    """Haar-random unitary via QR decomposition."""
    if dim < 1:
        raise ValueError("dim must be >= 1")

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(Z)
    phase = np.diag(R) / np.abs(np.diag(R))
    return (Q * phase).astype(np.complex128)


def verify_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True when U is unitary within max-entry tolerance."""
    ident = np.eye(U.shape[0], dtype=np.complex128)
    return bool(np.max(np.abs(U.conj().T @ U - ident)) < tol)


def unitary_to_mpo(M: np.ndarray, n: int, chi: int, d: int = 2):
    """
    Sequential SVD decomposition of a dense operator into MPO tensors.

    Returns a list of tensors with shape (bond_left, d_in, d_out, bond_right)
    and a list of retained singular values for each virtual bond.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if chi < 1:
        raise ValueError("chi must be >= 1")

    dim = d**n
    if M.shape != (dim, dim):
        raise ValueError(f"Expected M shape {(dim, dim)}, got {M.shape}")

    op_tensor = M.reshape([d] * n + [d] * n)
    interleaved_order = []
    for site in range(n):
        interleaved_order += [site, site + n]
    op_tensor = op_tensor.transpose(interleaved_order)

    remainder = op_tensor.reshape(d * d, (d * d) ** (n - 1))
    tensors = []
    singular_values = []
    bond_left = 1

    for _site in range(n - 1):
        rows = bond_left * d * d
        cols = remainder.size // rows
        remainder = remainder.reshape(rows, cols)

        U, S, Vh = np.linalg.svd(remainder, full_matrices=False)
        keep = min(len(S), chi)
        U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]

        tensors.append(U.reshape(bond_left, d, d, keep))
        singular_values.append(S)
        remainder = np.diag(S) @ Vh
        bond_left = keep

    tensors.append(remainder.reshape(bond_left, d, d, 1))
    return tensors, singular_values


def reconstruct_from_mpo(tensors: list[np.ndarray]) -> np.ndarray:
    """Contract MPO tensors back to a dense matrix."""
    if not tensors:
        raise ValueError("Expected at least one MPO tensor.")

    result = tensors[0][0]  # (d_in, d_out, bond_right)

    for tensor in tensors[1:]:
        if result.shape[-1] != tensor.shape[0]:
            raise ValueError(
                f"Bond mismatch: left contraction has {result.shape[-1]}, "
                f"next tensor has {tensor.shape[0]}"
            )

        shape = result.shape
        contracted = result.reshape(-1, shape[-1]) @ tensor.reshape(tensor.shape[0], -1)
        _, d_in, d_out, bond_right = tensor.shape
        contracted = contracted.reshape(*shape[:-1], d_in, d_out, bond_right)
        contracted = contracted.transpose(0, 2, 1, 3, 4)
        result = contracted.reshape(shape[0] * d_in, shape[1] * d_out, bond_right)

    return result.reshape(result.shape[0], result.shape[1])


def reconstruction_metrics(M: np.ndarray, M_recon: np.ndarray) -> dict:
    """Compute dense reconstruction quality metrics."""
    if M.shape != M_recon.shape:
        raise ValueError(f"Shape mismatch: {M.shape} vs {M_recon.shape}")

    dim = M.shape[0]
    diff = M_recon - M
    return {
        "shape": M_recon.shape,
        "process_fidelity": float((abs(np.trace(M.conj().T @ M_recon)) / dim) ** 2),
        "fro_error": float(np.linalg.norm(diff)),
        "relative_fro_error": float(np.linalg.norm(diff) / np.linalg.norm(M)),
        "unitarity_residual": float(
            np.max(np.abs(M_recon.conj().T @ M_recon - np.eye(dim, dtype=np.complex128)))
        ),
    }


def analyze_mpo(n_qubits: int = 4, chi: int = 16, seed: int = 42, verbose: bool = True) -> dict:
    """Build a target unitary, decompose/reconstruct it as an MPO, and report metrics."""
    dim = 2**n_qubits
    M = qr_haar(dim, seed)
    tensors, singular_values = unitary_to_mpo(M, n_qubits, chi)
    M_recon = reconstruct_from_mpo(tensors)
    metrics = reconstruction_metrics(M, M_recon)

    result = {
        "M": M,
        "M_recon": M_recon,
        "tensors": tensors,
        "singular_values": singular_values,
        "metrics": metrics,
    }

    if verbose:
        print(f"\nMPO reconstruction diagnostic: n={n_qubits}, chi={chi}, seed={seed}")
        print(f"  target shape       : {M.shape}")
        print(f"  target unitary     : {verify_unitary(M)}")
        print("  tensor shapes      :")
        for site, tensor in enumerate(tensors):
            print(f"    site {site}: {tensor.shape}")
        print("  retained bonds     :")
        for bond, values in enumerate(singular_values):
            print(f"    bond {bond}-{bond + 1}: chi_eff={len(values)}")
        print(f"  reconstructed shape: {metrics['shape']}")
        print(f"  process fidelity   : {metrics['process_fidelity']:.12f}")
        print(f"  Frobenius error    : {metrics['fro_error']:.3e}")
        print(f"  relative Fro error : {metrics['relative_fro_error']:.3e}")
        print(f"  unitarity residual : {metrics['unitarity_residual']:.3e}")

    return result


def main() -> None:
    for chi in (1, 2, 4, 16):
        analyze_mpo(n_qubits=4, chi=chi, seed=42, verbose=True)


if __name__ == "__main__":
    main()

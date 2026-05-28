import numpy as np
import quimb.tensor as qtn
from numpy.linalg import qr


def haar_random_unitary(num_qubits):
    """Generate a Haar-random unitary matrix using QR decomposition."""
    n = 2 ** num_qubits
    a = np.random.normal(size=(n, n))
    b = np.random.normal(size=(n, n))
    z = a + 1j * b

    q, r = qr(z)
    lam = np.diag([r[i, i] / np.abs(r[i, i]) for i in range(n)])
    return np.dot(q, lam)


def _as_unitary_matrix(matrix_u):
    """Normalize supported unitary-like inputs to a NumPy array."""
    if isinstance(matrix_u, np.ndarray):
        return matrix_u

    data = getattr(matrix_u, "data", None)
    if data is not None and not isinstance(data, memoryview):
        return np.asarray(data)

    return np.asarray(matrix_u)


def quimb_unitary_tensor(matrix_u, n_qubits):
    """Return the corresponding Quimb tensor with output indices first."""
    matrix_m = _as_unitary_matrix(matrix_u)
    expected_dim = 2 ** n_qubits
    if matrix_m.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"Expected a ({expected_dim}, {expected_dim}) unitary matrix, got {matrix_m.shape}."
        )

    tensor_shape = [2] * (2 * n_qubits)
    tensor_m_data = matrix_m.reshape(tensor_shape)

    input_inds = [f"k{i}" for i in range(n_qubits)]
    output_inds = [f"k{i}_out" for i in range(n_qubits)]

    return qtn.Tensor(
        data=tensor_m_data,
        inds=(*output_inds, *input_inds),
        tags={"random_unitary", "M"},
    )

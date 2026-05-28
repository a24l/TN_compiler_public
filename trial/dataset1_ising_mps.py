"""
dataset1_ising_mps.py
=====================
Dataset 1: 100 Ising-chain MPS unitaries
Based on: NFL Theorem for TN ML (arXiv:2412.05674), Fig. 1 scheme
         — supervised learning of unitaries encoded in MPS states.

Whiteboard design:
  Dataset 1 = { x_i, i=1..100 }
  x_i = MPS of U_i|0> where U_i is a random Ising chain unitary
        x_i = A[1] ⊗ A[2] ⊗ ... ⊗ A[n]  (local tensors U_1, U_2, ... from NFL Fig 1)

Usage:
    from dataset1_ising_mps import Dataset1, MPS
    ds = Dataset1(n_samples=100, n_qubits=4, n_layers=3, bond_dim=4)
    sample = ds[0]   # dict with keys: unitary, mps_tensors, output_state, fidelity
    ds.summary()
"""

import numpy as np


# ────────────────────────────────────────────────────────────────
# Gate primitives
# ────────────────────────────────────────────────────────────────

def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]])


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]])


def local_unitary(angles) -> np.ndarray:
    """Single-site Rz(a)·Ry(b)·Rz(c) unitary (ZYZ decomposition)."""
    a, b, c = angles
    return rz(a) @ ry(b) @ rz(c)


def zz_gate(theta: float) -> np.ndarray:
    """Two-site Ising ZZ coupling gate: exp(-i theta/2  Z⊗Z)."""
    phi = theta / 2
    return np.diag([np.exp(-1j * phi), np.exp(1j * phi),
                    np.exp(1j * phi),  np.exp(-1j * phi)])


# ────────────────────────────────────────────────────────────────
# Ising chain circuit
# ────────────────────────────────────────────────────────────────

def build_ising_unitary(n_qubits: int, n_layers: int, params: dict) -> np.ndarray:
    """
    Construct full 2^n × 2^n unitary for an n-qubit Ising chain circuit.

    Parameters
    ----------
    n_qubits  : number of qubits
    n_layers  : number of (single-site + ZZ) layers
    params    : dict with keys
                  'single' : ndarray (n_layers, n_qubits, 3)   — ZYZ angles
                  'zz'     : ndarray (n_layers, n_qubits-1)   — ZZ coupling angles
    """
    d = 2 ** n_qubits
    U = np.eye(d, dtype=complex)

    single = params["single"]  # (L, n, 3)
    zz     = params["zz"]      # (L, n-1)

    for layer in range(n_layers):
        # ── Single-site layer ──
        layer_U = np.array([[1.0 + 0j]])
        for q in range(n_qubits):
            layer_U = np.kron(layer_U, local_unitary(single[layer, q]))
        U = layer_U @ U

        # ── ZZ coupling layer ──
        for bond in range(n_qubits - 1):
            left  = np.eye(2 ** bond,               dtype=complex)
            right = np.eye(2 ** (n_qubits - bond - 2), dtype=complex)
            # Handle edge qubits (bond=0 or bond=n-2) gracefully
            if bond == 0:
                full_zz = np.kron(zz_gate(zz[layer, bond]), right)
            elif bond == n_qubits - 2:
                full_zz = np.kron(left, zz_gate(zz[layer, bond]))
            else:
                full_zz = np.kron(np.kron(left, zz_gate(zz[layer, bond])), right)
            U = full_zz @ U

    return U


# ────────────────────────────────────────────────────────────────
# MPS class (iterated SVD)
# ────────────────────────────────────────────────────────────────

class MPS:
    """
    Matrix Product State via iterated SVD truncation.

    Attributes
    ----------
    tensors    : list of ndarray, shapes (D_l, d, D_r)
    n_qubits   : int
    bond_dim   : int  (max bond dimension D)
    fidelity   : float  |<psi_mps | psi>|^2
    """

    def __init__(self, state_vec: np.ndarray, n_qubits: int, bond_dim: int):
        self.n_qubits = n_qubits
        self.bond_dim = bond_dim
        self.tensors  = self._svd_compress(state_vec, n_qubits, bond_dim)
        self.fidelity = self._compute_fidelity(state_vec)

    @staticmethod
    def _svd_compress(state_vec, n_qubits, bond_dim):
        d = 2
        tensors = []
        psi     = state_vec.reshape([d] * n_qubits)
        D_left  = 1

        for k in range(n_qubits - 1):
            psi    = psi.reshape(D_left * d, -1)
            U, S, Vt = np.linalg.svd(psi, full_matrices=False)
            D_keep = min(bond_dim, len(S))
            A      = U[:, :D_keep].reshape(D_left, d, D_keep)
            tensors.append(A)
            psi    = np.diag(S[:D_keep]) @ Vt[:D_keep, :]
            D_left = D_keep

        tensors.append(psi.reshape(D_left, d, 1))
        return tensors

    def to_state_vec(self) -> np.ndarray:
        """Reconstruct the (truncated) state vector from MPS tensors."""
        result = self.tensors[0].reshape(2, -1)  # (d, D)
        for k in range(1, self.n_qubits):
            A      = self.tensors[k]              # (D_l, d, D_r)
            result = np.tensordot(result, A, axes=([1], [0]))  # (..., d, D_r)
            result = result.reshape(-1, A.shape[2])
        vec  = result.reshape(-1)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-12 else vec

    def _compute_fidelity(self, target: np.ndarray) -> float:
        mps_vec = self.to_state_vec()
        return float(abs(np.dot(mps_vec.conj(), target)) ** 2)

    def bond_dimensions(self):
        return [t.shape[2] for t in self.tensors]

    def __repr__(self):
        shapes = [t.shape for t in self.tensors]
        return (f"MPS(n={self.n_qubits}, D={self.bond_dim}, "
                f"fidelity={self.fidelity:.6f}, shapes={shapes})")


# ────────────────────────────────────────────────────────────────
# Dataset 1
# ────────────────────────────────────────────────────────────────

class Dataset1:
    """
    Dataset 1  =  { x_i, i = 1..N }
    Each x_i is an MPS encoding of U_i|0>, where U_i is a random
    n-qubit Ising chain unitary (ZYZ single-site + ZZ couplings).

    Matches the NFL paper Fig. 1 setup:
      — data encoded into unitary-embedded MPS states (local tensors U_i)
      — label = U_i (the target unitary to learn)

    Parameters
    ----------
    n_samples  : number of unitaries (default 100)
    n_qubits   : system size          (default 4)
    n_layers   : circuit depth        (default 3)
    bond_dim   : MPS bond dimension D (default 4)
    seed       : RNG seed             (default 0)
    """

    def __init__(self, n_samples=100, n_qubits=4, n_layers=3,
                 bond_dim=4, seed=0):
        self.n_samples = n_samples
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.bond_dim  = bond_dim
        self.rng       = np.random.default_rng(seed)
        self.samples   = self._generate()

    def _generate(self):
        samples = []
        input_state = np.zeros(2 ** self.n_qubits, dtype=complex)
        input_state[0] = 1.0  # |0...0>

        for i in range(self.n_samples):
            params = {
                "single": self.rng.uniform(0, 2 * np.pi,
                           (self.n_layers, self.n_qubits, 3)),
                "zz":     self.rng.uniform(0, np.pi,
                           (self.n_layers, self.n_qubits - 1)),
            }
            U     = build_ising_unitary(self.n_qubits, self.n_layers, params)
            psi   = U @ input_state
            mps_i = MPS(psi, self.n_qubits, self.bond_dim)

            samples.append({
                "index":        i,
                "params":       params,
                "unitary":      U,          # 2^n × 2^n target M
                "input_state":  input_state.copy(),
                "output_state": psi,        # M|0> = |psi_i>  (label)
                "mps":          mps_i,      # x_i  (feature)
                "fidelity":     mps_i.fidelity,
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def fidelities(self):
        return np.array([s["fidelity"] for s in self.samples])

    def summary(self):
        f = self.fidelities()
        print(f"Dataset 1  —  Ising chain MPS unitaries")
        print(f"  Samples  : {self.n_samples}")
        print(f"  Qubits   : n = {self.n_qubits}")
        print(f"  Layers   : L = {self.n_layers}")
        print(f"  Bond dim : D = {self.bond_dim}")
        print(f"  Hilbert  : 2^n = {2**self.n_qubits}")
        print(f"  Fidelity : mean={f.mean():.6f}  min={f.min():.6f}  max={f.max():.6f}")
        print(f"  MPS shapes (sample 0):")
        for k, T in enumerate(self.samples[0]["mps"].tensors):
            print(f"    A[{k}]: {T.shape}")


# ────────────────────────────────────────────────────────────────
# Quick test
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = Dataset1(n_samples=100, n_qubits=4, n_layers=3, bond_dim=4, seed=0)
    ds.summary()
    print()
    print("Sample 0 MPS:")
    print(" ", ds[0]["mps"])

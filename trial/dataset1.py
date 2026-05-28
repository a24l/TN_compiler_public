# Generate Dataset 1: 100 Ising chain MPS unitaries
# Based on NFL paper Fig 1 - local unitaries embedded in MPS chain

import numpy as np
from itertools import product

np.random.seed(42)

# ─────────────────────────────────────────────
# 1. Single-site local unitary building blocks
# ─────────────────────────────────────────────
def rx(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -1j*s], [-1j*s, c]])

def ry(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]])

def rz(theta):
    return np.array([[np.exp(-1j*theta/2), 0], [0, np.exp(1j*theta/2)]])

def zz_ising_gate(theta):
    """Two-site Ising ZZ coupling: exp(-i theta/2 Z⊗Z)"""
    phi = theta / 2
    return np.diag([np.exp(-1j*phi), np.exp(1j*phi),
                    np.exp(1j*phi), np.exp(-1j*phi)])

def make_local_unitary(angles):
    """Single-site unitary: Rz(a) Ry(b) Rz(c)"""
    a, b, c = angles
    return rz(a) @ ry(b) @ rz(c)

# ─────────────────────────────────────────────
# 2. Build one Ising chain unitary (n=4 qubits)
#    Layer structure: single-site rotations + ZZ couplings
# ─────────────────────────────────────────────
def build_ising_chain_unitary(n_qubits, n_layers, params):
    """
    Build full n_qubit unitary from Ising chain circuit.
    params: (n_layers, n_qubits, 3) for single-site angles
            + (n_layers, n_qubits-1) for ZZ coupling angles
    """
    d = 2**n_qubits
    U = np.eye(d, dtype=complex)
    
    single_angles = params['single']   # (n_layers, n_qubits, 3)
    zz_angles     = params['zz']       # (n_layers, n_qubits-1)
    
    for layer in range(n_layers):
        # Single-site rotations
        layer_U = np.array([[1.0+0j]])
        for q in range(n_qubits):
            u_q = make_local_unitary(single_angles[layer, q])
            layer_U = np.kron(layer_U, u_q)
        U = layer_U @ U
        
        # ZZ Ising couplings (nearest neighbour)
        for bond in range(n_qubits - 1):
            zz_2 = zz_ising_gate(zz_angles[layer, bond])
            # Embed 2-site gate acting on qubits (bond, bond+1) into full space
            ops = [np.eye(2, dtype=complex)] * n_qubits
            # Replace with 2-site block
            left  = np.eye(2**bond, dtype=complex) if bond > 0 else np.array([[1.0+0j]])
            right = np.eye(2**(n_qubits-bond-2), dtype=complex) if bond < n_qubits-2 else np.array([[1.0+0j]])
            full_zz = np.kron(np.kron(left, zz_2), right)
            U = full_zz @ U
    
    return U

# ─────────────────────────────────────────────
# 3. MPS tensor representation (bond dim D=2)
# ─────────────────────────────────────────────
def state_to_mps(state_vec, n_qubits, bond_dim):
    """
    Convert a state vector to MPS via iterated SVD truncation.
    Returns list of tensors [A1, A2, ..., An].
    Each Ak has shape (D_left, d, D_right) with d=2.
    """
    d = 2
    tensors = []
    psi = state_vec.reshape([d] * n_qubits)
    D_left = 1
    
    for k in range(n_qubits - 1):
        psi = psi.reshape(D_left * d, -1)
        U, S, Vt = np.linalg.svd(psi, full_matrices=False)
        D_keep = min(bond_dim, len(S))
        U = U[:, :D_keep]
        S = S[:D_keep]
        Vt = Vt[:D_keep, :]
        # Reshape U into MPS tensor
        A = U.reshape(D_left, d, D_keep)
        tensors.append(A)
        psi = np.diag(S) @ Vt
        D_left = D_keep
    
    # Last tensor
    A = psi.reshape(D_left, d, 1)
    tensors.append(A)
    return tensors

def mps_fidelity(tensors, state_vec, n_qubits):
    """Compute fidelity |<psi_mps|psi>|^2 by reconstructing MPS state vector."""
    d = 2
    # Contract MPS back to state vector
    result = tensors[0][:, :, :]  # (1, d, D)
    result = result.reshape(d, -1)  # (d, D)
    
    for k in range(1, n_qubits):
        A = tensors[k]  # (D_left, d, D_right)
        D_left, _, D_right = A.shape
        # result shape: (d^k, D_left), A: (D_left, d, D_right)
        result = np.tensordot(result, A, axes=([1], [0]))  # (d^k, d, D_right)
        result = result.reshape(-1, D_right)
    
    mps_vec = result.reshape(-1)
    norm = np.linalg.norm(mps_vec)
    if norm < 1e-12:
        return 0.0
    mps_vec = mps_vec / norm
    return abs(np.dot(mps_vec.conj(), state_vec))**2

# ─────────────────────────────────────────────
# 4. Generate Dataset 1: 100 MPS unitaries
# ─────────────────────────────────────────────
N_SAMPLES  = 100
N_QUBITS   = 4
N_LAYERS   = 3
BOND_DIM   = 4
D_HILBERT  = 2**N_QUBITS

print(f"Generating Dataset 1: {N_SAMPLES} Ising-chain MPS unitaries")
print(f"System: n={N_QUBITS} qubits, {N_LAYERS} layers, bond dim D={BOND_DIM}")
print("="*60)

dataset1 = []
rng = np.random.default_rng(0)

input_state = np.zeros(D_HILBERT, dtype=complex)
input_state[0] = 1.0  # |0000>

for i in range(N_SAMPLES):
    params = {
        'single': rng.uniform(0, 2*np.pi, (N_LAYERS, N_QUBITS, 3)),
        'zz':     rng.uniform(0, np.pi,   (N_LAYERS, N_QUBITS-1))
    }
    
    U = build_ising_chain_unitary(N_QUBITS, N_LAYERS, params)
    output_state = U @ input_state
    
    # Encode into MPS
    mps_tensors = state_to_mps(output_state, N_QUBITS, BOND_DIM)
    fid = mps_fidelity(mps_tensors, output_state, N_QUBITS)
    
    dataset1.append({
        'index':       i,
        'params':      params,
        'unitary':     U,
        'output_state': output_state,
        'mps_tensors': mps_tensors,
        'fidelity':    fid
    })

fidelities = [s['fidelity'] for s in dataset1]
print(f"Mean fidelity (MPS D={BOND_DIM}): {np.mean(fidelities):.6f}")
print(f"Min  fidelity: {np.min(fidelities):.6f}")
print(f"Max  fidelity: {np.max(fidelities):.6f}")
print(f"\nDataset 1 ready: {len(dataset1)} samples")
print(f"Each sample: (U: {D_HILBERT}x{D_HILBERT} unitary, MPS: {N_QUBITS} tensors, fidelity)")

# Show MPS tensor shapes for sample 0
print("\nSample 0 MPS tensor shapes:")
for k, T in enumerate(dataset1[0]['mps_tensors']):
    print(f"  A[{k}]: shape {T.shape}")
import numpy as np
from numpy import random as rng
from scipy.linalg import svd
from quantum_unitary import build_circuit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix

n=3
seed = 42
psi= rng.rand(2**n)
print(psi.shape)
print(psi)


psi_norm = psi / np.linalg.norm(psi)

psi_f = np.reshape(psi_norm, (2,2,2))
print(psi_f.shape)
print(psi_f)


psi_to_matrix = np.reshape(psi_f, (2, 2**(n-1)))
print(psi_to_matrix.shape)
print(psi_to_matrix)


qc, U = build_circuit(n_qubits=n, seed=seed, add_measurement=False)
sv = Statevector.from_instruction(qc)
psi_sv = sv.data
print(psi_sv.shape)
print(psi_sv)

print(U.shape)
print(U)


psi_mps = U.reshape([2] * n * 2) 
print(psi_mps.shape)
print(psi_mps)
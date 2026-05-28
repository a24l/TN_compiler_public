from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary, Operator
import matplotlib.pyplot as plt
from qiskit.circuit.random import random_circuit

import numpy as np
import quimb.tensor as qtn

from graph_states import create_1d_graph_state_tn
from unitaries import haar_random_unitary, quimb_unitary_tensor


num_qubits = 4

max_bond_dimension = 4

tn_graph_state = create_1d_graph_state_tn(num_qubits) 

haar_U = haar_random_unitary(num_qubits)

# Create the Quimb Unitary Tensor
M_tensor = quimb_unitary_tensor(haar_U, num_qubits)

# ADD M TO THE GRAPH STATE NETWORK
# Because M's inputs are 'k0', 'k1', etc., Quimb automatically links them to the Graph State!
tn_uncompiled = tn_graph_state & M_tensor

##### compile and contract to MPS

def compile_to_mps(tn, max_bond=4, n_qubits=num_qubits):
    '''Compile the TN to an MPS with a specified maximum bond dimension.'''
    
    # 1. Contract the entire messy network into a single dense tensor.
    # This mathematically computes the exact quantum state amplitudes.
    dense_tensor = tn.contract()
    
    # 2. Reorder the physical indices so qubit 0 is first, qubit 1 is second, etc.
    out_inds = [f'k{i}_out' for i in range(n_qubits)]
    dense_tensor.transpose_(*out_inds)
    
    # Extract the raw numpy array (shape: 2x2x2x2)
    dense_array = dense_tensor.data
    
    # 3. Decompose the dense array into an MPS using SVD
    # This is the step where the No-Free-Lunch truncation happens!
    dims = [2] * n_qubits  # Each physical leg has dimension 2 (qubits)
    
    tn_compiled_mps = qtn.MatrixProductState.from_dense(
        dense_array, 
        dims=dims, 
        max_bond=max_bond
    )
    
    # (Optional) Re-tag the physical indices so they match our plotting expectations
    for i in range(n_qubits):
        # By default, from_dense names indices 'k0', 'k1'. We rename them to our out_inds.
        tn_compiled_mps.reindex({f'k{i}': f'k{i}_out'}, inplace=True)
        tn_compiled_mps[i].add_tag(f'q{i}') # add tags for plotting

    return tn_compiled_mps



tn_compiled_mps = compile_to_mps(tn_uncompiled, max_bond=4, n_qubits=num_qubits)

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: The Raw Graph State
tn_graph_state.draw(color=['node', 'cz'], ax=axs[0], show_inds=True, node_size_mapped=400)
axs[0].set_title("1. Raw 1D Graph State")

# Plot 2: Graph State + Random Unitary M
tn_uncompiled.draw(color=['node', 'cz', 'M'], ax=axs[1], show_inds=True, node_size_mapped=400)
axs[1].set_title("2. Uncompiled: Graph State + Unitary M")

# Plot 3: The Final Compiled MPS
# The compiler has absorbed everything into a clean 1D chain of tensors.
tn_compiled_mps.draw(ax=axs[2], show_inds=True, node_size_mapped=400)
axs[2].set_title(f"3. Compiled MPS (Max Bond D={max_bond_dimension})")

plt.tight_layout()
plt.show()
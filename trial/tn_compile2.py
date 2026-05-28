import networkx as nx
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from qiskit.quantum_info import random_unitary
from unitaries import haar_random_unitary, quimb_unitary_tensor

num_qubits = 4
max_bond_dimension = 4

# ==========================================
# STEP 1: DEFINE PHYSICS & GRAPH STATE (WITH WIRE TRACKING)
# ==========================================
plus_state = np.array([1.0, 1.0]) / np.sqrt(2)
cz_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
cz_tensor = cz_matrix.reshape(2, 2, 2, 2)

def create_1d_graph_state_tn(n_qubits):
    """Creates a 1D Graph State Tensor Network with strict wire tracking."""
    G = nx.path_graph(n_qubits)
    tn = qtn.TensorNetwork([])
    
    # COMPILER MEMORY: Track the current name of the wire for each qubit
    current_wire = {i: f'k{i}_0' for i in range(n_qubits)}
    
    # Add |+> nodes
    for node in G.nodes():
        tn.add_tensor(qtn.Tensor(data=plus_state, inds=(current_wire[node],), tags={f'q{node}', 'node'}))
        
    # Add CZ edges
    edge_step = 1
    for edge in G.edges():
        u, v = edge
        in_u, in_v = current_wire[u], current_wire[v]
        
        # Give the outputs brand new, unique names
        out_u, out_v = f'k{u}_{edge_step}', f'k{v}_{edge_step}'
        
        tn.add_tensor(qtn.Tensor(
            data=cz_tensor, 
            inds=(in_u, in_v, out_u, out_v), 
            tags={'cz', f'edge_{edge_step}'}
        ))
        
        # Update the compiler's memory
        current_wire[u] = out_u
        current_wire[v] = out_v
        edge_step += 1
        
    return tn, current_wire

# Generate the TN, and also get the final wire names from the compiler memory!
tn_graph_state, final_graph_wires = create_1d_graph_state_tn(num_qubits)


# ==========================================
# STEP 2: APPLY RANDOM TARGET UNITARY M
# ==========================================
haar_U = haar_random_unitary(num_qubits) # This is a 2^N x 2^N matrix, e.g. (16, 16) for 4 qubits
tensor_shape = [2] * (2 * num_qubits)
tensor_M_data = haar_U.reshape(tensor_shape)

# INPUTS: Must exactly match the final wires coming out of the Graph State
in_inds = [final_graph_wires[i] for i in range(num_qubits)]
# OUTPUTS: The final dangling legs of the entire network
out_inds = [f'k{i}_out' for i in range(num_qubits)]

M_tensor = qtn.Tensor(data=tensor_M_data, inds=(*out_inds, *in_inds), tags={'random_unitary', 'M'})

# Combine them! (Because wire names match perfectly, Quimb connects them)
tn_uncompiled = tn_graph_state & M_tensor


# ==========================================
# STEP 3: CONTRACT AND COMPILE TO MPS
# ==========================================
def compile_to_mps(tn, max_bond=4, n_qubits=num_qubits):
    '''Compile the TN to an MPS with a specified maximum bond dimension.'''
    # 1. Contract the entire messy network into a single dense tensor
    # (Because we fixed the wires, this will now succeed!)
    dense_tensor = tn.contract()
    
    # 2. Reorder the physical indices so qubit 0 is first, etc.
    out_inds = [f'k{i}_out' for i in range(n_qubits)]
    dense_tensor.transpose_(*out_inds)
    dense_array = dense_tensor.data
    
    # 3. Decompose into an MPS using SVD
    dims = [2] * n_qubits
    tn_compiled_mps = qtn.MatrixProductState.from_dense(
        dense_array, 
        dims=dims, 
        max_bond=max_bond
    )

    
    # 4. Re-tag for plotting
    for i in range(n_qubits):
        tn_compiled_mps.reindex({f'k{i}': f'k{i}_out'}, inplace=True)
        tn_compiled_mps[i].add_tag(f'q{i}')
        
    return tn_compiled_mps

tn_compiled_mps = compile_to_mps(tn_uncompiled, max_bond=max_bond_dimension, n_qubits=num_qubits)


# ==========================================
# STEP 4: VISUALIZE
# ==========================================
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

tn_graph_state.draw(color=['node', 'cz'], ax=axs[0], show_inds=True, node_size=2, layout='circular')
axs[0].set_title("1. Raw 1D Graph State")

tn_uncompiled.draw(color=['node', 'cz', 'M'], ax=axs[1], show_inds=True, node_size=2, layout='spring')
axs[1].set_title("2. Uncompiled: Graph State + Unitary M")

tn_compiled_mps.draw(ax=axs[2], show_inds=True, node_size=2, layout='spring')
axs[2].set_title(f"3. Compiled MPS (Max Bond D={max_bond_dimension})")

plt.tight_layout()
plt.savefig("tn_compilation_static.png", dpi=150, bbox_inches='tight')
plt.show()
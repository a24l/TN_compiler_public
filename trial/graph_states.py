import networkx as nx
import numpy as np
import quimb.tensor as qtn
import matplotlib.pyplot as plt
from qiskit.quantum_info import random_unitary

# --- 1. DEFINE THE PHYSICS TENSORS ---
# The |+> state: shape (2,)
plus_state = np.array([1.0, 1.0]) / np.sqrt(2)

# The CZ gate: shape (2, 2, 2, 2) -> (in1, in2, out1, out2)
cz_matrix = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1]])
cz_tensor = cz_matrix.reshape(2, 2, 2, 2)

# --- 2. CREATE THE GRAPHS USING NETWORKX ---
# Let's create a 1D Graph (Line) and a 2D Graph (Grid)

# 1D Graph State (e.g., 5 qubits)
G_1d = nx.path_graph(5)

# 2D Graph State (e.g., 3x3 grid = 9 qubits)
G_2d = nx.grid_2d_graph(3, 3)
# 2D nodes are tuples like (0,1). Let's convert them to flat integers (0 to 8) for the compiler.
G_2d = nx.convert_node_labels_to_integers(G_2d)

# --- 3. THE COMPILER FUNCTION: GRAPH -> TENSOR NETWORK ---
def compile_graph_to_tn(graph):
    """
    Takes a networkx graph and compiles it into an uncontracted Tensor Network.
    """
    tn = qtn.TensorNetwork([])
    
    # Step A: Add a |+> state tensor for every node
    for node in graph.nodes():
        # We give it an index name like 'k0', 'k1' to represent the physical qubit
        tn.add_tensor(qtn.Tensor(data=plus_state, inds=(f'k{node}',), tags={f'q{node}', 'node'}))
        
    # Step B: Add a CZ tensor for every edge
    edge_count = 0
    for edge in graph.edges():
        u, v = edge
        
        # The CZ gate absorbs the current physical indices of u and v, 
        # and outputs new physical indices.
        in_u = f'k{u}'
        in_v = f'k{v}'
        out_u = f'k{u}_new'
        out_v = f'k{v}_new'
        
        tn.add_tensor(qtn.Tensor(data=cz_tensor, 
                                 inds=(in_u, in_v, out_u, out_v), 
                                 tags={'cz', f'edge_{edge_count}'}))
        
        # Rename the output indices back to standard 'k' so subsequent gates attach correctly
        tn.reindex({out_u: f'k{u}', out_v: f'k{v}'}, inplace=True)
        edge_count += 1
        
    return tn

# --- 4. RUN THE COMPILER ---
print("Compiling 1D Graph State...")
tn_1d = compile_graph_to_tn(G_1d)
print(f"1D TN has {tn_1d.num_tensors} tensors and {tn_1d.num_indices} indices.")

print("\nCompiling 2D Graph State...")
tn_2d = compile_graph_to_tn(G_2d)
print(f"2D TN has {tn_2d.num_tensors} tensors and {tn_2d.num_indices} indices.")


fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: 1D NetworkX Graph
plt.sca(axs[0, 0])
nx.draw(G_1d, with_labels=True, node_color='lightblue', font_weight='bold', node_size=200)
axs[0, 0].set_title("1. Original 1D Graph (NetworkX)")

# Plot 2: 1D Quimb Tensor Network
# We use the 'color' argument to color nodes differently based on their tags!
tn_1d.draw(color=['node', 'cz'], 
           ax=axs[0, 1], 
           show_inds=True, # Show the physical indices (dangling lines)
           node_size=2)
axs[0, 1].set_title("2. Compiled 1D Tensor Network (Quimb)")

# Plot 3: 2D NetworkX Graph
plt.sca(axs[1, 0])
side = int(np.sqrt(G_2d.number_of_nodes()))
pos = {i: (i % side, 1 - i // side) for i in G_2d.nodes()}
nx.draw(G_2d, pos=pos, with_labels=True, node_color='lightgreen', font_weight='bold', node_size=200)
axs[1, 0].set_title("3. Original 2D Grid Graph (NetworkX)")

# Plot 4: 2D Quimb Tensor Network
tn_2d.draw(color=['node', 'cz'], 
           ax=axs[1, 1], 
           show_inds=True,
           node_size=2)
axs[1, 1].set_title("4. Compiled 2D Tensor Network (Quimb)")

plt.tight_layout()
plt.savefig("graph_state_tn.png", dpi=150, bbox_inches='tight')
print("Plot saved to graph_state_tn.png")


##### create a 1d graph state for qubits
def create_1d_graph_state_tn(n_qubits):
    """Creates an uncontracted 1D Graph State Tensor Network."""
    G = nx.path_graph(n_qubits)
    tn = qtn.TensorNetwork([])
    
    # Add |+> nodes
    for node in G.nodes():
        tn.add_tensor(qtn.Tensor(data=plus_state, inds=(f'k{node}',), tags={f'q{node}', 'node'}))
        
    # Add CZ edges
    edge_count = 0
    for edge in G.edges():
        u, v = edge
        in_u, in_v = f'k{u}', f'k{v}'
        out_u, out_v = f'k{u}_new', f'k{v}_new'
        tn.add_tensor(qtn.Tensor(data=cz_tensor, inds=(in_u, in_v, out_u, out_v), tags={'cz', f'edge_{edge_count}'}))
        tn.reindex({out_u: f'k{u}', out_v: f'k{v}'}, inplace=True)
        edge_count += 1
    return tn


num_qubits = 4
tn_graph_state = create_1d_graph_state_tn(num_qubits)
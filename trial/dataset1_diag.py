"""
dataset1_diag.py
=========================
Builds Dataset 1 (100 Ising-chain MPS unitaries) using quimb natively,
then visualizes the tensor network diagrams with quimb's built-in draw()
and networkx for layout — matching the whiteboard and NFL Fig. 1.

Requirements:
    pip install quimb[tensor] networkx matplotlib autoray
"""

import numpy as np
import quimb as qu
import quimb.tensor as qtn
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

os.makedirs('output', exist_ok=True)
rng = np.random.default_rng(0)

N_QUBITS  = 4
N_LAYERS  = 3
N_SAMPLES = 100
BOND_DIM  = 4

# ─────────────────────────────────────────────────────────────────
# 1. Build one Ising-chain circuit with quimb.Circuit
#    Matches NFL Fig. 1: local unitaries U_i on each site + ZZ couplings
# ─────────────────────────────────────────────────────────────────

def make_ising_circuit_quimb(n_qubits, n_layers, params, use_mps=True):
    """
    Build Ising circuit using quimb's CircuitMPS (keeps state as MPS).
    Gate sequence: Rz·Ry·Rz on each qubit + ZZPhase coupling.
    """
    if use_mps:
        circ = qtn.CircuitMPS(n_qubits, max_bond=BOND_DIM)
    else:
        circ = qtn.Circuit(n_qubits)

    single = params['single']   # (L, n, 3) ZYZ angles
    zz     = params['zz']       # (L, n-1)  ZZ angles

    for layer in range(n_layers):
        # Single-site ZYZ rotations (Rz · Ry · Rz)
        for q in range(n_qubits):
            a, b, c = single[layer, q]
            circ.apply_gate('RZ', a, q)
            circ.apply_gate('RY', b, q)
            circ.apply_gate('RZ', c, q)

        # ZZ Ising couplings between nearest neighbours
        for bond in range(n_qubits - 1):
            theta = zz[layer, bond]
            circ.apply_gate('RZZ', theta, bond, bond + 1)

    return circ


# ─────────────────────────────────────────────────────────────────
# 2. Generate Dataset 1 with quimb
# ─────────────────────────────────────────────────────────────────

print("Generating Dataset 1 with quimb CircuitMPS...")
dataset1 = []

for i in range(N_SAMPLES):
    params = {
        'single': rng.uniform(0, 2 * np.pi, (N_LAYERS, N_QUBITS, 3)),
        'zz':     rng.uniform(0, np.pi,     (N_LAYERS, N_QUBITS - 1)),
    }
    circ = make_ising_circuit_quimb(N_QUBITS, N_LAYERS, params, use_mps=True)
    
    # The MPS tensor network is stored in circ.psi
    mps_tn = circ.psi   # quimb MatrixProductState object

    dataset1.append({
        'index':    i,
        'params':   params,
        'circuit':  circ,
        'mps':      mps_tn,
    })

print(f"Dataset 1 ready: {len(dataset1)} samples")
print(f"Sample 0 MPS bond dimensions: {dataset1[0]['mps'].bond_sizes()}")
print(f"Sample 0 MPS tensor tags:     {[list(t.tags) for t in dataset1[0]['mps']]}")


# ─────────────────────────────────────────────────────────────────
# 3. DIAGRAM A — quimb native draw() of the MPS tensor network
# ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.patch.set_facecolor('#f0f0ee')

colors = ['#264653', '#2a9d8f', '#e76f51', '#f4a261']
site_colors = {f'I{k}': colors[k] for k in range(N_QUBITS)}

SITE_TAGS = ['I0', 'I1', 'I2', 'I3']          # quimb auto-tags sites as I0..In-1
PALETTE   = ['#264653', '#2a9d8f', '#e76f51', '#f4a261']


# Draw 3 different samples side-by-side
for col, sample_idx in enumerate([0, 33, 66]):
    ax = axes[col]
    mps = dataset1[sample_idx]['mps']

    # quimb draw() uses networkx layout internally
    mps.draw(
        ax=ax,
        color=SITE_TAGS,           # ✅ list of TAG NAMES
        custom_colors=PALETTE,     # ✅ list of colors for those tags
        node_color='#888888',      # ✅ single fallback string — NOT a list
        show_inds='bond-size',
        show_tags=True,
        node_size=2,
        node_alpha=0.92,
        edge_color='#333333',      # ✅ single string
        label_color='maroon',       # ✅ single string
        figsize=None,              # ✅ must be None when passing ax=
    )
    ax.set_facecolor('#f0f0ee')
    ax.set_title(f'Sample x_{sample_idx} — MPS Bond Dims: '
                 f'{dataset1[sample_idx]["mps"].bond_sizes()}',
                 fontsize=10, fontweight='bold', color='#1a1a2e', pad=8)

plt.suptitle('Dataset 1 — Ising Chain MPS Tensor Networks (quimb draw)\n'
             r'Each node = local tensor $A^{[k]}$, edges = contracted indices',
             fontsize=13, fontweight='bold', color='#1a1a2e', y=1.02)
plt.tight_layout()
plt.savefig('output/dataset1_quimb_mps.png', dpi=150,
            bbox_inches='tight', facecolor='#f0f0ee')
plt.close()
print("Saved: output/dataset1_quimb_mps.png")


# ─────────────────────────────────────────────────────────────────
# 4. DIAGRAM B — networkx graph of the MPS chain topology
# ─────────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.patch.set_facecolor('#f0f0ee')

# ── Left: Circuit-level networkx graph ──
ax = axes2[0]
ax.set_facecolor('#f0f0ee')
ax.set_title('Ising Circuit as Interaction Graph\n(nodes=qubits, edges=ZZ bonds)',
             fontsize=11, fontweight='bold', color='#1a1a2e')

G_circuit = nx.Graph()
for q in range(N_QUBITS):
    G_circuit.add_node(q, layer='qubit')
for bond in range(N_QUBITS - 1):
    G_circuit.add_edge(bond, bond + 1, weight=2.5)

pos = {q: (q, 0) for q in range(N_QUBITS)}
nx.draw_networkx_nodes(G_circuit, pos, ax=ax,
                       node_color=colors[:N_QUBITS],
                       node_size=1200, alpha=0.9)
nx.draw_networkx_labels(G_circuit, pos, ax=ax,
                        labels={q: f'$q_{q+1}$' for q in range(N_QUBITS)},
                        font_color='white', font_size=13, font_weight='bold')
nx.draw_networkx_edges(G_circuit, pos, ax=ax,
                       edge_color='#1a1a2e', width=3.5, alpha=0.8)
edge_labels = {(q, q+1): f'ZZ\nθ={dataset1[0]["params"]["zz"][0,q]:.2f}'
               for q in range(N_QUBITS - 1)}
nx.draw_networkx_edge_labels(G_circuit, pos, edge_labels, ax=ax,
                             font_size=8, font_color='#264653',
                             bbox=dict(boxstyle='round', fc='white', alpha=0.7))
ax.axis('off')

# ── Right: MPS tensor network as networkx graph ──
ax2 = axes2[1]
ax2.set_facecolor('#f0f0ee')
ax2.set_title('MPS as Tensor Network Graph (networkx)\n'
              '(circles=tensors, squares=physical indices)',
              fontsize=11, fontweight='bold', color='#1a1a2e')

G_mps = nx.Graph()
tensor_nodes = [f'A[{k}]' for k in range(N_QUBITS)]
phys_nodes   = [f'σ_{k+1}' for k in range(N_QUBITS)]

for k in range(N_QUBITS):
    G_mps.add_node(tensor_nodes[k], ntype='tensor')
    G_mps.add_node(phys_nodes[k],   ntype='physical')
    G_mps.add_edge(tensor_nodes[k], phys_nodes[k], etype='physical')

for k in range(N_QUBITS - 1):
    bd = dataset1[0]['mps'].bond_size(tensor_nodes[k] if False else k, k+1)
    G_mps.add_edge(tensor_nodes[k], tensor_nodes[k+1],
                   etype='bond', label=f'D={BOND_DIM}')

# Manual layout: tensors in a row, physical indices below
pos2 = {}
for k in range(N_QUBITS):
    pos2[tensor_nodes[k]] = (k * 1.5, 1.0)
    pos2[phys_nodes[k]]   = (k * 1.5, 0.0)

tensor_list = [n for n, d in G_mps.nodes(data=True) if d['ntype'] == 'tensor']
phys_list   = [n for n, d in G_mps.nodes(data=True) if d['ntype'] == 'physical']
bond_edges  = [(u, v) for u, v, d in G_mps.edges(data=True) if d['etype'] == 'bond']
phys_edges  = [(u, v) for u, v, d in G_mps.edges(data=True) if d['etype'] == 'physical']

nx.draw_networkx_nodes(G_mps, pos2, nodelist=tensor_list, ax=ax2,
                       node_color=colors[:N_QUBITS], node_size=1400,
                       node_shape='o', alpha=0.92)
nx.draw_networkx_nodes(G_mps, pos2, nodelist=phys_list, ax=ax2,
                       node_color='#dee2e6', node_size=700,
                       node_shape='s', alpha=0.9)
nx.draw_networkx_labels(G_mps, pos2, ax=ax2,
                        labels={n: n for n in tensor_list},
                        font_color='white', font_size=9, font_weight='bold')
nx.draw_networkx_labels(G_mps, pos2, ax=ax2,
                        labels={n: n for n in phys_list},
                        font_color='#495057', font_size=9)
nx.draw_networkx_edges(G_mps, pos2, edgelist=bond_edges, ax=ax2,
                       edge_color='#264653', width=3.0)
nx.draw_networkx_edges(G_mps, pos2, edgelist=phys_edges, ax=ax2,
                       edge_color='#adb5bd', width=1.8, style='dashed')
bond_edge_labels = {(tensor_nodes[k], tensor_nodes[k+1]): f'D={BOND_DIM}'
                    for k in range(N_QUBITS - 1)}
nx.draw_networkx_edge_labels(G_mps, pos2, bond_edge_labels, ax=ax2,
                             font_size=9, font_color='#264653',
                             bbox=dict(boxstyle='round', fc='white', alpha=0.8))

legend_elem = [
    mpatches.Patch(facecolor=colors[k], label=f'A[{k}] shape {dataset1[0]["mps"][k].shape}')
    for k in range(N_QUBITS)
] + [mpatches.Patch(facecolor='#dee2e6', label='Physical index σ_k (d=2)')]
ax2.legend(handles=legend_elem, loc='upper right', fontsize=8.5, framealpha=0.9)
ax2.axis('off')

plt.suptitle('Dataset 1 — Ising Chain Circuit & MPS Graph (networkx)\n'
             'Left: qubit interaction graph | Right: MPS tensor network',
             fontsize=13, fontweight='bold', color='#1a1a2e', y=1.02)
plt.tight_layout()
plt.savefig('output/dataset1_networkx_graph.png', dpi=150,
            bbox_inches='tight', facecolor='#f0f0ee')
plt.close()
print("Saved: output/dataset1_networkx_graph.png")


# ─────────────────────────────────────────────────────────────────
# 5. DIAGRAM C — quimb circuit diagram (gate-level view)
# ─────────────────────────────────────────────────────────────────

# Rebuild one sample as a plain Circuit (not MPS) so we can draw gates
circ_draw = make_ising_circuit_quimb(N_QUBITS, N_LAYERS,
                                      dataset1[0]['params'], use_mps=False)

fig3, ax3 = plt.subplots(1, 1, figsize=(12, 5))
fig3.patch.set_facecolor('#f0f0ee')

circ_draw.psi.draw(
    ax=ax3,
    show_inds='bond-size',
    show_tags=True,
    node_size=2,
    edge_color='#495057',
    label_color='maroon',
    font_size=8,
    figsize=None,
    color=['Set2'],
)
ax3.set_facecolor('#f0f0ee')
ax3.set_title('Full Circuit TN (quimb) — Sample 0, 4 qubits, 3 Ising layers\n'
              r'Nodes = gate tensors, edges = contracted qubit indices',
              fontsize=11, fontweight='bold', color='#1a1a2e')
plt.tight_layout()
plt.savefig('output/dataset1_circuit_tn.png', dpi=150,
            bbox_inches='tight', facecolor='#f0f0ee')
plt.close()
print("Saved: output/dataset1_circuit_tn.png")

print("\nAll diagrams generated successfully.")
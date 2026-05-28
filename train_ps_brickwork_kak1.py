"""
train_ps_brickwork.py — NFL Pipeline: P_S as MPS Brickwork Circuit
====================================================================
Trains a variational P_S unitary implemented as an MPS brickwork circuit.

TWO model classes are provided:
  MPSBrickworkPS     — ZYZ local + ZZ Ising entanglers (original)
  MPSBrickworkPSSU4  — ZYZ local + full SU(4) bonds via KAK1 decomposition

KAK1 decomposition (Tucci 2005, arXiv:quant-ph/0507171, Corollary 6):
  Any U ∈ SU(4) can be written EXACTLY as:
      U = (A1 ⊗ A0) · exp(i(kx·XX + ky·YY + kz·ZZ)) · (B1 ⊗ B0)
  where A1, A0, B1, B0 ∈ SU(2) (each parametrised as ZYZ, 3 params),
  and (kx, ky, kz) ∈ R^3 are the Weyl / canonical class parameters.

  Parameter count per SU(4) gate:
      Left local  A1⊗A0  :  3 + 3 = 6 params
      Weyl entangler      :  3 params  (kx, ky, kz)
      Right local B1⊗B0  :  3 + 3 = 6 params
      Total               : 15 params  = dim(su(4))  ← no redundancy

  For BOND_DIM = D, each bond hosts D *sequential* independent KAK1 gates,
  giving up to min(15D, 16) effective DOF per bond per layer.
  At D=2 a bond gate is already universal for U(4) (KAK theorem).

NFL paper loss (arXiv:2412.05674):
  L = (1/t) * sum_j ||P_S|x_j> - |phi_j>||^2 / ||x_j||^2
"""

import numpy as np
import os, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Make local module imports robust when running the script from any cwd.
sys.path.insert(0, os.path.dirname(__file__))
# Ensure all result artifacts can be written without manual directory setup.
os.makedirs("output", exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# USER HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════
DATASET_PATH  = "output/dataset_M4_D2_t6000_seq.npz"  # Dataset from M_dataset_agg.py.
T_TRAIN       = 5000                              # Number of training samples used (prefix slice).
N_EPOCHS      = 1000                              # Optimizer steps per layer-depth run.
LEARNING_RATE = 1e-3                              # Initial Adam learning rate.
LOG_EVERY     = 50                                # Print metrics every N epochs.
SEED          = 42                                # Reproducible initialization and optimization.
DEVICE        = "cuda"   # "auto" | "cuda" | "mps" | "cpu"

# ── MPS brickwork circuit hyperparameters ──────────────────────────
N_LAYERS  = list(range(3, 11))   # layer sweep L = 3 … 10
BOND_DIM  = 2                    # D: independent KAK1 gates stacked per bond
D_PHYS    = 2                    # physical dim per qubit (2 = qubit)

OUT_PLOT         = "output/training_loss_fidelity_mps_kak1_layers_3_10_seq.png"
OUT_SUMMARY_PLOT = "output/fidelity_summary_mps_kak1_layers_3_10_seq.png"
OUT_RESULTS      = "output/training_results_mps_kak1_layers_3_10_seq.npz"
# ══════════════════════════════════════════════════════════════════


def get_device(pref):
    """Resolve requested compute backend with explicit checks and safe fallback."""
    if not TORCH_AVAILABLE:
        # Keep script importable on systems without torch (training itself still needs torch).
        return "cpu_numpy"
    if pref == "cuda":
        # Fail loudly if the user requested CUDA but it is unavailable.
        assert torch.cuda.is_available(), "CUDA not available."
        return torch.device("cuda")
    elif pref == "mps":
        # Apple Metal backend (for Apple Silicon).
        assert torch.backends.mps.is_available(), "MPS not available."
        return torch.device("mps")
    elif pref == "cpu":
        return torch.device("cpu")
    else:
        # Auto mode: prefer CUDA, then MPS, then CPU.
        if torch.cuda.is_available():   return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")


# ══════════════════════════════════════════════════════════════════
# GATE PRIMITIVES
# ══════════════════════════════════════════════════════════════════

def rz_gate(angle):
    """2×2  Rz(angle) = diag(exp(-ia/2), exp(+ia/2))"""
    c = torch.cos(angle / 2)
    s = torch.sin(angle / 2)
    z = torch.zeros_like(c)
    row0 = torch.stack([torch.complex(c, -s), torch.complex(z, z)], dim=-1)
    row1 = torch.stack([torch.complex(z,  z), torch.complex(c, s)], dim=-1)
    return torch.stack([row0, row1], dim=-2)   # (2,2) complex64


def ry_gate(angle):
    """2×2  Ry(angle)"""
    c = torch.cos(angle / 2)
    s = torch.sin(angle / 2)
    z = torch.zeros_like(c)
    row0 = torch.stack([torch.complex(c, z), torch.complex(-s, z)], dim=-1)
    row1 = torch.stack([torch.complex(s, z), torch.complex( c, z)], dim=-1)
    return torch.stack([row0, row1], dim=-2)   # (2,2) complex64


def zyz_gate(alpha, beta, gamma):
    """2×2  ZYZ gate: Rz(gamma) @ Ry(beta) @ Rz(alpha)  — 3 real params, full SU(2)"""
    return rz_gate(gamma) @ ry_gate(beta) @ rz_gate(alpha)


def zz_entangler(theta, device):
    """
    4×4 ZZ Ising entangler: exp(-i * theta/2 * Z⊗Z)
    = diag(exp(-it/2), exp(+it/2), exp(+it/2), exp(-it/2))
    """
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(c)
    d = torch.stack([
        torch.complex(c, -s),   # |00>
        torch.complex(c,  s),   # |01>
        torch.complex(c,  s),   # |10>
        torch.complex(c, -s),   # |11>
    ])
    return torch.diag(d)   # (4,4) complex64


# ──────────────────────────────────────────────────────────────────
# KAK1 WEYL GENERATORS  (fixed constant tensors, built once per device)
# XX = σX⊗σX,  YY = σY⊗σY,  ZZ = σZ⊗σZ  in computational basis
# ──────────────────────────────────────────────────────────────────
_WEYL_CACHE = {}

def _weyl_generators(device):
    """Return (XX, YY, ZZ) as (4,4) complex64 tensors on `device`. Cached."""
    key = str(device)
    if key not in _WEYL_CACHE:
        XX = torch.tensor(
            [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]],
            dtype=torch.complex64, device=device)
        YY = torch.tensor(
            [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]],
            dtype=torch.complex64, device=device)
        ZZ = torch.diag(torch.tensor([1,-1,-1,1],
            dtype=torch.complex64, device=device))
        _WEYL_CACHE[key] = (XX, YY, ZZ)
    return _WEYL_CACHE[key]


def kak1_su4_gate(params_left, kxyz, params_right, device):
    """
    KAK1 SU(4) gate — Tucci (2005) Corollary 6.

    U = (A1⊗A0) · exp(i(kx·XX + ky·YY + kz·ZZ)) · (B1⊗B0)

    Parameters
    ----------
    params_left  : (6,) real  — [alpha1, beta1, gamma1, alpha0, beta0, gamma0]
                                ZYZ angles for A1 (qubit k) and A0 (qubit k+1)
    kxyz         : (3,) real  — Weyl canonical parameters [kx, ky, kz]
    params_right : (6,) real  — [alpha1, beta1, gamma1, alpha0, beta0, gamma0]
                                ZYZ angles for B1 (qubit k) and B0 (qubit k+1)
    device       : torch device

    Returns
    -------
    U : (4,4) complex64  ∈ U(4),  exactly 15 independent real DOF = dim(su(4))
    """
    # ── Left local block: A1 ⊗ A0 ────────────────────────────────
    A1 = zyz_gate(params_left[0], params_left[1], params_left[2])   # (2,2)
    A0 = zyz_gate(params_left[3], params_left[4], params_left[5])   # (2,2)
    L_block = torch.kron(A1, A0)                                     # (4,4)

    # ── Weyl entangler: exp(i*(kx·XX + ky·YY + kz·ZZ)) ──────────
    # Only 3 real parameters span the non-local content of any SU(4)
    XX, YY, ZZ = _weyl_generators(device)
    H_weyl = kxyz[0] * XX + kxyz[1] * YY + kxyz[2] * ZZ            # (4,4) Hermitian
    W_gate = torch.linalg.matrix_exp(1j * H_weyl)                   # (4,4)

    # ── Right local block: B1 ⊗ B0 ───────────────────────────────
    B1 = zyz_gate(params_right[0], params_right[1], params_right[2])
    B0 = zyz_gate(params_right[3], params_right[4], params_right[5])
    R_block = torch.kron(B1, B0)                                     # (4,4)

    return L_block @ W_gate @ R_block   # (4,4)


def su4_gate(r_sum, i_sum):
    """
    Full SU(4) bond gate via matrix exponential.

    r_sum, i_sum : (16,) real tensors (summed over D modes already)
    Builds A = r_sum + i·i_sum  as 4×4 complex,
    then H = (A − A†) / 2i  (Hermitian),
    then G = expm(i·H)       (unitary ∈ SU(4) up to global phase).
    """
    A = torch.complex(r_sum, i_sum).reshape(4, 4)
    H = (A - A.conj().T) / (2j)          # Hermitian 4×4
    return torch.linalg.matrix_exp(1j * H)  # (4,4) unitary

# ══════════════════════════════════════════════════════════════════
# EMBED GATES INTO n-QUBIT SPACE
# ══════════════════════════════════════════════════════════════════

def embed_1q(gate_2x2, k, n, device):
    """I ⊗ … ⊗ gate_k ⊗ … ⊗ I  →  (2^n, 2^n)"""
    ops = [torch.eye(2, dtype=torch.complex64, device=device)] * n
    ops[k] = gate_2x2
    result = ops[0]
    for op in ops[1:]:
        result = torch.kron(result, op)
    return result


def embed_2q(gate_4x4, k, n, device):
    """I_left ⊗ gate_(k,k+1) ⊗ I_right  →  (2^n, 2^n)"""
    assert k + 1 < n, f"Bond ({k},{k+1}) out of range for n={n}"
    left_dim  = 2 ** k
    right_dim = 2 ** (n - k - 2)
    I_left  = torch.eye(left_dim,  dtype=torch.complex64, device=device)
    I_right = torch.eye(right_dim, dtype=torch.complex64, device=device)
    return torch.kron(torch.kron(I_left, gate_4x4), I_right)


# ══════════════════════════════════════════════════════════════════
# MODEL 1 — ZYZ + ZZ Ising  (original, kept for comparison)
# ══════════════════════════════════════════════════════════════════

class MPSBrickworkPS(nn.Module):
    """
    MPS brickwork P_S: ZYZ local gates + ZZ Ising entanglers.

    Per layer l:
      (a) ZYZ on every qubit  — 3 params per qubit per layer
      (b) ZZ entangler on each bond (k, k+1), D modes per bond
          — 1 param per mode per bond per layer

    Total: L*(n*3 + (n-1)*D)  params
    """
    def __init__(self, n_qubits, n_layers, bond_dim=2, d_phys=2):
        super().__init__()
        # Persist architecture hyperparameters for circuit assembly and reporting.
        self.n, self.L, self.D, self.d = n_qubits, n_layers, bond_dim, d_phys
        # Global Hilbert-space dimension (2^n for qubits).
        self.dim = d_phys ** n_qubits
        # Local SU(2) rotations per qubit/layer: (alpha, beta, gamma) for ZYZ.
        self.zyz = nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)
        # One ZZ angle per bond/layer/mode.
        self.zz  = nn.Parameter(torch.randn(n_layers, n_qubits - 1, bond_dim) * 0.1)

    def get_W(self):
        dev = self.zyz.device
        n, L, D, dim = self.n, self.L, self.D, self.dim
        # Start from identity and left-multiply each gate in circuit order.
        W = torch.eye(dim, dtype=torch.complex64, device=dev)
        for l in range(L):
            # Step (a): local single-qubit rotations.
            for k in range(n):
                G1 = zyz_gate(self.zyz[l,k,0], self.zyz[l,k,1], self.zyz[l,k,2])
                W  = embed_1q(G1, k, n, dev) @ W
            # Step (b): nearest-neighbor entanglers, possibly stacked by D modes.
            for k in range(n - 1):
                for d in range(D):
                    G2 = zz_entangler(self.zz[l,k,d], dev)
                    W  = embed_2q(G2, k, n, dev) @ W
        return W

    def forward(self, X):
        # Row-stored ket amplitudes follow Phi = (W @ X.T).T = X @ W.T.
        return X @ self.get_W().T

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def architecture_str(self):
        n, L, D = self.n, self.L, self.D
        n_zyz = L * n * 3
        n_zz  = L * (n - 1) * D
        return (
            f"MPS Brickwork P_S (ZZ Ising) | n={n} | L={L} | D={D}\n"
            f"  ZYZ local  : {L}×{n}×3 = {n_zyz} params\n"
            f"  ZZ bonds   : {L}×{n-1}×D={D} = {n_zz} params\n"
            f"  Total      : {n_zyz + n_zz}"
        )


# ══════════════════════════════════════════════════════════════════
# MODEL 2 — ZYZ local + KAK1 SU(4) bonds  (CORRECTED)
# ══════════════════════════════════════════════════════════════════

class MPSBrickworkPSSU4(nn.Module):
    """
    MPS brickwork P_S with full SU(4) bond gates via KAK1 decomposition.

    Per layer l:
      (a) ZYZ local gates on every qubit k  [REMOVED — already inside KAK1 left block]
      (b) KAK1 SU(4) bond gate on each bond (k, k+1), D independent gates stacked:

            G_bond = G_D · G_{D-1} · … · G_1     (sequential composition)

          Each G_d = (A1_d⊗A0_d) · exp(i(kx·XX+ky·YY+kz·ZZ)) · (B1_d⊗B0_d)
          with exactly 15 real parameters:
              - 6 for left local block (ZYZ on each of 2 qubits)
              - 3 for Weyl entangler  (kx, ky, kz)
              - 6 for right local block (ZYZ on each of 2 qubits)

    NOTE: The standalone ZYZ step (a) from the original code is REMOVED.
    The left local block A1⊗A0 of the FIRST KAK1 gate on each bond already
    provides a full SU(2)⊗SU(2) rotation, making a separate step (a) redundant
    (two consecutive ZYZ gates on the same qubit collapse into one ZYZ).

    Parameter storage (per bond, per layer, per KAK1 mode d):
        self.kak_left  [l, k, d, 6]   — ZYZ angles for A1 (3) and A0 (3)
        self.kak_weyl  [l, k, d, 3]   — Weyl params (kx, ky, kz)
        self.kak_right [l, k, d, 6]   — ZYZ angles for B1 (3) and B0 (3)

    Total per bond per layer per mode: 15 params (= dim su(4), no redundancy)
    Total params: L * (n-1) * D * 15

    Expressivity:
        D=1 → 15 DOF per bond → exactly one SU(4) gate
        D=2 → 30 DOF per bond → universal for U(4)  (KAK theorem: 2 gates suffice)
        D≥3 → additional redundancy, helps escape local minima
    """
    def __init__(self, n_qubits, n_layers, bond_dim=2, d_phys=2):
        super().__init__()
        # Core architecture knobs.
        self.n, self.L, self.D, self.d = n_qubits, n_layers, bond_dim, d_phys
        self.dim = d_phys ** n_qubits
        n, L, D = n_qubits, n_layers, bond_dim

        # KAK1 parameters: 15 per (bond, mode) = 6 + 3 + 6
        self.kak_left  = nn.Parameter(torch.randn(L, n-1, D, 6) * 0.1)   # A1,A0 ZYZ
        self.kak_weyl  = nn.Parameter(torch.randn(L, n-1, D, 3) * 0.1)   # kx,ky,kz
        self.kak_right = nn.Parameter(torch.randn(L, n-1, D, 6) * 0.1)   # B1,B0 ZYZ

    def get_W(self):
        dev = self.kak_left.device
        n, L, D, dim = self.n, self.L, self.D, self.dim
        # Accumulate full-system unitary from identity.
        W = torch.eye(dim, dtype=torch.complex64, device=dev)

        for l in range(L):
            # ── KAK1 bond layer ───────────────────────────────────
            # No separate ZYZ step (a) needed: the left block of each
            # KAK1 gate already provides a full SU(2)⊗SU(2) rotation.
            for k in range(n - 1):
                # Stack D independent KAK1 gates sequentially on bond (k,k+1)
                G_bond = torch.eye(4, dtype=torch.complex64, device=dev)
                for d in range(D):
                    Gd = kak1_su4_gate(
                        self.kak_left [l, k, d],   # (6,)
                        self.kak_weyl [l, k, d],   # (3,)
                        self.kak_right[l, k, d],   # (6,)
                        dev
                    )
                    G_bond = Gd @ G_bond   # compose: last mode applied first
                # Lift 2-qubit bond unitary to full n-qubit space and compose.
                W = embed_2q(G_bond, k, n, dev) @ W

        return W   # (dim, dim) complex64

    def forward(self, X):
        """Batch: X (t, dim) → P_S|x_j⟩ (t, dim)"""
        return X @ self.get_W().T

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def architecture_str(self):
        n, L, D = self.n, self.L, self.D
        n_kak = L * (n-1) * D * 15
        return (
            f"MPS Brickwork P_S (KAK1 SU(4)) | n={n} | L={L} | D={D}\n"
            f"  KAK1 bonds : {L}×{n-1}×D={D}×15 = {n_kak} params\n"
            f"    per gate : 6 (left ZYZ) + 3 (Weyl) + 6 (right ZYZ) = 15\n"
            f"  Total      : {n_kak}  (U({self.dim}) needs {self.dim**2} DOF)"
        )


# ══════════════════════════════════════════════════════════════════
# LOSS AND FIDELITY
# ══════════════════════════════════════════════════════════════════

def nfl_loss_torch(Psi, Phi, X_norms_sq):
    """NFL paper loss: (1/t) Σ ‖P_S|x⟩−|ϕ⟩‖² / ‖x‖²"""
    # Per-sample complex error vector in Hilbert space.
    diff     = Psi - Phi
    # Squared L2 norm of complex vector = sum(real^2 + imag^2).
    sq_norm  = (diff.real**2 + diff.imag**2).sum(dim=-1)
    # Normalize each sample by input norm, then average over batch.
    return (sq_norm / X_norms_sq).mean()


def state_fidelity_torch(Psi, Phi):
    """F_j = |⟨ϕ|P_S|x⟩|² / (‖ϕ‖²·‖P_Sx‖²)  —  (t,) on device"""
    # Numerator: squared overlap magnitude.
    num   = (Phi.conj() * Psi).sum(dim=-1).abs() ** 2
    # Denominator: product of state norms.
    denom = (Phi.abs()**2).sum(dim=-1) * (Psi.abs()**2).sum(dim=-1)
    # Clamp prevents division-by-zero from pathological near-zero vectors.
    return num / denom.clamp(min=1e-30)


def process_fidelity(W_np, M_np, dim):
    """F_proc = |Tr(M†W)|² / dim²"""
    # Global channel-level overlap between learned unitary W and target M.
    return (abs(np.trace(M_np.conj().T @ W_np)) / dim) ** 2


def normalize_training_pairs(X_np, Phi_np, M_np, tol=1e-8):
    """
    Return row-normalized training pairs while preserving Phi = M @ X.

    Older generated datasets stored the raw MPS contraction output with
    variable norms. Since M is unitary, Phi has the same row norm as X; scale
    both arrays by that shared norm so stale archives remain usable.
    """
    X = np.asarray(X_np, dtype=np.complex128).copy()
    Phi = np.asarray(Phi_np, dtype=np.complex128).copy()
    M = np.asarray(M_np, dtype=np.complex128)

    if X.ndim != 2 or Phi.shape != X.shape:
        raise ValueError(f"Expected X and Phi with matching 2D shapes, got {X.shape} and {Phi.shape}")

    phi_expected = (M @ X.T).T
    raw_err = np.max(np.abs(phi_expected - Phi))
    if raw_err >= tol:
        raise ValueError(f"Dataset mismatch before normalization: max|Phi - M@X| = {raw_err:.2e}")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms <= 1e-14):
        raise ValueError("Cannot normalize dataset containing near-zero input states.")

    X /= norms
    Phi /= norms

    norm_err = np.max(np.abs((M @ X.T).T - Phi))
    if norm_err >= tol:
        raise ValueError(f"Dataset mismatch after normalization: max|Phi - M@X| = {norm_err:.2e}")

    return X, Phi


# ══════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════

def train_torch(X_np, Phi_np, n_qubits, n_layers, bond_dim, d_phys,
                n_epochs, lr, log_every, seed, device):

    # Ensure reproducible model initialization and optimizer trajectory.
    torch.manual_seed(seed)
    model = MPSBrickworkPSSU4(n_qubits, n_layers, bond_dim, d_phys).to(device)

    print()
    print("=" * 72)
    print("  Training P_S | MPS Brickwork (KAK1 SU(4) bonds) | NFL Loss")
    print("=" * 72)
    print(f"  {model.architecture_str()}")
    print(f"  dim={model.dim}  t={len(X_np)}  epochs={n_epochs}  lr={lr}")
    print("-" * 72)
    print(f"  Backend : PyTorch on {device}")
    if str(device) == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Adam handles heterogeneous parameter scales well for variational circuits.
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine decay gradually reduces step size for late-stage refinement.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.01)

    # Move dataset to device once to avoid host/device copies each epoch.
    Xt    = torch.from_numpy(X_np.astype(np.complex64)).to(device)
    Pt    = torch.from_numpy(Phi_np.astype(np.complex64)).to(device)
    # Precompute ||x_j||^2 used in the normalized NFL objective.
    X_nsq = (Xt.real**2 + Xt.imag**2).sum(dim=-1)

    loss_h, fid_h = [], []
    F_per_last    = None
    t0 = time.time()

    for ep in range(n_epochs):
        # Standard training step: clear grads -> forward -> loss -> backward -> update.
        opt.zero_grad()
        Psi  = model(Xt)
        loss = nfl_loss_torch(Psi, Pt, X_nsq)
        loss.backward()
        opt.step()
        sched.step()

        with torch.no_grad():
            # Report metrics for the parameters after the optimizer update.
            Psi_eval = model(Xt)
            loss_eval = nfl_loss_torch(Psi_eval, Pt, X_nsq)
            fids_ep  = state_fidelity_torch(Psi_eval, Pt)
            fid_mean = fids_ep.mean().item()

        loss_h.append(loss_eval.item())
        fid_h.append(fid_mean)

        if ep == n_epochs - 1:
            # Persist per-sample fidelities from the final epoch for summary stats.
            F_per_last = fids_ep.cpu().numpy().astype(np.float64)

        if ep % log_every == 0 or ep == n_epochs - 1:
            print(f"  ep {ep:>5d}/{n_epochs}  loss={loss_eval.item():.6f}"
                  f"  F̄={fid_mean:.6f}  [{time.time()-t0:.1f}s]")

    model.eval()
    with torch.no_grad():
        # Export learned full-system unitary in high precision for diagnostics/metrics.
        W_out = model.get_W().cpu().numpy().astype(np.complex128)

    unit_err = np.max(np.abs(W_out.conj().T @ W_out - np.eye(model.dim)))
    print(f"  W unitarity check : max|W†W - I| = {unit_err:.2e}")

    return W_out, np.array(loss_h), np.array(fid_h), F_per_last, model


# ══════════════════════════════════════════════════════════════════
# SWEEP SUMMARY + PLOTS
# ══════════════════════════════════════════════════════════════════

def summarize_run(n_layers, loss_h, fid_h, F_per, F_proc, n_params):
    # Canonical per-depth record used for both printed and plotted summaries.
    return dict(
        n_layers=int(n_layers), loss_h=loss_h, fid_h=fid_h, F_per=F_per,
        F_process=float(F_proc), final_loss=float(loss_h[-1]),
        final_fid=float(fid_h[-1]),
        F_std=float(F_per.std()), F_min=float(F_per.min()),
        F_max=float(F_per.max()), n_params=int(n_params),
    )


def print_sweep_table(rows):
    print()
    print("╔" + "═"*100 + "╗")
    print("║" + " Layer Sweep Summary — MPS Brickwork (KAK1 SU(4) bonds) ".center(100) + "║")
    print("╟" + "─"*100 + "╢")
    hdr = "  L | Final loss | Mean F̄  |  Std F  |  Min F  |  Max F  | Process F | Params"
    print(f"║ {hdr:<98} ║")
    print("╟" + "─"*100 + "╢")
    for row in sorted(rows, key=lambda x: x["n_layers"]):
        line = (
            f"{row['n_layers']:>3d} | "
            f"{row['final_loss']:>10.6f} | "
            f"{row['final_fid']:>7.5f} | "
            f"{row['F_std']:>7.5f} | "
            f"{row['F_min']:>7.5f} | "
            f"{row['F_max']:>7.5f} | "
            f"{row['F_process']:>9.6f} | "
            f"{row['n_params']:>6d}"
        )
        print(f"║  {line:<97} ║")
    print("╚" + "═"*100 + "╝")


def plot_sweep_histories(rows, t, dim, ep, bond_dim, device, path):
    # Shared style tokens for publication-friendly, consistent visual output.
    BG = "#f7f6f2"; DARK = "#264653"; MUTED = "#6c757d"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
    epochs = np.arange(1, ep + 1)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(rows)))

    for ax in (ax1, ax2):
        ax.set_facecolor("#ffffff")
        for side in ["top", "right"]:   ax.spines[side].set_visible(False)
        for side in ["left", "bottom"]: ax.spines[side].set_color("#dcd9d5")
        ax.tick_params(colors=MUTED, labelsize=10)

    # Plot one curve per tested layer depth L.
    for c, row in zip(colors, sorted(rows, key=lambda x: x["n_layers"])):
        label = f"L={row['n_layers']}"
        ax1.plot(epochs, row["loss_h"], lw=1.8, color=c, label=label)
        ax2.plot(epochs, row["fid_h"],  lw=1.8, color=c, label=label)

    ax1.set_title("Training Loss vs Epoch", fontsize=12, fontweight="bold", color=DARK)
    ax1.set_xlabel("Epoch", color=MUTED, fontsize=11)
    ax1.set_ylabel("(1/t)Σ‖PS|x⟩−|φ⟩‖²/⟨x|x⟩", color=MUTED, fontsize=10)
    ax1.legend(frameon=False, fontsize=9, ncol=2)

    ax2.axhline(1.0, color=DARK, lw=1.0, ls=":", alpha=0.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Mean State Fidelity F̄ vs Epoch", fontsize=12, fontweight="bold", color=DARK)
    ax2.set_xlabel("Epoch", color=MUTED, fontsize=11)
    ax2.set_ylabel("F̄ = (1/t)Σ|⟨φ|PS|x⟩|²", color=MUTED, fontsize=10)
    ax2.legend(frameon=False, fontsize=9, ncol=2)

    fig.suptitle(
        f"MPS Brickwork KAK1 Layer Sweep (L={min(r['n_layers'] for r in rows)}.."
        f"{max(r['n_layers'] for r in rows)})  n={int(np.log2(dim))}  D={bond_dim}  "
        f"epochs={ep}  t={t}  device={device}",
        fontsize=11, color=DARK, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Sweep plot saved  : {path}")


def plot_fidelity_summary_table(rows, t, dim, ep, bond_dim, device, path):
    BG = "#f7f6f2"; DARK = "#264653"; TEAL = "#2a9d8f"
    sorted_rows = sorted(rows, key=lambda x: x["n_layers"])
    best_layer  = max(sorted_rows, key=lambda x: x["final_fid"])["n_layers"]

    cell_text = []
    # Build table rows in the same order as layer sweep.
    for row in sorted_rows:
        cell_text.append([
            str(row["n_layers"]),
            f"{row['final_loss']:.6f}",
            f"{row['final_fid']:.6f}",
            f"{row['F_std']:.6f}",
            f"{row['F_min']:.6f}",
            f"{row['F_max']:.6f}",
            f"{row['F_process']:.6f}",
            str(row["n_params"]),
        ])

    fig_h = max(4.8, 1.6 + 0.45 * len(cell_text))
    fig, ax = plt.subplots(figsize=(13.5, fig_h), facecolor=BG)
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=["L", "Final loss", "Mean F̄", "Std F", "Min F",
                   "Max F", "Process F", "Params"],
        cellLoc="center", loc="center",
        colWidths=[0.07, 0.14, 0.12, 0.10, 0.10, 0.10, 0.12, 0.10],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.3)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dcd9d5")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color="white", fontweight="bold")
            continue
        row_layer = int(cell_text[r - 1][0])
        if row_layer == best_layer:
            cell.set_facecolor("#e8f4f3")
            cell.set_text_props(color=TEAL, fontweight="bold")
        else:
            cell.set_facecolor("#ffffff" if r % 2 else "#f9f8f5")
            cell.set_text_props(color=DARK)

    ax.set_title(
        f"Fidelity Summary by Layer Depth — KAK1 SU(4)  (best Mean F̄ at L={best_layer})\n"
        f"n={int(np.log2(dim))}  D={bond_dim}  epochs={ep}  t={t}  device={device}",
        fontsize=11, color=DARK, fontweight="bold", pad=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Summary table plot: {path}")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Resolve hardware backend according to DEVICE preference.
    device = get_device(DEVICE)
    print(f"\nDevice: {device}")

    # Load supervised training pairs and target unitary from dataset archive.
    print(f"Loading : {DATASET_PATH}")
    arch     = np.load(DATASET_PATH)
    X_all    = arch["X"]
    Phi_all  = arch["Phi"]
    M        = arch["M"]
    dim      = int(arch["dim"])
    n_qubits = int(arch["n_qubits"])

    assert T_TRAIN <= len(X_all), (
        f"T_TRAIN={T_TRAIN} > available {len(X_all)}. "
        f"Rerun generate_groundtruth.py with t_states >= {T_TRAIN}.")

    # Use the first T_TRAIN examples as training split (deterministic slicing).
    X_train   = X_all[:T_TRAIN]
    Phi_train = Phi_all[:T_TRAIN]
    X_train, Phi_train = normalize_training_pairs(X_train, Phi_train, M)

    # Quick sanity checks on normalization and dataset consistency.
    x_norms = np.linalg.norm(X_train, axis=1)
    print(f"  X shape   : {X_train.shape}  ‖x‖ range [{x_norms.min():.4f}, {x_norms.max():.4f}]")
    print(f"  Phi shape : {Phi_train.shape}")
    print(f"  M shape   : {M.shape}")

    Phi_check = (M @ X_train.T).T
    err = np.max(np.abs(Phi_check - Phi_train))
    print(f"  Phi sanity max|Phi - M@X| = {err:.2e} {'✓' if err < 1e-8 else '⚠ MISMATCH'}")

    # Normalize and validate layer sweep values from configuration.
    layer_values = sorted({int(l) for l in N_LAYERS})
    if not layer_values:
        raise ValueError("N_LAYERS must contain at least one integer depth.")
    if min(layer_values) < 1:
        raise ValueError(f"Invalid N_LAYERS={layer_values}; all values must be >= 1.")
    print(f"\nLayer sweep: {layer_values}")

    # Per-layer containers for reporting and result serialization.
    sweep_rows  = []
    W_per_layer = []

    for n_layers in layer_values:
        print(f"\n>>> Training with n_layers={n_layers}")
        W_final, loss_h, fid_h, F_per, model = train_torch(
            X_train, Phi_train,
            n_qubits  = n_qubits,
            n_layers  = n_layers,
            bond_dim  = BOND_DIM,
            d_phys    = D_PHYS,
            n_epochs  = N_EPOCHS,
            lr        = LEARNING_RATE,
            log_every = LOG_EVERY,
            seed      = SEED,
            device    = device,
        )
        F_proc = process_fidelity(W_final, M, dim)
        # Collect all metrics needed for plots, table, and final best-layer pick.
        row    = summarize_run(n_layers, loss_h, fid_h, F_per, F_proc, model.param_count())
        sweep_rows.append(row)
        W_per_layer.append(W_final)
        print(f"  L={n_layers}: loss={row['final_loss']:.6f}  "
              f"mean F̄={row['final_fid']:.6f}  process F={row['F_process']:.6f}")

    # Render cross-depth visual summaries.
    print_sweep_table(sweep_rows)
    plot_sweep_histories(sweep_rows, T_TRAIN, dim, N_EPOCHS, BOND_DIM, device, OUT_PLOT)
    plot_fidelity_summary_table(
        sweep_rows, T_TRAIN, dim, N_EPOCHS, BOND_DIM, device, OUT_SUMMARY_PLOT)

    # Compact arrays for saving and downstream analysis.
    layers_np  = np.array([r["n_layers"]   for r in sweep_rows], dtype=np.int64)
    final_loss = np.array([r["final_loss"] for r in sweep_rows], dtype=np.float64)
    final_fid  = np.array([r["final_fid"]  for r in sweep_rows], dtype=np.float64)
    n_params   = np.array([r["n_params"]   for r in sweep_rows], dtype=np.int64)

    np.savez(
        OUT_RESULTS,
        layers             = layers_np,
        W_per_layer        = np.stack(W_per_layer, axis=0),
        loss_histories     = np.stack([r["loss_h"]  for r in sweep_rows], axis=0),
        fid_histories      = np.stack([r["fid_h"]   for r in sweep_rows], axis=0),
        F_per_sample_last  = np.stack([r["F_per"]   for r in sweep_rows], axis=0),
        F_process          = np.array([r["F_process"] for r in sweep_rows], dtype=np.float64),
        final_loss         = final_loss,
        final_fid          = final_fid,
        bond_dim           = np.array(BOND_DIM),
        n_params           = n_params,
        t_train            = np.array(T_TRAIN),
        dim                = np.array(dim),
        n_epochs           = np.array(N_EPOCHS),
    )
    print(f"  Results saved : {OUT_RESULTS}")

    # Report the depth that maximizes final mean state fidelity.
    best_idx = int(np.argmax(final_fid))
    best_row = sweep_rows[best_idx]
    print(f"\n── Best Layer Summary ──────────────────────────────")
    print(f"   Best n_layers    = {best_row['n_layers']}")
    print(f"   Mean F̄          = {best_row['final_fid']:.6f}")
    print(f"   Process fidelity = {best_row['F_process']:.6f}")
    print(f"   Final loss       = {best_row['final_loss']:.6f}")
    print(f"   bond_dim D       = {BOND_DIM}")
    print(f"   Total params     = {best_row['n_params']}")
    print(f"───────────────────────────────────────────────────")

"""
train_mps.py — NFL Pipeline Step 4
====================================
Trains variational unitary W (MPS ansatz Psi_PS) on dataset (X, Phi).

NFL paper loss  (arXiv:2412.05674, Eq. before Eq.1):
    L = (1/t) * sum_j  | (M|x_j> - W|x_j>) / <x_j|x_j> |^2
      = (1/t) * sum_j  ||W|x_j> - |phi_j>||^2 / <x_j|x_j>

State fidelity (simple overlap of predicted vs target state):
    F_j = |<phi_j | W|x_j>|^2 / ( ||phi_j||^2 * ||W|x_j>||^2 )

where |phi_j> = M|x_j>  (ground-truth label, unnormalised)
      W|x_j>             (predicted output, unnormalised)

Both numerator and denominator use the RAW (unnormalised) states so that
the fidelity is purely a cosine-squared angle between the two vectors,
independent of their norms. This is the correct quantum state fidelity.

Run order:
    python build_pool_dataset.py   # generates pool + dataset
    python generate_groundtruth.py # generates M + labels Phi
    python train_mps.py            # this file

Device: auto -> CUDA -> MPS (Apple) -> CPU
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
    print("PyTorch not found — falling back to NumPy Riemannian GD")

from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("output", exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  USER PARAMETERS  — edit here
# ══════════════════════════════════════════════════════════════════
DATASET_PATH  = "output/dataset_M4_D2_t6000.npz"
T_TRAIN       = 5000       # training set size (must be <= t_states in dataset)
N_EPOCHS      = 1000       # gradient steps
LEARNING_RATE = 1e-3       # Adam learning rate
LOG_EVERY     = 50         # print interval (epochs)
SEED          = 42
DEVICE        = "cuda"     # "auto" | "cuda" | "mps" | "cpu"
OUT_PLOT      = "output/training_fidelity.png"
OUT_RESULTS   = "output/training_results.npz"
# ══════════════════════════════════════════════════════════════════


# ─── Device selection ──────────────────────────────────────────────
def get_device(pref: str):
    if not TORCH_AVAILABLE:
        return "cpu_numpy"
    if pref == "cuda":
        assert torch.cuda.is_available(), \
            "CUDA requested but not available. Use DEVICE='cpu'."
        return torch.device("cuda")
    elif pref == "mps":
        assert torch.backends.mps.is_available(), "MPS not available."
        return torch.device("mps")
    elif pref == "cpu":
        return torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():            return torch.device("cuda")
        if torch.backends.mps.is_available():    return torch.device("mps")
        return torch.device("cpu")


# ─── Loss: NFL paper Eq. (unnorm MSE) ─────────────────────────────
# L = (1/t) sum_j  ||W|x_j> - |phi_j>||^2 / <x_j|x_j>
#
# IMPORTANT: we do NOT pre-normalise X or Phi.
# The /‖x_j‖² term handles the unnormalised MPS states exactly as
# the paper writes it.  This ensures the loss landscape is correct.

def nfl_loss_torch(Psi, Phi, X_norms_sq):
    """
    Psi          : (t, dim) complex  — W|x_j>
    Phi          : (t, dim) complex  — M|x_j>  (ground truth)
    X_norms_sq   : (t,)     real     — <x_j|x_j>  (input norms squared)

    Returns scalar loss.
    """
    diff    = Psi - Phi                                      # (t, dim)
    sq_norm = (diff.real**2 + diff.imag**2).sum(dim=-1)      # (t,)
    return (sq_norm / X_norms_sq).mean()                     # scalar


# ─── Fidelity: simple state fidelity ──────────────────────────────
# F_j = |<phi_j | W|x_j>|^2 / ( ||phi_j||^2 * ||W|x_j>||^2 )
#
# This is purely the cosine-squared angle between the two vectors.
# It equals 1 iff W|x_j> is parallel to phi_j  (correct up to phase).
# It is independent of the norms, so unnormalised states are fine.

def state_fidelity_torch(Psi, Phi):
    """
    Psi  : (t, dim) complex  — W|x_j>
    Phi  : (t, dim) complex  — M|x_j>

    Returns F_j  shape (t,)  with F_j in [0, 1].
    """
    num   = (Phi.conj() * Psi).sum(dim=-1).abs()**2    # |<phi|Psi>|^2
    denom = (Phi.abs()**2).sum(dim=-1) * (Psi.abs()**2).sum(dim=-1)
    return num / denom.clamp(min=1e-30)                # (t,)

def state_fidelity_np(Psi_np, Phi_np):
    """NumPy version for final evaluation."""
    num   = np.abs(np.einsum("ij,ij->i", Phi_np.conj(), Psi_np))**2
    denom = (np.abs(Phi_np)**2).sum(axis=1) * (np.abs(Psi_np)**2).sum(axis=1)
    return num / np.where(denom > 0, denom, 1.0)


# ─── Variational Unitary model (PyTorch) ──────────────────────────
class VarUnitary(nn.Module):
    """
    W = expm(i * H(theta))  where  H = (A - A†) / 2i  is Hermitian.
    Guarantees W†W = I for any theta.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim    = dim
        self.A_real = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.A_imag = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def get_W(self) -> torch.Tensor:
        A  = torch.complex(self.A_real, self.A_imag)   # (dim,dim) complex
        H  = (A - A.conj().T) / 2j                     # Hermitian
        W  = torch.linalg.matrix_exp(1j * H)           # unitary
        return W

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Apply W to batch: X (t,dim) -> W|x_j> (t,dim)."""
        W = self.get_W()
        return X @ W.T.conj()


# ─── Riemannian GD on U(dim) — NumPy fallback ─────────────────────
def nfl_loss_np(W, X, Phi):
    """
    L = (1/t) sum_j  ||W|x_j> - phi_j||^2 / <x_j|x_j>
    Also returns Euclidean gradient dL/dW* and per-sample fidelities.
    """
    t      = len(X)
    Psi    = (W @ X.T).T                                     # (t,dim)
    diff   = Psi - Phi                                       # (t,dim)
    xnorms2= np.sum(np.abs(X)**2, axis=1)                   # (t,)
    loss   = float(np.mean(np.sum(np.abs(diff)**2, axis=1) / xnorms2))
    # Euclidean gradient dL/dW  (∂L/∂W*)
    # d||W|x>-phi||^2/dW* = diff * x†  -> batched outer product / norm
    eg     = (1/t) * np.einsum("ij,ik->jk", diff / xnorms2[:, None], X.conj())
    fids   = state_fidelity_np(Psi, Phi)
    return loss, eg, fids

def riemannian_step(W, egrad, lr):
    """
    Riemannian gradient step on U(dim):
      skew  = W†·eg - eg†·W   (skew-Hermitian tangent direction)
      W_new = W·(I - lr/2·skew)
      retract via SVD: W_new = U·Vh
    """
    G     = W.conj().T @ egrad
    skew  = G - G.conj().T
    W_new = W @ (np.eye(W.shape[0]) - (lr / 2) * skew)
    U, _, Vh = np.linalg.svd(W_new)
    return U @ Vh


# ─── Training: PyTorch (CUDA/MPS/CPU) ─────────────────────────────
def train_torch(X_np, Phi_np, dim, n_epochs, lr, log_every, seed, device):
    torch.manual_seed(seed)
    model  = VarUnitary(dim).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                 opt, T_max=n_epochs, eta_min=lr * 0.01)

    # Send raw (unnormalised) tensors to device
    Xt     = torch.from_numpy(X_np.astype(np.complex64)).to(device)
    Pt     = torch.from_numpy(Phi_np.astype(np.complex64)).to(device)
    # Pre-compute input norms squared on device — stays fixed
    X_nsq  = (Xt.real**2 + Xt.imag**2).sum(dim=-1)          # (t,)

    print(f"  Backend      : PyTorch on {device}")
    if str(device) == "cuda":
        print(f"  GPU          : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM         : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    loss_h, fid_h = [], []
    t0 = time.time()

    for ep in range(n_epochs):
        opt.zero_grad()
        Psi  = model(Xt)                              # W|x_j>
        loss = nfl_loss_torch(Psi, Pt, X_nsq)        # NFL paper loss
        loss.backward()
        opt.step()
        sched.step()

        with torch.no_grad():
            fids = state_fidelity_torch(Psi.detach(), Pt)
            fid  = fids.mean().item()

        loss_h.append(loss.item())
        fid_h.append(fid)

        if ep % log_every == 0 or ep == n_epochs - 1:
            print(f"  ep {ep:>5d}/{n_epochs}  "
                  f"loss={loss.item():.6f}  "
                  f"F̄={fid:.6f}  "
                  f"[{time.time()-t0:.1f}s]")

    model.eval()
    with torch.no_grad():
        W_out = model.get_W().cpu().numpy().astype(np.complex128)

    return W_out, np.array(loss_h), np.array(fid_h)


# ─── Training: NumPy Riemannian GD fallback ───────────────────────
def train_numpy(X_np, Phi_np, dim, n_epochs, lr, log_every, seed):
    rng      = np.random.default_rng(seed)
    Z        = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    W, _     = np.linalg.qr(Z)

    print(f"  Backend      : NumPy Riemannian GD (CPU)")
    loss_h, fid_h = [], []
    t0 = time.time()

    for ep in range(n_epochs):
        loss, eg, fids = nfl_loss_np(W, X_np, Phi_np)
        loss_h.append(loss)
        fid_h.append(float(fids.mean()))
        W = riemannian_step(W, eg, lr)
        if ep % log_every == 0 or ep == n_epochs - 1:
            print(f"  ep {ep:>5d}/{n_epochs}  "
                  f"loss={loss:.6f}  "
                  f"F̄={fids.mean():.6f}  "
                  f"[{time.time()-t0:.1f}s]")

    return W, np.array(loss_h), np.array(fid_h)


# ─── Final fidelity evaluation ─────────────────────────────────────
def evaluate_fidelity(W_np, X_np, Phi_np, M_np, dim):
    """
    Compute per-sample state fidelity and process fidelity.

    State fidelity F_j = |<phi_j | W|x_j>|^2 / (||phi_j||^2 * ||W|x_j>||^2)
    Process fidelity    = |Tr(M†W)|^2 / dim^2
    """
    Psi_np   = (W_np @ X_np.T).T               # W|x_j>  shape (t,dim)
    F_per    = state_fidelity_np(Psi_np, Phi_np)
    F_proc   = (abs(np.trace(M_np.conj().T @ W_np)) / dim)**2
    return F_per, float(F_proc)


# ─── Fidelity table ────────────────────────────────────────────────
def print_table(F_per, F_proc, final_loss, n_qubits, dim, t, ep, lr, device):
    Fm = F_per.mean(); Fs = F_per.std()
    Fn = F_per.min();  Fx = F_per.max()
    print()
    print("╔" + "═"*54 + "╗")
    print("║{:^54}║".format("  Fidelity Results — NFL MPS Training  "))
    print("╠" + "═"*54 + "╣")
    rows = [
        ("Dataset type",          "MPS"),
        ("Device",                str(device)),
        ("n_qubits",               str(n_qubits)),
        ("dim  (2^n)",             str(dim)),
        ("Training samples  t",   str(t)),
        ("Epochs",                 str(ep)),
        ("Optimiser",              f"Adam + CosineAnnealing"),
        ("Loss",                   "NFL: Σ||W|x>-|φ>||²/<x|x>"),
        ("Fidelity",               "|<φ|W|x>|² / (||φ||²||Wx||²)"),
        ("", ""),
        ("Final train loss",       f"{final_loss:.8f}"),
        ("", ""),
        ("Mean state fidelity F̄", f"{Fm:.6f}"),
        ("Std  state fidelity",    f"{Fs:.6f}"),
        ("Min  state fidelity",    f"{Fn:.6f}"),
        ("Max  state fidelity",    f"{Fx:.6f}"),
        ("", ""),
        ("Process fidelity",       f"{F_proc:.6f}"),
        ("|Tr(M†W)| / dim",        f"{F_proc**0.5:.6f}"),
    ]
    for k, v in rows:
        if k == "":  print("║" + "─"*54 + "║")
        else:        print(f"║  {k:<30}{v:>20}  ║")
    print("╚" + "═"*54 + "╝")
    return dict(F_mean=Fm, F_std=Fs, F_min=Fn, F_max=Fx, F_process=F_proc)


# ─── Plot ──────────────────────────────────────────────────────────
def plot_results(loss_h, fid_h, F_per, fsum, t, dim, ep, lr, device, path):
    BG   = '#f7f6f2'; TEAL = '#2a9d8f'
    DARK = '#264653'; MUTED= '#6c757d'; OR = '#e76f51'
    fig  = plt.figure(figsize=(18, 12), facecolor=BG)
    gs   = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35,
                             left=0.07, right=0.96, top=0.88, bottom=0.06)
    epochs = np.arange(1, ep + 1)

    def sty(ax):
        ax.set_facecolor('#ffffff')
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
        for s in ['left', 'bottom']: ax.spines[s].set_color('#dcd9d5')
        ax.tick_params(colors=MUTED, labelsize=10)

    # Loss curve
    ax1 = fig.add_subplot(gs[0, 0]); sty(ax1)
    ax1.plot(epochs, loss_h, color=TEAL, lw=2)
    ax1.fill_between(epochs, loss_h, alpha=0.08, color=TEAL)
    ax1.set_title('NFL Training Loss', fontsize=12, fontweight='bold', color=DARK, pad=8)
    ax1.set_xlabel('Epoch', color=MUTED, fontsize=11)
    ax1.set_ylabel('(1/t)Σ||W|x>-|φ>||²/<x|x>', color=MUTED, fontsize=10)
    ax1.text(0.97, 0.97, f'Final: {loss_h[-1]:.5f}', transform=ax1.transAxes,
             ha='right', va='top', fontsize=10, color=TEAL,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff', edgecolor=TEAL, lw=1))

    # Fidelity curve
    ax2 = fig.add_subplot(gs[0, 1]); sty(ax2)
    ax2.plot(epochs, fid_h, color=OR, lw=2)
    ax2.fill_between(epochs, fid_h, alpha=0.08, color=OR)
    ax2.axhline(1.0, color=DARK, lw=1.0, ls=':', alpha=0.4, label='F=1.0')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Mean State Fidelity F̄ During Training',
                  fontsize=12, fontweight='bold', color=DARK, pad=8)
    ax2.set_xlabel('Epoch', color=MUTED, fontsize=11)
    ax2.set_ylabel('F̄ = mean|<φ|W|x>|²/(‖φ‖²‖Wx‖²)', color=MUTED, fontsize=10)
    ax2.text(0.97, 0.07, f'Final: {fsum["F_mean"]:.5f}', transform=ax2.transAxes,
             ha='right', va='bottom', fontsize=10, color=OR,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff', edgecolor=OR, lw=1))
    ax2.legend(fontsize=9, framealpha=0)

    # Per-sample bars
    ax3 = fig.add_subplot(gs[1, 0]); sty(ax3)
    cols = [TEAL if f >= 0.5 else OR for f in F_per]
    ax3.bar(np.arange(t), F_per, color=cols, alpha=0.8, width=0.8)
    ax3.axhline(fsum['F_mean'], color=DARK, lw=1.8, ls='--',
                label=f'Mean = {fsum["F_mean"]:.4f}')
    ax3.axhline(0.5, color=OR, lw=1.0, ls=':', alpha=0.6, label='F=0.5')
    ax3.set_ylim(0, 1.05)
    ax3.set_title(f'Per-Sample State Fidelity  (t={t})',
                  fontsize=12, fontweight='bold', color=DARK, pad=8)
    ax3.set_xlabel('Sample j', color=MUTED, fontsize=11)
    ax3.set_ylabel('F_j', color=MUTED, fontsize=11)
    ax3.legend(fontsize=9, framealpha=0)

    # Summary table
    ax4 = fig.add_subplot(gs[1, 1]); ax4.axis('off')
    td = [
        ['Dataset',            'MPS'],
        ['Device',             str(device)],
        ['n_qubits',            str(int(np.log2(dim)))],
        ['dim  (2^n)',          str(dim)],
        ['t_train',            str(t)],
        ['Epochs',              str(ep)],
        ['Loss',               'NFL paper (norm.)'],
        ['Final loss',         f'{loss_h[-1]:.6f}'],
        ['Mean F̄',            f'{fsum["F_mean"]:.6f}'],
        ['Std  F',             f'{fsum["F_std"]:.6f}'],
        ['Min  F',             f'{fsum["F_min"]:.6f}'],
        ['Max  F',             f'{fsum["F_max"]:.6f}'],
        ['Process fidelity',   f'{fsum["F_process"]:.6f}'],
    ]
    tbl = ax4.table(cellText=td, colLabels=['Metric', 'Value'],
                    cellLoc='left', loc='center', colWidths=[0.58, 0.40])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#dcd9d5'); cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color='white', fontweight='bold')
        elif r > 0 and 'F̄' in td[r-1][0]:
            cell.set_facecolor('#e8f4f3')
            cell.set_text_props(color=TEAL, fontweight='bold')
        elif r > 0 and 'loss' in td[r-1][0].lower():
            cell.set_facecolor('#fff3e8')
            cell.set_text_props(color=OR, fontweight='bold')
        else:
            cell.set_facecolor('#ffffff' if r % 2 == 0 else '#f9f8f5')
            cell.set_text_props(color=DARK)
    ax4.set_title('Fidelity Summary', fontsize=12, fontweight='bold', color=DARK, pad=10)

    fig.suptitle(
        f'Variational MPS  Ψ^PS  Training  —  NFL Pipeline\n'
        f'Loss = (1/t)Σ‖W|xⱼ⟩−|φⱼ⟩‖²/⟨xⱼ|xⱼ⟩    '
        f'Fidelity = |⟨φⱼ|W|xⱼ⟩|²/(‖φⱼ‖²‖Wxⱼ‖²)    '
        f'device={device}  t={t}  dim={dim}',
        fontsize=11, color=DARK, fontweight='bold', y=0.97
    )
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"  Plot saved : {path}")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Device ────────────────────────────────────────────────────
    device = get_device(DEVICE)
    print(f"\nDevice: {device}")

    # ── Load dataset  (raw, DO NOT pre-normalise here) ────────────
    print(f"Loading : {DATASET_PATH}")
    arch     = np.load(DATASET_PATH)
    X_all    = arch["X"]          # (t_total, dim)  — raw MPS states
    Phi_all  = arch["Phi"]        # (t_total, dim)  — M|x_j>  labels
    M        = arch["M"]          # (dim, dim)       — target unitary
    dim      = int(arch["dim"])
    n_qubits = int(arch["n_qubits"])

    assert T_TRAIN <= len(X_all), (
        f"T_TRAIN={T_TRAIN} > available {len(X_all)} samples. "
        f"Re-run generate_groundtruth.py with t_states >= {T_TRAIN}.")

    # Use raw states — loss function handles /<x|x> normalisation internally
    X_train   = X_all[:T_TRAIN]    # (T, dim) complex, raw
    Phi_train = Phi_all[:T_TRAIN]  # (T, dim) complex, raw  = M @ X_train.T

    x_norms = np.linalg.norm(X_train, axis=1)
    print(f"  X   shape : {X_train.shape}   ‖x‖ range [{x_norms.min():.4f}, {x_norms.max():.4f}]")
    print(f"  Phi shape : {Phi_train.shape}  (M applied, same norms)")
    print(f"  M   shape : {M.shape}")

    # Sanity: verify Phi = M @ X exactly
    Phi_check = (M @ X_train.T).T
    err = np.max(np.abs(Phi_check - Phi_train))
    print(f"  Phi sanity check  max|Phi - M@X| = {err:.2e}  "
          f"{'✓' if err < 1e-8 else '⚠ MISMATCH — rerun generate_groundtruth.py'}")

    # ── Train ─────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  Training  Ψ^PS  |  NFL loss  |  CosineAnnealing Adam")
    print("=" * 65)
    print(f"  dim={dim}  n_qubits={n_qubits}  t={T_TRAIN}  "
          f"epochs={N_EPOCHS}  lr={LEARNING_RATE}")
    print("-" * 65)

    if TORCH_AVAILABLE and str(device) != "cpu_numpy":
        W_final, loss_h, fid_h = train_torch(
            X_train, Phi_train, dim,
            N_EPOCHS, LEARNING_RATE, LOG_EVERY, SEED, device)
    else:
        W_final, loss_h, fid_h = train_numpy(
            X_train, Phi_train, dim,
            N_EPOCHS, LEARNING_RATE, LOG_EVERY, SEED)

    # ── Evaluate fidelity ─────────────────────────────────────────
    F_per, F_proc = evaluate_fidelity(W_final, X_train, Phi_train, M, dim)
    final_loss    = float(loss_h[-1])

    # Unitarity check on trained W
    unit_err = np.max(np.abs(W_final.conj().T @ W_final - np.eye(dim)))
    print(f"\n  W unitarity check : max|W†W - I| = {unit_err:.2e}")

    # ── Print table ───────────────────────────────────────────────
    fsum = print_table(F_per, F_proc, final_loss,
                       n_qubits, dim, T_TRAIN, N_EPOCHS, LEARNING_RATE, device)

    # ── Plot ──────────────────────────────────────────────────────
    plot_results(loss_h, fid_h, F_per, fsum,
                 T_TRAIN, dim, N_EPOCHS, LEARNING_RATE, device, OUT_PLOT)

    # ── Save ──────────────────────────────────────────────────────
    np.savez(OUT_RESULTS,
             W            = W_final,
             loss_history = loss_h,
             fid_history  = fid_h,
             F_per_sample = F_per,
             F_process    = np.array(F_proc),
             final_loss   = np.array(final_loss),
             t_train      = np.array(T_TRAIN),
             dim          = np.array(dim))
    print(f"  Results saved : {OUT_RESULTS}")
    print(f"\n  ── Summary ──────────────────────────────")
    print(f"     Mean F̄       = {fsum['F_mean']:.6f}")
    print(f"     Process F     = {fsum['F_process']:.6f}")
    print(f"     Final loss    = {final_loss:.6f}")
    print(f"  ─────────────────────────────────────────")

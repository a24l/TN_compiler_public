"""
train_mps.py — NFL Pipeline Step 4
====================================
Trains variational unitary W (MPS ansatz Psi_PS) on dataset (X, Phi).

NFL paper loss  (arXiv:2412.05674):
    L = (1/t) * sum_j  ||W|x_j> - |phi_j>||^2 / <x_j|x_j>

State fidelity:
    F_j = |<phi_j | W|x_j>|^2 / ( ||phi_j||^2 * ||W|x_j>||^2 )

All fidelity values (mean + per-sample) come from the same on-device
float32 computation inside the training loop — no post-hoc NumPy
re-evaluation of W is used for fidelity reporting.
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

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("output", exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  USER PARAMETERS
# ══════════════════════════════════════════════════════════════════
DATASET_PATH  = "output/dataset_M4_D2_t6000.npz"
T_TRAIN       = 5000
N_EPOCHS      = 1000
LEARNING_RATE = 1e-3
LOG_EVERY     = 50
SEED          = 42
DEVICE        = "cuda"       # "auto" | "cuda" | "mps" | "cpu"
OUT_PLOT      = "output/training_fidelity.png"
OUT_RESULTS   = "output/training_results.npz"
# ══════════════════════════════════════════════════════════════════


def get_device(pref):
    if not TORCH_AVAILABLE:
        return "cpu_numpy"
    if pref == "cuda":
        assert torch.cuda.is_available(), "CUDA not available."
        return torch.device("cuda")
    elif pref == "mps":
        assert torch.backends.mps.is_available(), "MPS not available."
        return torch.device("mps")
    elif pref == "cpu":
        return torch.device("cpu")
    else:
        if torch.cuda.is_available():          return torch.device("cuda")
        if torch.backends.mps.is_available():  return torch.device("mps")
        return torch.device("cpu")


# ── NFL paper loss ────────────────────────────────────────────────
def nfl_loss_torch(Psi, Phi, X_norms_sq):
    """
    Psi          : (t, dim) complex  — W|x_j>
    Phi          : (t, dim) complex  — M|x_j>  (ground truth)
    X_norms_sq   : (t,)     real     — <x_j|x_j>  (input norms squared)

    Returns scalar loss.
    """
    diff    = Psi - Phi
    sq_norm = (diff.real**2 + diff.imag**2).sum(dim=-1)
    return (sq_norm / X_norms_sq).mean()


# ─── Fidelity: simple state fidelity ──────────────────────────────
# F_j = |<phi_j | W|x_j>|^2 / ( ||phi_j||^2 * ||W|x_j>||^2 )
#
# This is purely the cosine-squared angle between the two vectors.
# It equals 1 iff W|x_j> is parallel to phi_j  (correct up to phase).
# It is independent of the norms, so unnormalised states are fine.


# ── State fidelity — on-device torch ─────────────────────────────
def state_fidelity_torch(Psi, Phi):
    """
    F_j = |<phi|Psi>|^2 / (||phi||^2 * ||Psi||^2)
    Psi, Phi : (t, dim) complex tensors on device
    Returns  : (t,) float tensor on device
    """
    num   = (Phi.conj() * Psi).sum(dim=-1).abs()**2
    denom = (Phi.abs()**2).sum(dim=-1) * (Psi.abs()**2).sum(dim=-1)
    return num / denom.clamp(min=1e-30)


# ── Variational unitary model ─────────────────────────────────────
class VarUnitary(nn.Module):
    """
    W = expm(i * H(theta))  where  H = (A - A†) / 2i  is Hermitian.
    Guarantees W†W = I for any theta.
    """
    def __init__(self, dim):
        super().__init__()
        self.A_real = nn.Parameter(torch.randn(dim, dim) * 0.01)
        self.A_imag = nn.Parameter(torch.randn(dim, dim) * 0.01)

    def get_W(self):
        A = torch.complex(self.A_real, self.A_imag)
        H = (A - A.conj().T) / 2j
        return torch.linalg.matrix_exp(1j * H)

    def forward(self, X):
        """Apply W to batch: X (t,dim) -> W|x_j> (t,dim)."""
        return X @ self.get_W().T.conj()


# ── PyTorch training ──────────────────────────────────────────────
def train_torch(X_np, Phi_np, dim, n_epochs, lr, log_every, seed, device):
    torch.manual_seed(seed)
    model = VarUnitary(dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_epochs, eta_min=lr * 0.01)

    Xt    = torch.from_numpy(X_np.astype(np.complex64)).to(device)
    Pt    = torch.from_numpy(Phi_np.astype(np.complex64)).to(device)
    X_nsq = (Xt.real**2 + Xt.imag**2).sum(dim=-1)   # (t,) fixed

    print(f"  Backend : PyTorch on {device}")
    if str(device) == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    loss_h = []
    fid_h  = []
    # Per-sample fidelities from the LAST epoch — captured on-device,
    # same computation as fid_h, moved to CPU only once at the end.
    F_per_last = None

    t0 = time.time()
    for ep in range(n_epochs):
        opt.zero_grad()
        Psi  = model(Xt)                              # W|x_j>  (t, dim)
        loss = nfl_loss_torch(Psi, Pt, X_nsq)
        loss.backward()
        opt.step()
        sched.step()

        with torch.no_grad():
            # (t,) per-sample fidelities — on device, float32
            fids_ep = state_fidelity_torch(Psi.detach(), Pt)
            fid_mean = fids_ep.mean().item()

        loss_h.append(loss.item())
        fid_h.append(fid_mean)

        # Capture per-sample array from the LAST epoch only
        # (avoids storing a (t,) tensor every epoch)
        if ep == n_epochs - 1:
            F_per_last = fids_ep.cpu().numpy().astype(np.float64)

        if ep % log_every == 0 or ep == n_epochs - 1:
            print(f"  ep {ep:>5d}/{n_epochs}  "
                  f"loss={loss.item():.6f}  "
                  f"F̄={fid_mean:.6f}  [{time.time()-t0:.1f}s]")

    model.eval()
    with torch.no_grad():
        W_out = model.get_W().cpu().numpy().astype(np.complex128)

    # F_per_last : (t,) float64, computed on-device at final epoch
    # fid_h[-1]  : scalar = F_per_last.mean() — both are consistent
    return W_out, np.array(loss_h), np.array(fid_h), F_per_last


# ── NumPy Riemannian GD fallback ─────────────────────────────────
def state_fidelity_np(Psi_np, Phi_np):
    num   = np.abs(np.einsum("ij,ij->i", Phi_np.conj(), Psi_np))**2
    denom = (np.abs(Phi_np)**2).sum(axis=1) * (np.abs(Psi_np)**2).sum(axis=1)
    return num / np.where(denom > 0, denom, 1.0)

def nfl_loss_np(W, X, Phi):
    t       = len(X)
    Psi     = (W @ X.T).T
    diff    = Psi - Phi
    xnorms2 = np.sum(np.abs(X)**2, axis=1)
    loss    = float(np.mean(np.sum(np.abs(diff)**2, axis=1) / xnorms2))
    eg      = (1/t) * np.einsum("ij,ik->jk", diff / xnorms2[:, None], X.conj())
    fids    = state_fidelity_np(Psi, Phi)
    return loss, eg, fids

def riemannian_step(W, egrad, lr):
    G     = W.conj().T @ egrad
    skew  = G - G.conj().T
    W_new = W @ (np.eye(W.shape[0]) - (lr / 2) * skew)
    U, _, Vh = np.linalg.svd(W_new)
    return U @ Vh

def train_numpy(X_np, Phi_np, dim, n_epochs, lr, log_every, seed):
    rng  = np.random.default_rng(seed)
    Z    = rng.standard_normal((dim,dim)) + 1j*rng.standard_normal((dim,dim))
    W, _ = np.linalg.qr(Z)
    print("  Backend : NumPy Riemannian GD (CPU)")
    loss_h, fid_h = [], []
    F_per_last    = None
    t0 = time.time()
    for ep in range(n_epochs):
        loss, eg, fids = nfl_loss_np(W, X_np, Phi_np)
        loss_h.append(loss)
        fid_h.append(float(fids.mean()))
        W = riemannian_step(W, eg, lr)
        if ep == n_epochs - 1:
            F_per_last = fids.astype(np.float64)
        if ep % log_every == 0 or ep == n_epochs - 1:
            print(f"  ep {ep:>5d}/{n_epochs}  "
                  f"loss={loss:.6f}  F̄={fids.mean():.6f}  [{time.time()-t0:.1f}s]")
    return W, np.array(loss_h), np.array(fid_h), F_per_last


# ── Process fidelity ──────────────────────────────────────────────
def process_fidelity(W_np, M_np, dim):
    return (abs(np.trace(M_np.conj().T @ W_np)) / dim)**2


# ── Summary table ─────────────────────────────────────────────────
def print_table(F_per, F_proc, final_loss, final_fid, n_qubits, dim, t, ep, device):
    """
    F_per      : (t,) per-sample fidelities from last training epoch, on-device
    final_fid  : fid_h[-1] = F_per.mean() — mean from the same computation
    All four statistics (mean/std/min/max) are from the SAME on-device tensor.
    """
    print()
    print("╔" + "═"*56 + "╗")
    print("║{:^56}║".format("  Fidelity Results — NFL MPS Training  "))
    print("╠" + "═"*56 + "╣")
    rows = [
        ("Dataset type",            "MPS"),
        ("Device",                   str(device)),
        ("n_qubits",                 str(n_qubits)),
        ("dim  (2^n)",               str(dim)),
        ("Training samples  t",     str(t)),
        ("Epochs",                   str(ep)),
        ("Optimiser",                "Adam + CosineAnnealing"),
        ("Loss",                     "NFL: Σ‖W|x⟩−|φ⟩‖²/⟨x|x⟩"),
        ("Fidelity",                 "|⟨φ|W|x⟩|²/(‖φ‖²‖Wx‖²)"),
        ("", ""),
        ("Final train loss",         f"{final_loss:.8f}"),
        ("", ""),
        ("Mean F̄  [from training]",  f"{final_fid:.6f}"),
        ("Std  F  [last epoch]",     f"{F_per.std():.6f}"),
        ("Min  F  [last epoch]",     f"{F_per.min():.6f}"),
        ("Max  F  [last epoch]",     f"{F_per.max():.6f}"),
        ("", ""),
        ("Process fidelity",         f"{F_proc:.6f}"),
        ("|Tr(M†W)|/dim",            f"{F_proc**0.5:.6f}"),
    ]
    for k, v in rows:
        if k == "":
            print("║" + "─"*56 + "║")
        else:
            print(f"║  {k:<32}{v:>20}  ║")
    print("╚" + "═"*56 + "╝")
    return dict(F_mean=final_fid,
                F_std=F_per.std(),
                F_min=F_per.min(),
                F_max=F_per.max(),
                F_process=F_proc)


# ── Plot ──────────────────────────────────────────────────────────
def plot_results(loss_h, fid_h, F_per, fsum, t, dim, ep, device, path):
    BG   = '#f7f6f2'; TEAL = '#2a9d8f'
    DARK = '#264653'; MUTED= '#6c757d'; OR = '#e76f51'
    fig  = plt.figure(figsize=(18, 12), facecolor=BG)
    gs   = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.35,
                             left=0.07, right=0.96, top=0.88, bottom=0.06)
    epochs = np.arange(1, ep + 1)

    def sty(ax):
        ax.set_facecolor('#ffffff')
        for s in ['top', 'right']:   ax.spines[s].set_visible(False)
        for s in ['left', 'bottom']: ax.spines[s].set_color('#dcd9d5')
        ax.tick_params(colors=MUTED, labelsize=10)

    # Loss curve
    ax1 = fig.add_subplot(gs[0, 0]); sty(ax1)
    ax1.plot(epochs, loss_h, color=TEAL, lw=2)
    ax1.fill_between(epochs, loss_h, alpha=0.08, color=TEAL)
    ax1.set_title('NFL Training Loss', fontsize=12, fontweight='bold', color=DARK, pad=8)
    ax1.set_xlabel('Epoch', color=MUTED, fontsize=11)
    ax1.set_ylabel('(1/t)Σ‖W|x⟩−|φ⟩‖²/⟨x|x⟩', color=MUTED, fontsize=10)
    ax1.text(0.97, 0.97, f'Final: {loss_h[-1]:.5f}', transform=ax1.transAxes,
             ha='right', va='top', fontsize=10, color=TEAL,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff', edgecolor=TEAL, lw=1))

    # Fidelity curve — fid_h is the canonical source throughout
    ax2 = fig.add_subplot(gs[0, 1]); sty(ax2)
    ax2.plot(epochs, fid_h, color=OR, lw=2)
    ax2.fill_between(epochs, fid_h, alpha=0.08, color=OR)
    ax2.axhline(1.0, color=DARK, lw=1.0, ls=':', alpha=0.4, label='F=1.0')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Mean State Fidelity F̄ During Training',
                  fontsize=12, fontweight='bold', color=DARK, pad=8)
    ax2.set_xlabel('Epoch', color=MUTED, fontsize=11)
    ax2.set_ylabel('F̄ = (1/t)Σ|⟨φ|W|x⟩|²/(‖φ‖²‖Wx‖²)', color=MUTED, fontsize=10)
    ax2.text(0.97, 0.07, f'Final: {fid_h[-1]:.5f}', transform=ax2.transAxes,
             ha='right', va='bottom', fontsize=10, color=OR,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff', edgecolor=OR, lw=1))
    ax2.legend(fontsize=9, framealpha=0)

    # Per-sample bar — F_per is from on-device last-epoch computation
    ax3 = fig.add_subplot(gs[1, 0]); sty(ax3)
    cols = [TEAL if f >= 0.5 else OR for f in F_per]
    ax3.bar(np.arange(len(F_per)), F_per, color=cols, alpha=0.8, width=0.8)
    ax3.axhline(fsum['F_mean'], color=DARK, lw=1.8, ls='--',
                label=f'Mean = {fsum["F_mean"]:.4f}')
    ax3.axhline(0.5, color=OR, lw=1.0, ls=':', alpha=0.6, label='F=0.5')
    ax3.set_ylim(0, 1.05)
    ax3.set_title(f'Per-Sample Fidelity — Last Epoch  (t={t})',
                  fontsize=12, fontweight='bold', color=DARK, pad=8)
    ax3.set_xlabel('Sample j', color=MUTED, fontsize=11)
    ax3.set_ylabel('F_j', color=MUTED, fontsize=11)
    ax3.legend(fontsize=9, framealpha=0)

    # Summary table
    ax4 = fig.add_subplot(gs[1, 1]); ax4.axis('off')
    td = [
        ['Dataset',            'MPS'],
        ['Device',              str(device)],
        ['n_qubits',            str(int(np.log2(dim)))],
        ['dim  (2^n)',          str(dim)],
        ['t_train',             str(t)],
        ['Epochs',              str(ep)],
        ['Loss',               'NFL paper (normalised)'],
        ['Final loss',         f'{loss_h[-1]:.6f}'],
        ['Mean F̄ (training)',  f'{fsum["F_mean"]:.6f}'],
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
        f'Loss=(1/t)Σ‖W|xⱼ⟩−|φⱼ⟩‖²/⟨xⱼ|xⱼ⟩    '
        f'F̄={fid_h[-1]:.4f} at epoch {ep}    '
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

    device = get_device(DEVICE)
    print(f"\nDevice: {device}")

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

    X_train   = X_all[:T_TRAIN]
    Phi_train = Phi_all[:T_TRAIN]

    x_norms = np.linalg.norm(X_train, axis=1)
    print(f"  X   shape : {X_train.shape}   ‖x‖ range [{x_norms.min():.4f}, {x_norms.max():.4f}]")
    print(f"  Phi shape : {Phi_train.shape}  (M applied, same norms)")
    print(f"  M   shape : {M.shape}")

    Phi_check = (M @ X_train.T).T
    err = np.max(np.abs(Phi_check - Phi_train))
    print(f"  Phi sanity  max|Phi - M@X| = {err:.2e}  "
          f"{'✓' if err < 1e-8 else '⚠ MISMATCH'}")

    print()
    print("=" * 65)
    print("  Training  Ψ^PS  |  NFL loss  |  CosineAnnealing Adam")
    print("=" * 65)
    print(f"  dim={dim}  n_qubits={n_qubits}  t={T_TRAIN}  "
          f"epochs={N_EPOCHS}  lr={LEARNING_RATE}")
    print("-" * 65)

    if TORCH_AVAILABLE and str(device) != "cpu_numpy":
        W_final, loss_h, fid_h, F_per = train_torch(
            X_train, Phi_train, dim,
            N_EPOCHS, LEARNING_RATE, LOG_EVERY, SEED, device)
    else:
        W_final, loss_h, fid_h, F_per = train_numpy(
            X_train, Phi_train, dim,
            N_EPOCHS, LEARNING_RATE, LOG_EVERY, SEED)

    # Unitarity check on trained W
    unit_err = np.max(np.abs(W_final.conj().T @ W_final - np.eye(dim)))
    print(f"\n  W unitarity check : max|W†W - I| = {unit_err:.2e}")

    # Process fidelity (scalar, float64 W_final is fine here)
    F_proc = process_fidelity(W_final, M, dim)

    # fid_h[-1]  — mean F̄ from last epoch (on-device, consistent with log)
    # F_per      — per-sample F_j from last epoch (same computation, same device)
    # F_per.mean() == fid_h[-1] by construction
    fsum = print_table(F_per, F_proc, loss_h[-1], fid_h[-1],
                       n_qubits, dim, T_TRAIN, N_EPOCHS, device)

    plot_results(loss_h, fid_h, F_per, fsum,
                 T_TRAIN, dim, N_EPOCHS, device, OUT_PLOT)

    np.savez(OUT_RESULTS,
             W            = W_final,
             loss_history = loss_h,
             fid_history  = fid_h,
             F_per_sample = F_per,          # on-device, last epoch
             F_process    = np.array(F_proc),
             final_loss   = np.array(loss_h[-1]),
             final_fid    = np.array(fid_h[-1]),
             t_train      = np.array(T_TRAIN),
             dim          = np.array(dim))
    print(f"  Results saved : {OUT_RESULTS}")

    print(f"\n  ── Final Summary ────────────────────────────────")
    print(f"     Mean F̄  (fid_h[-1])   = {fid_h[-1]:.6f}")
    print(f"     Std  F  (last epoch)   = {F_per.std():.6f}")
    print(f"     Min  F  (last epoch)   = {F_per.min():.6f}")
    print(f"     Max  F  (last epoch)   = {F_per.max():.6f}")
    print(f"     Process fidelity       = {F_proc:.6f}")
    print(f"     Final loss             = {loss_h[-1]:.6f}")
    print(f"  ─────────────────────────────────────────────────")

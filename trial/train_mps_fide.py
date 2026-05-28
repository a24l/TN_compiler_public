"""
train_mps.py
============
Step 4 of the NFL pipeline.

Train a variational unitary W (MPS ansatz) on dataset (X, Phi)
from generate_M_and_dataset.py, then evaluate state fidelity.

Loss:
    L(θ) = (1/t) Σ_j  ‖W(θ)|x_j⟩ − M|x_j⟩‖²

State fidelity per sample:
    F_j = |⟨φ_j | W|x_j⟩|² / (‖φ_j‖² · ‖W|x_j⟩‖²)
    where  |φ_j⟩ = M|x_j⟩  (label)
           |φ̃_j⟩ = W|x_j⟩  (prediction)

Mean state fidelity:
    F̄ = (1/t) Σ_j F_j

Process fidelity (unitary overlap):
    F_process = |Tr(M†W)|² / dim²

Run:
    python build_pool.py
    python generate_M_and_dataset.py
    python train_mps.py

Requires: numpy, scipy, matplotlib
"""

import argparse
import numpy as np
import os, sys, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import expm as scipy_expm

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("output", exist_ok=True)

try:
    import cupy as cp
    from cupyx.scipy.linalg import expm as cupy_expm
except ImportError:
    cp = None
    cupy_expm = None

XP = np
EXPM = scipy_expm
TO_NUMPY = np.asarray
ACTIVE_DEVICE = "cpu"


def _select_backend(device: str):
    requested = device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Invalid device '{device}'. Use one of: auto, cpu, cuda.")

    if requested == "cpu":
        return np, scipy_expm, np.asarray, "cpu"

    if cp is not None and cupy_expm is not None:
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp, cupy_expm, cp.asnumpy, "cuda"
        except Exception as exc:
            if requested == "cuda":
                raise RuntimeError(f"CUDA backend requested but unavailable: {exc}") from exc

    if requested == "cuda":
        raise RuntimeError(
            "CUDA backend requested but unavailable. Install a CUDA-matched CuPy build "
            "(for example: `pip install cupy-cuda12x`) and ensure `nvidia-smi` works."
        )

    return np, scipy_expm, np.asarray, "cpu"


def configure_backend(device: str) -> None:
    global XP, EXPM, TO_NUMPY, ACTIVE_DEVICE
    XP, EXPM, TO_NUMPY, ACTIVE_DEVICE = _select_backend(device)


# ══════════════════════════════════════════════════════════════════
#  USER PARAMETERS
# ══════════════════════════════════════════════════════════════════
DATASET_PATH  = "output/dataset_M4_D2_t6000.npz"
T_TRAIN       = 6000      # ← training set size (≤ t_states in dataset)
                        #   change to 5, 10, 20, 30, 50, ...
N_EPOCHS      = 1000     # gradient steps
LEARNING_RATE = 0.001    # Adam lr
LOG_EVERY     = 50      # print interval
SEED          = 1       # reproducibility
OUT_PLOT      = "output/training_fidelity.png"
OUT_RESULTS   = "output/training_results.npz"
# ══════════════════════════════════════════════════════════════════


# ── Unitary parametrisation via Lie algebra ───────────────────────
def params_to_unitary(theta, dim):
    """W = expm(i(A−A†)/2)  guarantees W†W=I for any θ."""
    n2  = dim * dim
    A   = theta[:n2].reshape(dim, dim) + 1j * theta[n2:].reshape(dim, dim)
    return EXPM(1j * (A - A.conj().T) / 2.0)

def init_params(dim, rng):
    return XP.asarray(rng.standard_normal(2 * dim * dim) * 0.01)


# ── Loss ──────────────────────────────────────────────────────────
def mps_loss(W, X, Phi):
    """L = (1/t) Σ ‖W|x_j⟩ − |φ_j⟩‖²"""
    diff = (W @ X.T).T - Phi
    return float(XP.mean(XP.sum(XP.abs(diff)**2, axis=1)))


# ── Numerical gradient ────────────────────────────────────────────
def numerical_gradient(theta, dim, X, Phi, eps=1e-5):
    grad = XP.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        grad[i] = (mps_loss(params_to_unitary(tp, dim), X, Phi) -
                   mps_loss(params_to_unitary(tm, dim), X, Phi)) / (2 * eps)
    return grad


# ── Fidelity metrics ──────────────────────────────────────────────
def state_fidelities(W, X, Phi):
    """
    Per-sample normalised state fidelity:
        F_j = |⟨φ_j | W|x_j⟩|² / (‖φ_j‖² · ‖W|x_j⟩‖²)

    Since W is unitary: ‖W|x_j⟩‖ = ‖|x_j⟩‖ = ‖|φ_j⟩‖,
    so denominator = ‖x_j‖⁴.

    Returns array (t,) with F_j ∈ [0, 1].
    """
    Psi   = (W @ X.T).T
    num   = XP.abs(XP.einsum("ij,ij->i", Phi.conj(), Psi))**2
    denom = (XP.linalg.norm(Phi, axis=1)**2) * (XP.linalg.norm(Psi, axis=1)**2)
    return num / XP.where(denom > 0, denom, 1.0)

def process_fidelity(W, M, dim):
    """F_process = |Tr(M†W)|² / dim²  (Haar-averaged gate fidelity)"""
    return float((abs(XP.trace(M.conj().T @ W)) / dim)**2)


# ── Adam optimiser ────────────────────────────────────────────────
class Adam:
    def __init__(self, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps
        self.m=None; self.v=None; self.t=0
    def step(self, theta, grad):
        if self.m is None:
            self.m=XP.zeros_like(grad); self.v=XP.zeros_like(grad)
        self.t+=1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*grad**2
        mh = self.m/(1-self.b1**self.t)
        vh = self.v/(1-self.b2**self.t)
        return theta - self.lr*mh/(XP.sqrt(vh)+self.eps)


# ── Training loop ─────────────────────────────────────────────────
def train(X, Phi, M, dim, n_epochs, lr, log_every, seed):
    rng   = np.random.default_rng(seed)
    theta = init_params(dim, rng)
    opt   = Adam(lr=lr)

    train_losses, fid_history = [], []
    t0 = time.time()

    print("=" * 65)
    print("  Training Variational MPS Model  Ψ^PS")
    print("=" * 65)
    print(f"  dim={dim}  t_train={len(X)}  epochs={n_epochs}  lr={lr}")
    print(f"  params = {len(theta)}  (2 × {dim}²)")
    print("-" * 65)

    for epoch in range(n_epochs):
        W    = params_to_unitary(theta, dim)
        loss = mps_loss(W, X, Phi)
        fids = state_fidelities(W, X, Phi)
        mean_fid = float(XP.mean(fids))
        train_losses.append(loss)
        fid_history.append(mean_fid)

        grad  = numerical_gradient(theta, dim, X, Phi)
        theta = opt.step(theta, grad)

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:>4d}/{n_epochs}   "
                  f"loss={loss:.6f}   "
                  f"mean_fidelity={mean_fid:.6f}   "
                  f"[{time.time()-t0:.1f}s]")

    W_final = params_to_unitary(theta, dim)
    return W_final, np.array(train_losses), np.array(fid_history)


# ── Fidelity table printer ────────────────────────────────────────
def print_fidelity_table(F_per_sample, F_process, final_loss,
                          n_qubits, dim, t_train, n_epochs, lr):
    F_mean = F_per_sample.mean()
    F_std  = F_per_sample.std()
    F_min  = F_per_sample.min()
    F_max  = F_per_sample.max()

    sep = "─" * 48
    print()
    print("╔" + "═"*48 + "╗")
    print("║{:^48}║".format("  Fidelity Results — NFL MPS Dataset  "))
    print("╠" + "═"*48 + "╣")
    rows = [
        ("Dataset type",         "MPS"),
        ("n_qubits",              str(n_qubits)),
        ("dim  (d^n)",            f"{dim}  (2^{n_qubits})"),
        ("Training samples  t",  str(t_train)),
        ("Epochs",                str(n_epochs)),
        ("Optimiser",             f"Adam  lr={lr}"),
        (sep, ""),
        ("Final train loss",      f"{final_loss:.8f}"),
        (sep, ""),
        ("Mean state fidelity F̄", f"{F_mean:.6f}"),
        ("Std  state fidelity",   f"{F_std:.6f}"),
        ("Min  state fidelity",   f"{F_min:.6f}"),
        ("Max  state fidelity",   f"{F_max:.6f}"),
        (sep, ""),
        ("Process fidelity",      f"{F_process:.6f}"),
        ("|Tr(M†W)| / dim",       f"{F_process**0.5:.6f}"),
    ]
    for k, v in rows:
        if k.startswith("─"):
            print("║" + "─"*48 + "║")
        else:
            print(f"║  {k:<26}{v:>18}  ║")
    print("╚" + "═"*48 + "╝")
    return {"F_mean": F_mean, "F_std": F_std, "F_min": F_min,
            "F_max": F_max, "F_process": F_process}


# ── Plot ──────────────────────────────────────────────────────────
def plot_results(train_losses, fid_history, F_per_sample, fid_summary,
                 t_train, dim, n_epochs, lr, final_loss, save_path):
    BG='#f7f6f2'; TEAL='#2a9d8f'; DARK='#264653'
    MUTED='#6c757d'; ORANGE='#e76f51'
    F_mean = fid_summary["F_mean"]
    F_process = fid_summary["F_process"]

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs  = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.32,
                            left=0.07, right=0.96, top=0.88, bottom=0.06)
    epochs = np.arange(1, n_epochs+1)

    def style_ax(ax):
        ax.set_facecolor('#ffffff')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        for sp in ['left','bottom']: ax.spines[sp].set_color('#dcd9d5')
        ax.tick_params(colors=MUTED, labelsize=10)

    # Loss curve
    ax1 = fig.add_subplot(gs[0, 0]); style_ax(ax1)
    ax1.plot(epochs, train_losses, color=TEAL, lw=2.2)
    ax1.fill_between(epochs, train_losses, alpha=0.08, color=TEAL)
    ax1.set_title('Training Loss', fontsize=12, fontweight='bold', color=DARK, pad=8)
    ax1.set_xlabel('Epoch', color=MUTED, fontsize=11)
    ax1.set_ylabel('Loss', color=MUTED, fontsize=11)
    ax1.text(0.97, 0.97, f'Final: {final_loss:.5f}', transform=ax1.transAxes,
             ha='right', va='top', fontsize=10, color=TEAL,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff',
                       edgecolor=TEAL, lw=1))

    # Fidelity curve
    ax2 = fig.add_subplot(gs[0, 1]); style_ax(ax2)
    ax2.plot(epochs, fid_history, color=ORANGE, lw=2.2)
    ax2.fill_between(epochs, fid_history, alpha=0.08, color=ORANGE)
    ax2.set_title('Mean State Fidelity During Training',
                  fontsize=12, fontweight='bold', color=DARK, pad=8)
    ax2.set_xlabel('Epoch', color=MUTED, fontsize=11)
    ax2.set_ylabel(r'$\bar{F}$', color=MUTED, fontsize=12)
    ax2.text(0.97, 0.07, f'Final: {F_mean:.5f}', transform=ax2.transAxes,
             ha='right', va='bottom', fontsize=10, color=ORANGE,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff',
                       edgecolor=ORANGE, lw=1))

    # Per-sample bar
    ax3 = fig.add_subplot(gs[1, 0]); style_ax(ax3)
    xpos = np.arange(t_train)
    ax3.bar(xpos, F_per_sample, color=TEAL, alpha=0.75, width=0.8)
    ax3.axhline(F_mean, color=ORANGE, lw=1.8, ls='--',
                label=f'Mean = {F_mean:.4f}')
    ax3.set_title(f'Per-Sample State Fidelity  (t = {t_train})',
                  fontsize=12, fontweight='bold', color=DARK, pad=8)
    ax3.set_xlabel('Training sample  j', color=MUTED, fontsize=11)
    ax3.set_ylabel(r'$F_j$', color=MUTED, fontsize=12)
    ax3.set_xlim(-0.5, t_train-0.5)
    ax3.set_xticks(np.arange(0, t_train+1, 10))
    ax3.legend(fontsize=10, framealpha=0)

    # Summary table
    ax4 = fig.add_subplot(gs[1, 1]); ax4.axis('off')
    table_data = [
        ['Dataset type',         'MPS'],
        ['n_qubits',              str(int(np.log2(dim)))],
        ['dim  (2^n)',            str(dim)],
        ['Training samples  t',  str(t_train)],
        ['Epochs',                str(n_epochs)],
        ['Optimiser',             f'Adam  lr={lr}'],
        ['Final train loss',      f'{final_loss:.6f}'],
        ['Mean state fidelity',   f'{F_mean:.6f}'],
        ['Std  state fidelity',   f'{fid_summary["F_std"]:.6f}'],
        ['Min  state fidelity',   f'{fid_summary["F_min"]:.6f}'],
        ['Max  state fidelity',   f'{fid_summary["F_max"]:.6f}'],
        ['Process fidelity',      f'{F_process:.6f}'],
        [' |Tr(M†W)| / dim',      f'{F_process**0.5:.6f}'],
    ]
    tbl = ax4.table(cellText=table_data, colLabels=['Metric', 'Value'],
                    cellLoc='left', loc='center', colWidths=[0.60, 0.38])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    highlight_rows = {'Mean state fidelity', 'Process fidelity', 'Final train loss'}
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#dcd9d5'); cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(DARK)
            cell.set_text_props(color='white', fontweight='bold')
        elif table_data[r-1][0] in highlight_rows:
            fc = '#e8f4f3' if 'fidelity' in table_data[r-1][0] else '#fff3e8'
            col_ = TEAL if 'fidelity' in table_data[r-1][0] else ORANGE
            cell.set_facecolor(fc)
            cell.set_text_props(color=col_, fontweight='bold')
        else:
            cell.set_facecolor('#ffffff' if r%2==0 else '#f9f8f5')
            cell.set_text_props(color=DARK)
    ax4.set_title('Fidelity Summary', fontsize=12, fontweight='bold',
                  color=DARK, pad=10)

    fig.suptitle(
        'Variational MPS Training & State Fidelity  —  NFL Pipeline\n'
        r'$F_j = |\langle\varphi_j|W|x_j\rangle|^2 / (\|\varphi_j\|^2\|\tilde\varphi_j\|^2)$'
        f'     |φⱼ⟩=M|xⱼ⟩  (label)     |φ̃ⱼ⟩=W|xⱼ⟩  (prediction)',
        fontsize=12, color=DARK, fontweight='bold', y=0.97
    )
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Saved plot: {save_path}")


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the variational MPS model with CPU or CUDA backend."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=os.environ.get("TN_DEVICE", "auto"),
        help="Execution device. auto picks CUDA when available.",
    )
    args = parser.parse_args()
    configure_backend(args.device)
    print(f"Backend: {ACTIVE_DEVICE}")

    # Load dataset
    print(f"Loading: {DATASET_PATH}")
    arch     = np.load(DATASET_PATH)
    X_all    = arch["X"]; Phi_all = arch["Phi"]; M = arch["M"]
    dim      = int(arch["dim"]); n_qubits = int(arch["n_qubits"])

    assert T_TRAIN <= len(X_all), (
        f"T_TRAIN={T_TRAIN} > available {len(X_all)}. "
        f"Re-run generate_M_and_dataset.py with t_states ≥ {T_TRAIN}.")

    X_train   = X_all[:T_TRAIN]
    Phi_train = Phi_all[:T_TRAIN]
    print(f"  X={X_train.shape}  Phi={Phi_train.shape}  M={M.shape}")

    X_train_xp = XP.asarray(X_train)
    Phi_train_xp = XP.asarray(Phi_train)
    M_xp = XP.asarray(M)

    # Train
    W_final, train_losses, fid_history = train(
        X_train_xp, Phi_train_xp, M_xp, dim,
        N_EPOCHS, LEARNING_RATE, LOG_EVERY, SEED
    )

    # Compute final fidelities on training set
    F_per_sample = TO_NUMPY(state_fidelities(W_final, X_train_xp, Phi_train_xp))
    F_proc       = process_fidelity(W_final, M_xp, dim)
    final_loss   = float(train_losses[-1])

    # Print table
    fid_summary = print_fidelity_table(
        F_per_sample, F_proc, final_loss,
        n_qubits, dim, T_TRAIN, N_EPOCHS, LEARNING_RATE
    )

    # Plot
    plot_results(train_losses, fid_history, F_per_sample, fid_summary,
                 T_TRAIN, dim, N_EPOCHS, LEARNING_RATE, final_loss, OUT_PLOT)

    # Save
    np.savez(OUT_RESULTS,
             W=TO_NUMPY(W_final), train_losses=train_losses, fid_history=fid_history,
             F_per_sample=F_per_sample, F_process=np.array(F_proc),
             final_loss=np.array(final_loss), t_train=np.array(T_TRAIN), dim=np.array(dim))
    print(f"Saved results: {OUT_RESULTS}")

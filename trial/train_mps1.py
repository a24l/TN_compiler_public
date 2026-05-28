"""
train_mps.py  —  NFL Pipeline Step 4
Fidelity loss + Riemannian GD + CUDA support
"""
import numpy as np, os, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch, torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found — falling back to numpy Riemannian GD")

from scipy.linalg import expm
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("output", exist_ok=True)

# ══════════════════════════════════════════════════════════
#  USER PARAMETERS
# ══════════════════════════════════════════════════════════
DATASET_PATH  = "output/dataset_M4_D2_t6000.npz"
T_TRAIN       = 5000      # training set size (≤ t_states in dataset)
N_EPOCHS      = 1000     # gradient steps
LEARNING_RATE = 0.0001    # step size
LOG_EVERY     = 50
SEED          = 42
DEVICE        = "auto"  # "auto" | "cuda" | "mps" | "cpu"
OUT_PLOT      = "output/training_fidelity.png"
OUT_RESULTS   = "output/training_results.npz"
# ══════════════════════════════════════════════════════════


def get_device(pref):
    if not TORCH_AVAILABLE:
        return "cpu_numpy"
    import torch
    if pref == "cuda":
        assert torch.cuda.is_available(), "CUDA not available"
        return torch.device("cuda")
    elif pref == "mps":
        assert torch.backends.mps.is_available(), "MPS not available"
        return torch.device("mps")
    elif pref == "cpu":
        return torch.device("cpu")
    else:
        if torch.cuda.is_available():   return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")


def normalise(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


# ── Fidelity loss and Euclidean gradient ──────────────────
def fid_egrad(W, X, Phi):
    """
    L = 1 - (1/t) Σ_j |⟨φ_j|W|x_j⟩|²
    Returns (loss, egrad, per_sample_fidelities)
    egrad = dL/dW = -(1/t) Σ_j conj(ovlp_j) |φ_j⟩⟨x_j|
    """
    t    = len(X)
    Psi  = (W @ X.T).T                                     # (t, dim)
    ovlp = np.einsum("ij,ij->i", Phi.conj(), Psi)          # (t,) complex
    fids = np.abs(ovlp)**2                                  # (t,) real
    loss = 1.0 - fids.mean()
    eg   = -(1/t) * np.einsum("i,ij,ik->jk",
                               ovlp.conj(), Phi, X.conj()) # (dim,dim)
    return float(loss), eg, fids


# ── Riemannian step — stays on U(dim) exactly ─────────────
def riemannian_step(W, egrad, lr):
    """
    1. Pull egrad to tangent space:  skew = W†·eg - eg†·W
    2. First-order Cayley update:    W_new = W·(I - lr/2·skew)
    3. SVD retraction back to U(dim): W_new = U·Vh
    """
    G     = W.conj().T @ egrad
    skew  = G - G.conj().T                         # skew-Hermitian
    W_new = W @ (np.eye(W.shape[0]) - lr/2 * skew)
    U, _, Vh = np.linalg.svd(W_new)
    return U @ Vh


# ── PyTorch model (used when CUDA available) ───────────────
def make_torch_model(dim, device):
    import torch, torch.nn as nn
    class VarUnitary(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.A_real = nn.Parameter(torch.randn(dim,dim)*0.01)
            self.A_imag = nn.Parameter(torch.randn(dim,dim)*0.01)
        def get_W(self):
            A = torch.complex(self.A_real, self.A_imag)
            H = (A - A.conj().T) / 2j
            return torch.linalg.matrix_exp(1j * H)
        def forward(self, X):
            return X @ self.get_W().T.conj()
    return VarUnitary(dim).to(device)


# ── Training dispatcher ────────────────────────────────────
def train(X_np, Phi_np, M_np, dim, n_epochs, lr, log_every, seed, device):
    rng = np.random.default_rng(seed)

    if str(device) != "cpu_numpy" and TORCH_AVAILABLE:
        # ── PyTorch path (CUDA / MPS / CPU) ───────────────
        import torch
        torch.manual_seed(seed)
        model = make_torch_model(dim, device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=n_epochs, eta_min=lr*0.01)

        Xt  = torch.from_numpy(X_np.astype(np.complex64)).to(device)
        Pt  = torch.from_numpy(Phi_np.astype(np.complex64)).to(device)

        print(f"  Backend : PyTorch on {device}")
        loss_h, fid_h = [], []
        t0 = time.time()

        for ep in range(n_epochs):
            opt.zero_grad()
            Psi  = model(Xt)
            ovlp = (Pt.conj() * Psi).sum(dim=-1)
            loss = 1.0 - (ovlp.abs()**2).mean()
            loss.backward(); opt.step(); sched.step()
            fid = 1.0 - loss.item()
            loss_h.append(loss.item()); fid_h.append(fid)
            if ep % log_every == 0 or ep == n_epochs-1:
                print(f"  ep {ep:>5d}/{n_epochs}  loss={loss.item():.6f}"
                      f"  F̄={fid:.6f}  [{time.time()-t0:.1f}s]")

        model.eval()
        with torch.no_grad():
            W_out = model.get_W().cpu().numpy().astype(np.complex128)

    else:
        # ── NumPy Riemannian GD fallback ───────────────────
        print(f"  Backend : NumPy Riemannian GD (CPU)")
        Z    = rng.standard_normal((dim,dim)) + 1j*rng.standard_normal((dim,dim))
        W, _ = np.linalg.qr(Z)

        loss_h, fid_h = [], []
        t0 = time.time()

        for ep in range(n_epochs):
            loss, eg, fids = fid_egrad(W, X_np, Phi_np)
            loss_h.append(loss); fid_h.append(float(fids.mean()))
            W = riemannian_step(W, eg, lr)
            if ep % log_every == 0 or ep == n_epochs-1:
                print(f"  ep {ep:>5d}/{n_epochs}  loss={loss:.6f}"
                      f"  F̄={fids.mean():.6f}  [{time.time()-t0:.1f}s]")

        W_out = W

    return W_out, np.array(loss_h), np.array(fid_h)


# ── Fidelity metrics ───────────────────────────────────────
def compute_fidelities(W, X, Phi, M, dim):
    _, _, F_per = fid_egrad(W, X, Phi)
    F_proc = (abs(np.trace(M.conj().T @ W)) / dim)**2
    return F_per, F_proc


# ── Fidelity table ─────────────────────────────────────────
def print_table(F_per, F_proc, loss, n_qubits, dim, t, ep, lr, device):
    Fm,Fs,Fn,Fx = F_per.mean(),F_per.std(),F_per.min(),F_per.max()
    print()
    print("╔" + "═"*52 + "╗")
    print("║{:^52}║".format("  Fidelity Results — NFL MPS Dataset  "))
    print("╠" + "═"*52 + "╣")
    rows = [
        ("Dataset type",         "MPS"),
        ("Device",               str(device)),
        ("n_qubits",              str(n_qubits)),
        ("dim  (2^n)",            f"{dim}"),
        ("Training samples  t",  str(t)),
        ("Epochs",                str(ep)),
        ("Loss",                 "1 − |⟨φ|W|x⟩|²"),
        ("", ""),
        ("Final fidelity loss",  f"{loss:.8f}"),
        ("", ""),
        ("Mean state fidelity",  f"{Fm:.6f}"),
        ("Std  state fidelity",  f"{Fs:.6f}"),
        ("Min  state fidelity",  f"{Fn:.6f}"),
        ("Max  state fidelity",  f"{Fx:.6f}"),
        ("", ""),
        ("Process fidelity",     f"{F_proc:.6f}"),
        ("|Tr(M†W)| / dim",      f"{F_proc**0.5:.6f}"),
    ]
    for k,v in rows:
        if k=="":  print("║"+"─"*52+"║")
        else:      print(f"║  {k:<28}{v:>20}  ║")
    print("╚"+"═"*52+"╝")
    return dict(F_mean=Fm,F_std=Fs,F_min=Fn,F_max=Fx,F_process=F_proc)


# ── Plot ───────────────────────────────────────────────────
def plot(loss_h, fid_h, F_per, fsum, t, dim, ep, lr, device, path):
    BG='#f7f6f2'; TEAL='#2a9d8f'; DARK='#264653'; MUTED='#6c757d'; OR='#e76f51'
    fig = plt.figure(figsize=(18,12), facecolor=BG)
    gs  = fig.add_gridspec(2,2,hspace=0.42,wspace=0.35,
                            left=0.07,right=0.96,top=0.88,bottom=0.06)
    epochs = np.arange(1,ep+1)

    def sty(ax):
        ax.set_facecolor('#ffffff')
        for s in ['top','right']: ax.spines[s].set_visible(False)
        for s in ['left','bottom']: ax.spines[s].set_color('#dcd9d5')
        ax.tick_params(colors=MUTED,labelsize=10)

    ax1=fig.add_subplot(gs[0,0]); sty(ax1)
    ax1.plot(epochs,loss_h,color=TEAL,lw=2)
    ax1.fill_between(epochs,loss_h,alpha=0.08,color=TEAL)
    ax1.set_title('Fidelity Loss  1−F̄',fontsize=12,fontweight='bold',color=DARK,pad=8)
    ax1.set_xlabel('Epoch',color=MUTED,fontsize=11)
    ax1.set_ylabel('1 − F̄',color=MUTED,fontsize=11)
    ax1.text(0.97,0.97,f'Final: {loss_h[-1]:.5f}',transform=ax1.transAxes,
             ha='right',va='top',fontsize=10,color=TEAL,
             bbox=dict(boxstyle='round,pad=0.3',facecolor='#fff',edgecolor=TEAL,lw=1))

    ax2=fig.add_subplot(gs[0,1]); sty(ax2)
    ax2.plot(epochs,fid_h,color=OR,lw=2)
    ax2.fill_between(epochs,fid_h,alpha=0.08,color=OR)
    ax2.axhline(1.0,color=DARK,lw=1,ls=':',alpha=0.4,label='F=1.0')
    ax2.set_ylim(0,1.05)
    ax2.set_title('Mean State Fidelity F̄',fontsize=12,fontweight='bold',color=DARK,pad=8)
    ax2.set_xlabel('Epoch',color=MUTED,fontsize=11)
    ax2.set_ylabel('F̄',color=MUTED,fontsize=11)
    ax2.text(0.97,0.07,f'Final: {fsum["F_mean"]:.5f}',transform=ax2.transAxes,
             ha='right',va='bottom',fontsize=10,color=OR,
             bbox=dict(boxstyle='round,pad=0.3',facecolor='#fff',edgecolor=OR,lw=1))
    ax2.legend(fontsize=9,framealpha=0)

    ax3=fig.add_subplot(gs[1,0]); sty(ax3)
    cols=[TEAL if f>=0.5 else OR for f in F_per]
    ax3.bar(np.arange(t),F_per,color=cols,alpha=0.8,width=0.8)
    ax3.axhline(fsum['F_mean'],color=DARK,lw=1.8,ls='--',
                label=f'Mean={fsum["F_mean"]:.4f}')
    ax3.axhline(0.5,color=OR,lw=1,ls=':',alpha=0.6,label='F=0.5')
    ax3.set_ylim(0,1.05)
    ax3.set_title(f'Per-Sample Fidelity  (t={t})',fontsize=12,
                  fontweight='bold',color=DARK,pad=8)
    ax3.set_xlabel('Sample j',color=MUTED,fontsize=11)
    ax3.set_ylabel('F_j',color=MUTED,fontsize=11)
    ax3.legend(fontsize=9,framealpha=0)

    ax4=fig.add_subplot(gs[1,1]); ax4.axis('off')
    td=[['Dataset','MPS'],['Device',str(device)],
        ['n_qubits',str(int(np.log2(dim)))],['dim',str(dim)],
        ['t_train',str(t)],['Epochs',str(ep)],
        ['Loss','1−|⟨φ|W|x⟩|²'],
        ['Final loss',f'{loss_h[-1]:.6f}'],
        ['Mean F̄',f'{fsum["F_mean"]:.6f}'],
        ['Std F',f'{fsum["F_std"]:.6f}'],
        ['Min F',f'{fsum["F_min"]:.6f}'],
        ['Max F',f'{fsum["F_max"]:.6f}'],
        ['Process F',f'{fsum["F_process"]:.6f}']]
    tbl=ax4.table(cellText=td,colLabels=['Metric','Value'],
                  cellLoc='left',loc='center',colWidths=[0.58,0.40])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor('#dcd9d5'); cell.set_linewidth(0.6)
        if r==0:
            cell.set_facecolor(DARK); cell.set_text_props(color='white',fontweight='bold')
        elif r>0 and 'F̄' in td[r-1][0]:
            cell.set_facecolor('#e8f4f3'); cell.set_text_props(color=TEAL,fontweight='bold')
        elif r>0 and 'loss' in td[r-1][0].lower():
            cell.set_facecolor('#fff3e8'); cell.set_text_props(color=OR,fontweight='bold')
        else:
            cell.set_facecolor('#ffffff' if r%2==0 else '#f9f8f5')
            cell.set_text_props(color=DARK)
    ax4.set_title('Fidelity Summary',fontsize=12,fontweight='bold',color=DARK,pad=10)

    fig.suptitle(
        f'Variational MPS Training & Fidelity  —  NFL Pipeline\n'
        f'Loss = 1 − (1/t)Σ|⟨φⱼ|W|xⱼ⟩|²     device={device}  t={t}  dim={dim}',
        fontsize=12,color=DARK,fontweight='bold',y=0.97)
    fig.savefig(path,dpi=150,bbox_inches='tight',facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    device = get_device(DEVICE)
    if TORCH_AVAILABLE and str(device) != "cpu_numpy":
        import torch
        print(f"Device: {device}")
        if str(device) == "cuda":
            print(f"  GPU : {torch.cuda.get_device_name(0)}")

    arch      = np.load(DATASET_PATH)
    X_all     = arch["X"]; Phi_all = arch["Phi"]; M = arch["M"]
    dim       = int(arch["dim"]); n_qubits = int(arch["n_qubits"])

    assert T_TRAIN <= len(X_all), \
        f"T_TRAIN={T_TRAIN} > available {len(X_all)}. Increase t_states in generate_M_and_dataset.py"

    X_train   = normalise(X_all[:T_TRAIN])
    Phi_train = normalise((M @ X_train.T).T)

    print(f"\nDataset: X={X_train.shape}  Phi={Phi_train.shape}  M={M.shape}")
    print("="*65)
    print("  Training  |  Loss = 1 − (1/t)Σ|⟨φⱼ|W|xⱼ⟩|²  |  Riemannian GD")
    print("="*65)

    W_final, loss_h, fid_h = train(
        X_train, Phi_train, M, dim,
        N_EPOCHS, LEARNING_RATE, LOG_EVERY, SEED, device
    )

    F_per, F_proc = compute_fidelities(W_final, X_train, Phi_train, M, dim)
    fsum = print_table(F_per, F_proc, loss_h[-1],
                       n_qubits, dim, T_TRAIN, N_EPOCHS, LEARNING_RATE, device)

    plot(loss_h, fid_h, F_per, fsum,
         T_TRAIN, dim, N_EPOCHS, LEARNING_RATE, device, OUT_PLOT)

    np.savez(OUT_RESULTS,
             W=W_final, loss_history=loss_h, fid_history=fid_h,
             F_per_sample=F_per, F_process=np.array(F_proc),
             final_loss=np.array(loss_h[-1]), t_train=np.array(T_TRAIN))
    print(f"Results saved: {OUT_RESULTS}")
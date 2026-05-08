import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def plot_data_distributions(data_loader):
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_max = 100
    X, Y = [], []
    for i, (x, y) in enumerate(tqdm(data_loader, total=n_max)):
        if i >= n_max:
            break
        X.append(x)
        Y.append(y)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    print(X.shape, Y.shape)

    plt.hist(X.flatten(), bins=100, alpha=0.5, label='X')
    plt.yscale('log')
    plt.show()
    plt.clf()

    n_labels = 1 if Y.ndim == 1 else Y.shape[1]

    if n_labels == len(data_loader.dataset.label_names):
        label_names = data_loader.dataset.label_names
    else:
        label_names = [str(i) for i in range(n_labels)]

    fig, ax = plt.subplots(figsize=(5, 4))
    df = pd.DataFrame(Y, columns=label_names)

    # Use seaborn's default style
    sns.set_style("whitegrid")

    # Create the histograms
    df.hist(figsize=(12, 10), bins=20, edgecolor='black', ax=ax)
    plt.tight_layout()
    plt.show()
    plt.clf()


def plot_event_animations(data_loader):
    images_, labels = next(iter(data_loader))
    images = images_.numpy()  # Convert to numpy for easier handling
    print(f"images_.shape = {images_.shape}, labels.shape = {labels.shape}")
    # Calculate percentiles once
    vmax = np.percentile(np.abs(images), 99)

    n_events = 10
    animations = []

    for event_idx in range(n_events):
        fig, ax = plt.subplots(figsize=(6, 5))
        
        def update(frame, idx=event_idx):  # Capture event_idx in closure
            ax.clear()
            img = images[idx, frame]
            print(f"img.shape = {img.shape}, min={img.min()}, max={img.max()}")
            ax.imshow(images[idx, frame], cmap='seismic', norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
            ax.set_xlabel('X Pixel')
            ax.set_ylabel('Y Pixel')
            labels_str = ""
            for i, col in enumerate(labels.columns):
                val = labels.iloc[idx][col]
                if isinstance(val, float):
                    labels_str += f"{col}={round(val, 2)}, "
                else:
                    labels_str += f"{col}={val}, "
                if i % 4 == 3:
                    labels_str += "\n"
            labels_str = labels_str.rstrip(', ')
            ax.set_title(f'Event {idx}, Time: {frame}\n{labels_str}', fontsize=10)
            return ax
        
        ani = FuncAnimation(fig, update, frames=images_.shape[1], interval=100, repeat=True)
        animations.append(ani)
        
        print(f"Animation {event_idx}:")
        display(HTML(ani.to_jshtml()))
        print("\n" + "="*80 + "\n")


def plot_acceptance_vs_pt(pt_true_signed: np.ndarray, pred_dict: dict[str, np.ndarray], outpath: Path):
    
    # use variable bin edges
    edges = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.15, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    edges = np.array(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    for model_name, preds in pred_dict.items():
        assert pt_true_signed.shape == preds.shape, "pt_true_signed and preds must have the same shape"
        acceptances = []
        for i_bin in range(len(edges) - 1):
            idx = (pt_true_signed >= edges[i_bin]) & (pt_true_signed < edges[i_bin + 1])
            n = int(idx.sum())
            if n == 0:
                acceptances = np.nan
            else:
                acceptance = float((preds[idx] == 2).sum()) / n
            print(f"Bin {i_bin}: true pt in [{edges[i_bin]:.2f}, {edges[i_bin + 1]:.2f}), acceptance: {acceptance:.4f} ({(preds[idx] == 2).sum()}/{n})")
            acceptances.append(acceptance)
        plt.plot(centers, acceptances, label=model_name, marker="o")
    # add horizontal lines and y-ticks at 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.35)

    plt.xlabel(r"true $p_T$ [GeV]")
    # plt.ylabel(r"acceptance as \textit{high-$p_T$}")
    plt.ylabel(r"acceptance as $\it{high}$-$p_T$")
    plt.legend(loc="best", fontsize=8, framealpha=0.2)
    plt.show()
    #plt.savefig(outpath, dpi=160)



def savefig(fig, path: Path, dpi: int = 160):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_confusion(cm_by_model: Dict[str, np.ndarray], outpath: Path):
    n = len(cm_by_model)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cm_by_model.items()):
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("pred")
        ax.set_ylabel("true")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["pos low", "neg low", "high"])
        ax.set_yticklabels(["pos low", "neg low", "high"])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, outpath)


def plot_roc(rocs: Dict[str, Tuple[np.ndarray, np.ndarray, float]], outpath: Path):
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 4.4), constrained_layout=True)
    for name, (fpr, tpr, auc) in rocs.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("High-$p_T$ ROC")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.2)
    plt.show()
    savefig(fig, outpath)


def plot_prob_vs_pt(pt_true_signed: np.ndarray, p_high_by_model: Dict[str, np.ndarray], outpath: Path,
                    xmin: float = -5.0, xmax: float = 5.0, nbins: int = 40):
    edges = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2), constrained_layout=True)

    for name, p_high in p_high_by_model.items():
        mean = np.full(nbins, np.nan, dtype=np.float64)
        err = np.full(nbins, np.nan, dtype=np.float64)
        for i in range(nbins):
            m = (pt_true_signed >= edges[i]) & (pt_true_signed < edges[i + 1])
            n = int(m.sum())
            if n == 0:
                continue
            ph = float(np.mean(p_high[m]))
            mean[i] = ph
            err[i] = np.sqrt(ph * (1.0 - ph) / n)
        ax.errorbar(centers, mean, yerr=err, fmt="o", markersize=3.0, capsize=2.0, label=name)

    ax.set_xlabel("true $p_T$ [GeV]")
    ax.set_ylabel(r"$P(\mathrm{high}\ p_T)$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8, framealpha=0.2)
    plt.show()
    savefig(fig, outpath)


def plot_acceptance_vs_pt(pt_true_signed: np.ndarray, pred_dict: dict[str, np.ndarray], outpath: Path):
    
    # use variable bin edges
    edges = [-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.8, -1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.15, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    edges = np.array(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    for model_name, preds in pred_dict.items():
        assert pt_true_signed.shape == preds.shape, "pt_true_signed and preds must have the same shape"
        acceptances = []
        for i_bin in range(len(edges) - 1):
            idx = (pt_true_signed >= edges[i_bin]) & (pt_true_signed < edges[i_bin + 1])
            n = int(idx.sum())
            if n == 0:
                acceptances = np.nan
            else:
                acceptance = float((preds[idx] == 2).sum()) / n
            print(f"Bin {i_bin}: true pt in [{edges[i_bin]:.2f}, {edges[i_bin + 1]:.2f}), acceptance: {acceptance:.4f} ({(preds[idx] == 2).sum()}/{n})")
            acceptances.append(acceptance)
        plt.plot(centers, acceptances, label=model_name, marker="o")
    # add horizontal lines and y-ticks at 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.35)

    plt.xlabel(r"true $p_T$ [GeV]")
    # plt.ylabel(r"acceptance as \textit{high-$p_T$}")
    plt.ylabel(r"acceptance as $\it{high}$-$p_T$")
    plt.legend(loc="best", fontsize=8, framealpha=0.2)
    plt.savefig(outpath, dpi=160)
    plt.show()

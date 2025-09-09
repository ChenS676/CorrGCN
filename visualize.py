# ---- utils: normalize & visualize adj ----
import torch
import matplotlib
matplotlib.use("Agg")  # 无界面环境更快更稳
import matplotlib.pyplot as plt

def to01(x: torch.Tensor) -> torch.Tensor:
    """将张量线性归一化到[0,1]，并处理 NaN/Inf；常量矩阵直接返回0矩阵。"""
    x = x.detach()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    minv = torch.min(x)
    maxv = torch.max(x)
    denom = (maxv - minv)
    if torch.isclose(denom, torch.tensor(0., device=x.device)):
        return torch.zeros_like(x)  # 常量矩阵
    return (x - minv) / (denom + 1e-12)

def binarize01(x: torch.Tensor, thr: float = 0.5, mode: str = "absolute") -> torch.Tensor:
    """
    对已归一化后的矩阵进行二值化：
      - mode='absolute'：直接用阈值 thr（0~1）
      - mode='percentile'：用分位数作为阈值（thr=0.9 => 上10%置1）
    """
    x01 = to01(x)
    if mode == "percentile":
        thr_val = torch.quantile(x01.flatten(), thr)
    else:
        thr_val = torch.as_tensor(thr, device=x01.device)
    return (x01 >= thr_val).to(x01.dtype)

def save_adj(x: torch.Tensor, path: str, cmap: str = "gray", colorbar: bool = False):
    """保存连续值的邻接矩阵图（已固定到[0,1]，更便于对比）"""
    a = to01(x).cpu().numpy()
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111)
    im = ax.imshow(a, vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def save_adj_binary(x: torch.Tensor, path: str, thr: float = 0.5, mode: str = "absolute", cmap: str = "gray"):
    """保存二值化（0/1）邻接矩阵图，便于直观看边有/无"""
    b = binarize01(x, thr=thr, mode=mode).cpu().numpy()
    fig = plt.figure(figsize=(6, 6), dpi=200)
    ax = fig.add_subplot(111)
    im = ax.imshow(b, vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    # 二值图通常不需要 colorbar；如需可打开：
    # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


import torch
import numpy as np
epoch = 1
adj1  = torch.load(f"pts_adj/adj1_{epoch}.pt", map_location="cpu")
adj2  = torch.load(f"pts_adj/adj2_{epoch}.pt", map_location="cpu")


def visual_adj(A_plot, path_name):
    A_plot = A_plot.detach().float().cpu().numpy()
    N = A_plot.shape[0]
    # deg = adj1.sum(axis=1)  # out-degree (row-sum)
    # order = np.argsort(-deg)  # descending
    # A_plot = adj1[order][:, order]

    # Figure size scales mildly with N (cap to keep files manageable)
    w = min(12, 1.0 + 0.25 * N)
    h = min(10, 1.0 + 0.25 * N)
    plt.figure(figsize=(w, h))
    im = plt.imshow(A_plot, aspect='equal', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'Adjacency Heatmap {path_name[:-4]}(N={N})')
    plt.xlabel('j')
    plt.ylabel('i')
    plt.tight_layout()
    plt.savefig(path_name, dpi=300)
    plt.close()


visual_adj(adj1, f'adj1_{epoch}_nn.pdf')
visual_adj(adj2, f'adj2_{epoch}_nn.pdf')
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


import matplotlib.pyplot as plt
import pandas as pd

# Data manually extracted from the LaTeX table (MAE only for brevity, RMSE similarly)
data = {
    ("Diffusion GCN", "Easy"): [0.1953, 0.2521, 0.3051, 0.3579],
    ("Diffusion GCN", "Medium"): [0.2960, 0.3734, 0.4346, 0.5002],
    ("Diffusion GCN", "Hard"): [0.3426, 0.4187, 0.4718, 0.5090],
    ("Diffusion GCN", "Very Hard"): [0.4194, 0.4712, 0.4911, 0.5044],

    ("Power-law Filters", "Easy"): [0.1917, 0.2477, 0.2984, 0.3514],
    ("Power-law Filters", "Medium"): [0.2907, 0.3703, 0.4270, 0.4904],
    ("Power-law Filters", "Hard"): [0.3364, 0.4162, 0.4633, 0.4977],
    ("Power-law Filters", "Very Hard"): [0.4098, 0.4637, 0.4821, 0.4928],

    ("Dual-Graph Prop", "Easy"): [0.1932, 0.2491, 0.3000, 0.3518],
    ("Dual-Graph Prop", "Medium"): [0.2922, 0.3689, 0.4288, 0.4890],
    ("Dual-Graph Prop", "Hard"): [0.3382, 0.4140, 0.4671, 0.5031],
    ("Dual-Graph Prop", "Very Hard"): [0.4135, 0.4652, 0.4859, 0.4967],

    ("Chebyshev Conv", "Easy"): [0.1940, 0.2508, 0.3021, 0.3531],
    ("Chebyshev Conv", "Medium"): [0.2945, 0.3716, 0.4302, 0.4927],
    ("Chebyshev Conv", "Hard"): [0.3404, 0.4176, 0.4698, 0.5063],
    ("Chebyshev Conv", "Very Hard"): [0.4160, 0.4683, 0.4881, 0.4989],

    ("PowerMixDual", "Easy"): [0.1901, 0.2464, 0.2970, 0.3500],
    ("PowerMixDual", "Medium"): [0.2892, 0.3687, 0.4253, 0.4879],
    ("PowerMixDual", "Hard"): [0.3351, 0.4147, 0.4612, 0.4957],
    ("PowerMixDual", "Very Hard"): [0.4083, 0.4619, 0.4800, 0.4906],
}



horizons = [6]
difficulties = ["Easy", "Medium", "Hard", "Very Hard"]
methods = ["Diffusion GCN", "Power-law Filters", "Dual-Graph Prop", "Chebyshev Conv", "PowerMixDual"]

# Plot: one subplot for each horizon
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axes = axes.flatten()

x = np.arange(len(difficulties))  # [0,1,2,3]
linestyles = ["-", "--", "-.", ":"]  # one style per horizon

plt.figure(figsize=(10, 6))

for m_idx, method in enumerate(methods):
    for h_idx, h in enumerate(horizons):
        values = [data[(method, d)][h_idx] for d in difficulties]
        plt.plot(
            x, values, 
            marker="o", 
            linestyle=linestyles[h_idx], 
            label=f"{method} (H{h})"
        )

plt.xticks(x, difficulties)
plt.ylabel("MAE")
plt.xlabel("Difficulty Level")
plt.title("Synthetic Dataset: MAE across Difficulty Levels and Horizons")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # place legend outside
plt.tight_layout()

# Save & show
plt.savefig("synthetic_performance.pdf", dpi=300, bbox_inches="tight")
plt.show()
# visual_adj.py
# -*- coding: utf-8 -*-
"""
可视化训练时保存的邻接矩阵 (adj1, adj2) 以及对比图 (Pearson, adj1-adj2)。
会把所有生成的图片保存到指定目录。
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 无GUI环境下安全
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_matrix(M, title, path, cmap="coolwarm", vmin=None, vmax=None):
    """绘制矩阵并保存"""
    plt.figure(figsize=(6, 6))
    im = plt.imshow(M, cmap=cmap, aspect="equal", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_pearson_from_data(npy_path, save_path=None):
    """
    从 npz/npy 数据文件里计算 Pearson 相关性矩阵。
    npz 文件应包含 'x' 键，形状 [B, T, N, C] 或 [B, T, N]。
    """
    arr = np.load(npy_path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        x = arr["x"]
    else:
        x = arr
    if x.ndim == 4:   # [B, T, N, C]
        series = x[:, :, :, 0].reshape(-1, x.shape[2])  # 用第一个特征
    elif x.ndim == 3: # [B, T, N]
        series = x.reshape(-1, x.shape[2])
    else:
        raise ValueError(f"Unexpected data shape {x.shape}")

    corr = np.corrcoef(series.T)
    if save_path:
        plot_matrix(corr, "Correlation (Pearson)", save_path, cmap="coolwarm", vmin=-1, vmax=1)
    return corr


def visualize_epoch(save_dir, epoch, data_file=None):
    """
    可视化某个 epoch 的 adj1/adj2 及其对比。
    Args:
        save_dir: 保存路径
        epoch: epoch 号 (int)
        data_file: 可选, npz/npy 文件路径，用于计算 Pearson corr
    """
    ensure_dir(save_dir)

    # 加载 adj1, adj2
    adj1_path = os.path.join(save_dir, f"adj1_epoch{epoch}.npy")
    adj2_path = os.path.join(save_dir, f"adj2_epoch{epoch}.npy")
    adj1 = np.load(adj1_path)
    adj2 = np.load(adj2_path)

    # 画 adj1, adj2
    plot_matrix(adj1, f"Adj1 (epoch {epoch})", os.path.join(save_dir, f"adj1_epoch{epoch}.png"))
    plot_matrix(adj2, f"Adj2 (epoch {epoch})", os.path.join(save_dir, f"adj2_epoch{epoch}.png"))

    # 画 adj1 - adj2
    diff = adj1 - adj2
    plot_matrix(diff, f"Adj1 - Adj2 (epoch {epoch})", os.path.join(save_dir, f"adj_diff_epoch{epoch}.png"),
                cmap="bwr", vmin=-1, vmax=1)

    # 可选: 画 Pearson corr
    if data_file is not None:
        corr = compute_pearson_from_data(data_file, save_path=os.path.join(save_dir, f"corr_epoch{epoch}.png"))
        plot_matrix(adj1 - corr, f"Adj1 - Corr (epoch {epoch})", os.path.join(save_dir, f"adj1_minus_corr_epoch{epoch}.png"),
                    cmap="bwr", vmin=-1, vmax=1)
        plot_matrix(adj2 - corr, f"Adj2 - Corr (epoch {epoch})", os.path.join(save_dir, f"adj2_minus_corr_epoch{epoch}.png"),
                    cmap="bwr", vmin=-1, vmax=1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_dir", type=str, default="./garage/france", help="存放 adj1/adj2 的目录")
    ap.add_argument("--epoch", type=int, default=1, help="要可视化的 epoch")
    ap.add_argument("--data_file", type=str, default=None, help="数据文件路径 (.npz/.npy)，用于计算 Pearson corr")
    args = ap.parse_args()

    visualize_epoch(args.save_dir, args.epoch, data_file=args.data_file)
    print(f"✅ 可视化完成，图片已保存到 {args.save_dir}")

# train_adj.py
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch

import util
from engine import trainer

# ====== W&B (optional) ======
try:
    import wandb
except ImportError:
    wandb = None

# ============================
# CLI
# ============================
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data', type=str, default='data/METR-LA')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl')
parser.add_argument('--adjtype', type=str, default='doubletransition')
parser.add_argument('--gcn_bool', action='store_true')
parser.add_argument('--aptonly', action='store_true')
parser.add_argument('--addaptadj', action='store_true')
parser.add_argument('--randomadj', action='store_true')
parser.add_argument('--seq_length', type=int, default=96)
parser.add_argument('--pred_length', type=int, default=12)
parser.add_argument('--nhid', type=int, default=32)
parser.add_argument('--in_dim', type=int, default=2)
parser.add_argument('--num_nodes', type=int, default=207)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--save', type=str, default='./garage/metr')
parser.add_argument('--expid', type=int, default=1)

# Power / MixProp / Cheby / PowerMix
parser.add_argument("--use_power", action="store_true")
parser.add_argument("--use_cheby", action="store_true")
parser.add_argument("--use_mixprop", action="store_true")
parser.add_argument("--use_powermix", action="store_true")

parser.add_argument("--diag_mode", type=str, default="self_and_neighbor",
                    choices=["self_and_neighbor", "neighbor"])

parser.add_argument("--power_order", type=int, default=2)
parser.add_argument("--power_init", type=str, default="plain",
                    choices=["plain", "decay", "softmax"])

parser.add_argument("--cheby_k", type=int, default=3)

parser.add_argument("--mixprop_k", type=int, default=3)
parser.add_argument("--adj_dropout", type=float, default=0.1)
parser.add_argument("--adj_temp", type=float, default=1.0)

parser.add_argument("--powermix_k", type=int, default=3)
parser.add_argument("--powermix_dropout", type=float, default=0.3)
parser.add_argument("--powermix_temp", type=float, default=1.0)

args = parser.parse_args()

# ============================
# Dataset auto-config
# ============================
def configure_dataset_params(args):
    data_path = args.data.upper()

    if 'FRANCE' in data_path:
        args.num_nodes = 10
        args.adjdata = 'data/sensor_graph/adj_mx_france.pkl'
        args.save = './garage/france/'
    elif 'GERMANY' in data_path:
        args.num_nodes = 16
        args.adjdata = 'data/sensor_graph/adj_mx_germany.pkl'
        args.save = './garage/germany/'
    elif 'METR' in data_path:
        args.num_nodes = 207
        args.adjdata = 'data/sensor_graph/adj_mx.pkl'
        args.save = './garage/metr/'
    elif 'BAY' in data_path:
        args.num_nodes = 325
        args.adjdata = 'data/sensor_graph/adj_mx_bay.pkl'
        args.save = './garage/bay/'
    elif 'SOLAR' in data_path:
        args.num_nodes = 137
        args.adjdata = 'data/sensor_graph/adj_mx_solar.pkl'
        args.save = './garage/solar/'
    elif 'ELECTRICITY' in data_path:
        args.num_nodes = 321
        args.adjdata = 'data/sensor_graph/adj_mx_electricity.pkl'
        args.save = './garage/electricity/'
    else:
        print(f"未识别的数据集: {data_path}")

    os.makedirs(args.save, exist_ok=True)
    return args

args = configure_dataset_params(args)

# ============================
# Main training loop
# ============================
def main(args):
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]
    aptinit = None
    if args.aptonly:
        supports = None

    engine = trainer(
        scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
        args.learning_rate, args.weight_decay, args.device, supports, args.gcn_bool,
        args.addaptadj, aptinit, pred_length=args.pred_length,
        diag_mode=args.diag_mode,
        use_power=args.use_power, power_order=args.power_order, power_init=args.power_init,
        use_cheby=args.use_cheby, cheby_k=args.cheby_k,
        use_mixprop=args.use_mixprop, mixprop_k=args.mixprop_k,
        adj_dropout=args.adj_dropout, adj_temp=args.adj_temp,
        use_powermix=args.use_powermix, powermix_k=args.powermix_k,
        powermix_dropout=args.powermix_dropout, powermix_temp=args.powermix_temp
    )

    print("Start training & saving adj ...")

    his_loss = []
    for epoch in range(1, args.epochs+1):
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            _ = engine.train(trainx, trainy[:, 0, :, :args.pred_length])

        # --- 保存 adj1 / adj2 / powercoef ---
        save_dir = args.save
        os.makedirs(save_dir, exist_ok=True)

        try:
            m = engine.model.powermix_convs[0]  # 第一个 PowerMix 模块
            np.save(os.path.join(save_dir, f"adj1_epoch{epoch}.npy"),
                    m.adj_1.detach().cpu().numpy())
            np.save(os.path.join(save_dir, f"adj2_epoch{epoch}.npy"),
                    m.adj_2.detach().cpu().numpy())
            np.save(os.path.join(save_dir, f"powercoef_epoch{epoch}.npy"),
                    m.power_coef.detach().cpu().numpy())
            print(f"✅ Saved adj1/adj2/powercoef at epoch {epoch}")
        except Exception as e:
            print(f"⚠️ Failed to save adj at epoch {epoch}: {e}")

if __name__ == "__main__":
    main(args)

# python train_adj.py --data data/FRANCE --device cuda:0 --epochs 5     --use_powermix --powermix_k 2 --powermix_dropout 0.0 --powermix_temp 1.0     --power_order 2 --power_init decay --save ./garage/france
# python visual_adj.py --save_dir ./garage/france --epoch 5 --data_file data/FRANCE/train.npz
"""python train.py   --data data/FRANCE --device cuda:0 --batch_size 64 --epochs 5 --seq_length 96 --pred_length 12 --learning_rate 0.0005   --dropout 0 --nhid 64 --weight_decay 0.0001   --print_every 50 --gcn_bool --addaptadj --randomadj --adjtype doubletransition --diag_mode neighbor   --use_powermix --powermix_k 2 --powermix_dropout 0 --powermix_temp 1.0 --power_order 2 --power_init decay
"""
import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import os
import csv
import pandas as pd

# === NEW: 需要用到的函数 ===
import torch.nn.functional as F

# === NEW END ===

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='data/METR-LA', help='data path')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl', help='adj data path')
parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
parser.add_argument('--gcn_bool',action='store_true', help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true', help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true', help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true', help='whether random initialize adaptive adj')
parser.add_argument('--seq_length', type=int, default=96, help='input sequence length')      ######   
parser.add_argument('--pred_length', type=int, default=12, help='prediction length (output sequence length)')  ######
parser.add_argument('--nhid', type=int, default=32, help='')
parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
parser.add_argument('--num_nodes', type=int, default=207, help='number of nodes')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='')
parser.add_argument('--print_every', type=int, default=50, help='')
#parser.add_argument('--seed', type=int, default=99, help='random seed')
parser.add_argument('--save', type=str, default='./garage/metr', help='save path')
parser.add_argument('--expid', type=int, default=1, help='experiment id')
parser.add_argument('--run_multiple_experiments',action='store_true', help='run experiments with different sequence lengths')

# === NEW: earlystop ===
parser.add_argument('--early_stop_patience', type=int, default=10,
                    help='若验证集 loss 连续 patience 个 epoch 无显著下降则提前停止')
parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                    help='判定“显著下降”的阈值 (new_best < best - min_delta)')


# === NEW: 增强开关 + 少量可调参数 ===
parser.add_argument('--enhance', type=str, default='none',
                    choices=['none', 'series', 'graph', 'both'],
                    help='enhancement module: series (time), graph (spectral), both, or none')
parser.add_argument('--series_kernel', type=int, default=25,
                    help='kernel size for series decomposition (odd number recommended)')
parser.add_argument('--graph_mode', type=str, default='lowpass',
                    choices=['lowpass', 'highpass', 'none'],
                    help='graph filtering mode on inputs')
parser.add_argument('--graph_alpha', type=float, default=0.5,
                    help='graph filter strength alpha')

# ==== GraphWaveNet / Power / MixProp / Cheby / PowerMix 相关可选项 ====
parser.add_argument("--use_power", action="store_true", help="启用 PowerLaw 传播")
parser.add_argument("--use_cheby", action="store_true", help="启用 Chebyshev 传播")
parser.add_argument("--use_mixprop", action="store_true", help="启用 MixPropDual")
parser.add_argument("--use_powermix", action="store_true", help="启用 PowerMixDual")

# 共享/结构相关
parser.add_argument("--diag_mode", type=str, default="self_and_neighbor",
                    choices=["self_and_neighbor", "neighbor"], help="对角连边模式")

# PowerLaw 专用
parser.add_argument("--power_order", type=int, default=2, help="PowerLaw 最大阶数")
parser.add_argument("--power_init", type=str, default="plain",
                    choices=["plain", "decay", "softmax"], help="幂律系数初始化策略")

# Chebyshev
parser.add_argument("--cheby_k", type=int, default=3, help="Chebyshev K 阶")

# MixPropDual
parser.add_argument("--mixprop_k", type=int, default=3, help="MixPropDual 递推步长")
parser.add_argument("--adj_dropout", type=float, default=0.1, help="邻接 dropout")
parser.add_argument("--adj_temp", type=float, default=1.0, help="邻接温度")

# PowerMixDual
parser.add_argument("--powermix_k", type=int, default=3, help="PowerMix 递推步长")
parser.add_argument("--powermix_dropout", type=float, default=0.3, help="PowerMix A-dropout")
parser.add_argument("--powermix_temp", type=float, default=1.0, help="PowerMix 温度")
args = parser.parse_args()

# Auto-detect dataset type and configure parameters accordingly

def configure_dataset_params(args):
    """
    Configure dataset-specific parameters based on data path
    """
    data_path = args.data.upper()

    if 'FRANCE' in data_path:
        args.num_nodes = 10
        args.adjdata = 'data/sensor_graph/adj_mx_france.pkl'
        args.save = './garage/france/'
        print(f"检测到原始France数据集")
        print(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    elif 'GERMANY' in data_path:
        args.num_nodes = 16
        args.adjdata = 'data/sensor_graph/adj_mx_germany.pkl'
        args.save = './garage/germany/'
        print(f"检测到Germany数据集")
        print(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    elif 'METR' in data_path:
        args.num_nodes = 207
        args.adjdata = 'data/sensor_graph/adj_mx.pkl'
        args.save = './garage/metr/'
        print(f"检测到METR-LA数据集")
        print(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    elif 'BAY' in data_path:
        args.num_nodes = 325
        args.adjdata = 'data/sensor_graph/adj_mx_bay.pkl'
        args.save = './garage/bay/'
        print(f"检测到PEMS-BAY数据集")
        print(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # ========= 新增：四套合成数据 =========
    elif 'SYNTHETIC_EASY' in data_path:
        args.num_nodes = 12
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_easy.pkl'
        args.save = './garage/synth_easy/'
        print(f"检测到合成数据集：SYNTHETIC_EASY")
        print(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    elif 'SYNTHETIC_MEDIUM' in data_path:
        args.num_nodes = 12
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_medium.pkl'
        args.save = './garage/synth_medium/'
        print(f"检测到合成数据集：SYNTHETIC_MEDIUM")
        print(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    elif 'SYNTHETIC_HARD' in data_path:
        args.num_nodes = 12
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_hard.pkl'
        args.save = './garage/synth_hard/'
        print(f"检测到合成数据集：SYNTHETIC_HARD")
        print(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    elif 'SYNTHETIC_VERY_HARD' in data_path:
        args.num_nodes = 12
        args.adjdata = 'data/sensor_graph/adj_mx_synthetic_very_hard.pkl'
        args.save = './garage/synth_very_hard/'
        print(f"检测到合成数据集：SYNTHETIC_VERY_HARD")
        print(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")
        
    # ===================================
    
    elif 'SOLAR' in data_path:
        args.num_nodes = 137
        args.adjdata = 'data/sensor_graph/adj_mx_solar.pkl'
        args.save = './garage/solar/'
        print(f"检测到Solar数据集")
        print(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    elif 'ELECTRICITY' in data_path:
        args.num_nodes = 321
        args.adjdata = 'data/sensor_graph/adj_mx_electricity.pkl'
        args.save = './garage/electricity/'
        print(f"检测到Electricity数据集")
        print(f"配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")

    # ===================================

    else:
        print(f"未识别的数据集: {data_path}")
        print(f"使用默认配置: 节点数={args.num_nodes}")

    os.makedirs(args.save, exist_ok=True)
    return args
    


# Configure parameters based on dataset
args = configure_dataset_params(args)

def run_experiments_with_different_seq_lengths():
    """
    Run experiments with different seq_length values and save results to CSV
    """
    seq_lengths = [6, 12]
    results = []
    
    print("Starting experiments with different sequence lengths...")
    print(f"Sequence lengths to test: {seq_lengths}")
    print("Note: pred_length will be set equal to seq_length for each experiment")
    
    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Starting experiment with seq_length = {seq_len}, pred_length = {seq_len}")
        print(f"{'='*60}")
        
        args.seq_length = seq_len
        args.pred_length = seq_len
        args.expid = seq_len
        
        print(f"为 seq_length={seq_len}, pred_length={seq_len} 生成数据...")
        generate_data_for_seq_length(seq_len, seq_len)
        
        experiment_start_time = time.time()
        result = main_experiment()
        experiment_end_time = time.time()
        
        result['seq_length'] = seq_len
        result['pred_length'] = seq_len
        result['total_experiment_time'] = experiment_end_time - experiment_start_time
        results.append(result)
        
        print(f"Completed experiment with seq_length = {seq_len}, pred_length = {seq_len}")
        print(f"Experiment time: {experiment_end_time - experiment_start_time:.4f} seconds")
    
    save_results_to_csv(results)
    print(f"\nAll experiments completed! Results saved to 'experiment_results.csv'")
    return results

def generate_data_for_seq_length(seq_length, pred_length):
    """
    Generate dataset for specific sequence length based on the dataset type
    """
    import subprocess
    import sys
    
    data_path = args.data.upper()
    
    if 'FRANCE' in data_path:
        dataset_name = 'FRANCE'
        process_script = 'process_france_with_dataloader.py'
        data_file = f'data/FRANCE/train.npz'
    elif 'GERMANY' in data_path:
        dataset_name = 'GERMANY'
        process_script = 'process_germany_with_dataloader.py'
        data_file = f'data/GERMANY/train.npz'
    else:
        print(f"新的数据集类型: {data_path}")
        print("使用现有数据")
        return
    
    regenerate = True
    
    if os.path.exists(data_file):
        try:
            data = np.load(data_file)
            existing_seq_len = data['x'].shape[1]
            if existing_seq_len == seq_length:
                print(f"{dataset_name}数据已存在且序列长度匹配 (seq_length={seq_length})")
                regenerate = False
            else:
                print(f"现有{dataset_name}数据序列长度不匹配 ({existing_seq_len} != {seq_length})，重新生成...")
        except Exception as e:
            print(f"检查现有{dataset_name}数据时出错: {e}")
    
    if regenerate:
        print(f"生成{dataset_name}数据: seq_length={seq_length}, pred_length={pred_length}...")
        cmd = [
            sys.executable, process_script,
            '--step', 'process',
            '--seq_length', str(seq_length),
            '--pred_length', str(pred_length)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"{dataset_name}数据生成完成")
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-5:]:
                if line.strip():
                    print(f"  {line}")
        except subprocess.CalledProcessError as e:
            print(f"{dataset_name}数据生成失败: {e}")
            print(f"错误输出: {e.stderr}")
            raise

def main_experiment():
    """
    Modified main function that returns test results instead of just printing them
    """
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

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

    print("start training...",flush=True)

    his_loss =[]
    val_time = []
    train_time = []
    
    best_val = float('inf') ###
    epochs_no_improve = 0 ###
    
    for i in range(1, args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):

            trainx = torch.Tensor(x).to(device)          # [B, T, N, C]
            trainx = trainx.transpose(1, 3)              # -> [B, C, N, T]

            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)              # [B, C, N, T]
            # metrics = engine.train(trainx, trainy[:,0,:,:])  # 目标仍用原第一特征
            metrics = engine.train(trainx, trainy[:, 0, :, :args.pred_length])

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
        t2 = time.time()
        train_time.append(t2-t1)

        # validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):

            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)  # [B, C, N, T]

            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)

            metrics = engine.eval(testx, testy[:, 0, :, :args.pred_length])

            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i) + "_" + str(round(mvalid_loss, 2))+".pth")

        # === EARLY STOPPING ===
        if mvalid_loss < best_val - args.early_stop_min_delta:
            best_val = mvalid_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {i}. "
                f"Best valid loss: {best_val:.4f} (epoch {np.argmin(his_loss)+1}).")
            break
        # === END EARLY STOPPING ===

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    outputs = []
    #realy = torch.Tensor(dataloader['y_test']).to(device)
    #realy = realy.transpose(1,3)[:,0,:,:]
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:args.pred_length]


    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        try:
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)  # [B, C, N, T]

            with torch.no_grad():
                preds = engine.model(testx).transpose(1,3)
            if len(preds.shape) == 4:
                preds = preds[:, 0, :, :]
            outputs.append(preds)

            del testx, preds
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                if args.batch_size > 1:
                    args.batch_size = args.batch_size // 2
                    print(f"显存不足，尝试减小 batch_size 至 {args.batch_size}")
                print(f"第 {iter} 批次推理时显存不足，跳过该批次")
                torch.cuda.empty_cache()
            else:
                raise e
        
    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    
    amae = []
    amape = []
    armse = []
    horizon_results = []
    
    for i in range(args.pred_length):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        metrics = util.metric(pred,real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        
        horizon_results.append({
            'horizon': i+1,
            'mae': metrics[0],
            'mape': metrics[1],
            'rmse': metrics[2]
        })

    avg_mae = np.mean(amae)
    avg_mape = np.mean(amape)
    avg_rmse = np.mean(armse)
    
    log = 'On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(args.pred_length, avg_mae, avg_mape, avg_rmse))
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")

    return {
        'valid_loss_best': his_loss[bestid],
        'avg_train_time_per_epoch': np.mean(train_time),
        'avg_inference_time': np.mean(val_time),
        'test_mae_avg': avg_mae,
        'test_mape_avg': avg_mape,
        'test_rmse_avg': avg_rmse,
        'horizon_results': horizon_results
    }

def save_results_to_csv(results):
    """
    Save experiment results to CSV file
    """
    csv_data = []
    for result in results:
        row = {
            'seq_length': result['seq_length'],
            'pred_length': result['pred_length'],
            'total_experiment_time': result['total_experiment_time'],
            'valid_loss_best': result['valid_loss_best'],
            'avg_train_time_per_epoch': result['avg_train_time_per_epoch'],
            'avg_inference_time': result['avg_inference_time'],
            'test_mae_avg': result['test_mae_avg'],
            'test_mape_avg': result['test_mape_avg'],
            'test_rmse_avg': result['test_rmse_avg']
        }
        for horizon_result in result['horizon_results']:
            row[f'horizon_{horizon_result["horizon"]}_mae'] = horizon_result['mae']
            row[f'horizon_{horizon_result["horizon"]}_mape'] = horizon_result['mape']
            row[f'horizon_{horizon_result["horizon"]}_rmse'] = horizon_result['rmse']
        csv_data.append(row)
    df = pd.DataFrame(csv_data)
    df.to_csv('experiment_results.csv', index=False)
    print(f"\nResults saved to 'experiment_results.csv'")
    print(f"Columns saved: {list(df.columns)}")
    
    summary_data = []
    for result in results:
        summary_data.append({
            'seq_length': result['seq_length'],
            'pred_length': result['pred_length'],
            'total_time_hours': result['total_experiment_time'] / 3600,
            'valid_loss': result['valid_loss_best'],
            'test_mae': result['test_mae_avg'],
            'test_mape': result['test_mape_avg'],
            'test_rmse': result['test_rmse_avg']
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('experiment_summary.csv', index=False)
    print(f"Summary saved to 'experiment_summary.csv'")

if __name__ == "__main__":
    total_start_time = time.time()
    
    if args.run_multiple_experiments:
        print("运行多个序列长度实验...")
        results = run_experiments_with_different_seq_lengths()
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(f"\nAll experiments completed!")
        print(f"Total time for all experiments: {total_time:.4f} seconds ({total_time/3600:.2f} hours)")
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        for result in results:
            print(f"Seq Length {result['seq_length']:2d}, Pred Length {result['pred_length']:2d}: MAE={result['test_mae_avg']:.4f}, MAPE={result['test_mape_avg']:.4f}, RMSE={result['test_rmse_avg']:.4f}")
    else:
        print("运行单个实验...")
        print(f"配置: seq_length={args.seq_length}, pred_length={args.pred_length}, enhance={args.enhance}")
        print(f"为 seq_length={args.seq_length}, pred_length={args.pred_length} 生成数据...")
        generate_data_for_seq_length(args.seq_length, args.pred_length)
        result = main_experiment()
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        print(f"\nExperiment completed!")
        print(f"Total time: {total_time:.4f} seconds ({total_time/3600:.2f} hours)")
        print(f"Results: MAE={result['test_mae_avg']:.4f}, MAPE={result['test_mape_avg']:.4f}, RMSE={result['test_rmse_avg']:.4f}")



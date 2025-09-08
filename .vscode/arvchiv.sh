

# 原始（无增强）：
# python train.py --data data/FRANCE --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 10


# 只开时间域增强（series_decomp）：
# python train.py --data data/FRANCE --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 10 --enhance series --series_kernel 25 

# 只开图域增强（graph filter）：
# python train.py --data data/FRANCE --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 10 --enhance graph --graph_mode lowpass --graph_alpha 0.5

# 两者同时：
# python train.py --data data/FRANCE --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 10 --enhance both --series_kernel 25 --graph_mode lowpass --graph_alpha 0.5


### nvidia-smi

### srun -p 4090 --pty --gpus 1 -t 2:00:00 bash -i
### srun -p 4090 --pty --gpus 2 -t 1:00:00 bash -i
### srun -p 4090 --pty --gpus 1 -t 12:00:00 bash -i

# srun -p 4090 --nodelist=aifb-websci-gpunode1 --gres=gpu:1 -t 2:00:00 --pty bash -i
# srun -p 4090 --nodelist=aifb-websci-gpunode2 --gres=gpu:1 -t 2:00:00 --pty bash -i


### conda activate Energy-TSF

### cd /mnt/webscistorage/cc7738/ws_joella/EnergyTSF
### cd /mnt/webscistorage/cc7738/ws_joella/EnergyTSF_latest

### squeue -u $USER


# wandb login

# cd /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin
# python train.py --data data/FRANCE --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 10



'''
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

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=96,help='input sequence length')         
parser.add_argument('--pred_length',type=int,default=96,help='prediction length (output sequence length)')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--run_multiple_experiments',action='store_true',help='run experiments with different sequence lengths')

args = parser.parse_args()

# Auto-detect dataset type and configure parameters accordingly
def configure_dataset_params(args):
    """
    Configure dataset-specific parameters based on data path
    """
    data_path = args.data.upper()
    
    if 'FRANCE' in data_path:
        # Original France dataset parameters
        args.num_nodes = 10
        args.adjdata = 'data/sensor_graph/adj_mx_france.pkl'
        args.save = './garage/france/'
        print(f"✅ 检测到原始France数据集")
        print(f"📊 配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")
    elif 'GERMANY' in data_path:
        # Germany dataset parameters
        args.num_nodes = 16
        args.adjdata = 'data/sensor_graph/adj_mx_germany.pkl'
        args.save = './garage/germany/'
        print(f"✅ 检测到Germany数据集")
        print(f"📊 配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")
    elif 'METR' in data_path:
        # METR-LA dataset parameters
        args.num_nodes = 207
        args.adjdata = 'data/sensor_graph/adj_mx.pkl'
        args.save = './garage/metr/'
        print(f"✅ 检测到METR-LA数据集")
        print(f"📊 配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")
    elif 'BAY' in data_path:
        # PEMS-BAY dataset parameters  
        args.num_nodes = 325
        args.adjdata = 'data/sensor_graph/adj_mx_bay.pkl'
        args.save = './garage/bay/'
        print(f"✅ 检测到PEMS-BAY数据集")
        print(f"📊 配置参数: 节点数={args.num_nodes}, 邻接矩阵={args.adjdata}")



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
    # seq_lengths = [6, 12, 48, 96]
    seq_lengths = [6, 12]
    results = []
    
    print("Starting experiments with different sequence lengths...")
    print(f"Sequence lengths to test: {seq_lengths}")
    print("Note: pred_length will be set equal to seq_length for each experiment")
    
    for seq_len in seq_lengths:
        print(f"\n{'='*60}")
        print(f"Starting experiment with seq_length = {seq_len}, pred_length = {seq_len}")
        print(f"{'='*60}")
        
        # Modify args for this experiment - both seq_length and pred_length should be equal
        args.seq_length = seq_len
        args.pred_length = seq_len  # Set pred_length equal to seq_length
        args.expid = seq_len  # Use seq_length as experiment id
        
        # Generate data for this specific seq_length and pred_length
        print(f"🔄 为 seq_length={seq_len}, pred_length={seq_len} 生成数据...")
        generate_data_for_seq_length(seq_len, seq_len)  # Both parameters are equal
        
        # Run the main training function
        experiment_start_time = time.time()
        result = main_experiment()
        experiment_end_time = time.time()
        
        # Store results
        result['seq_length'] = seq_len
        result['pred_length'] = seq_len  # Also store pred_length
        result['total_experiment_time'] = experiment_end_time - experiment_start_time
        results.append(result)
        
        print(f"Completed experiment with seq_length = {seq_len}, pred_length = {seq_len}")
        print(f"Experiment time: {experiment_end_time - experiment_start_time:.4f} seconds")
    
    # Save results to CSV
    save_results_to_csv(results)
    print(f"\nAll experiments completed! Results saved to 'experiment_results.csv'")
    
    return results




########



#######
def generate_data_for_seq_length(seq_length, pred_length):
    """
    Generate dataset for specific sequence length based on the dataset type
    """
    import subprocess
    import sys
    
    # Auto-detect dataset type based on args.data
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
        print(f"⚠️  不支持的数据集类型: {data_path}")
        print("跳过数据生成，使用现有数据...")
        return
    
    # Check if data already exists for this configuration
    regenerate = True
    
    if os.path.exists(data_file):
        # Check if existing data has correct seq_length
        try:
            data = np.load(data_file)
            existing_seq_len = data['x'].shape[1]  # Shape: [samples, seq_len, nodes, features]
            if existing_seq_len == seq_length:
                print(f"✅ {dataset_name}数据已存在且序列长度匹配 (seq_length={seq_length})")
                regenerate = False
            else:
                print(f"🔄 现有{dataset_name}数据序列长度不匹配 ({existing_seq_len} != {seq_length})，重新生成...")
        except Exception as e:
            print(f"⚠️  检查现有{dataset_name}数据时出错: {e}")
    
    if regenerate:
        print(f"🔄 生成{dataset_name}数据: seq_length={seq_length}, pred_length={pred_length}...")
        cmd = [
            sys.executable, process_script,
            '--step', 'process',
            '--seq_length', str(seq_length),
            '--pred_length', str(pred_length)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ {dataset_name}数据生成完成")
            # Print last few lines of output for verification
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-5:]:
                if line.strip():
                    print(f"  {line}")
        except subprocess.CalledProcessError as e:
            print(f"❌ {dataset_name}数据生成失败: {e}")
            print(f"错误输出: {e.stderr}")
            raise

def main_experiment():
    """
    Modified main function that returns test results instead of just printing them
    """
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
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

    if args.aptonly:
        supports = None

    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, pred_length=args.pred_length)

    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1,3)[:,0,:,:]

    # for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
    #     testx = torch.Tensor(x).to(device)
    #     testx = testx.transpose(1,3)
    #     with torch.no_grad():
    #         preds = engine.model(testx).transpose(1,3)
        
    #     if len(preds.shape) == 4:
    #         # 如果有4个维度，取第一个特征
    #         preds = preds[:, 0, :, :]
    #     outputs.append(preds)

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        try:
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1,3)

            with torch.no_grad():
                preds = engine.model(testx).transpose(1,3)
            
            if len(preds.shape) == 4:
                preds = preds[:, 0, :, :]
            outputs.append(preds)

            # 主动释放无用变量，清缓存
            del testx, preds
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                if args.batch_size > 1:
                    args.batch_size = args.batch_size // 2
                    print(f"⚠️ 显存不足，尝试减小 batch_size 至 {args.batch_size}")
                print(f"⚠️ 第 {iter} 批次推理时显存不足，跳过该批次")
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
        
        # Store individual horizon results
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

    # Return results for CSV storage
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
    # Prepare data for CSV
    csv_data = []
    
    for result in results:
        # Basic experiment info
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
        
        # Add individual horizon results
        for horizon_result in result['horizon_results']:
            row[f'horizon_{horizon_result["horizon"]}_mae'] = horizon_result['mae']
            row[f'horizon_{horizon_result["horizon"]}_mape'] = horizon_result['mape']
            row[f'horizon_{horizon_result["horizon"]}_rmse'] = horizon_result['rmse']
        
        csv_data.append(row)
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv('experiment_results.csv', index=False)
    
    print(f"\nResults saved to 'experiment_results.csv'")
    print(f"Columns saved: {list(df.columns)}")
    
    # Also save a summary CSV
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
        # Run experiments with different sequence lengths
        print("🔬 运行多个序列长度实验...")
        results = run_experiments_with_different_seq_lengths()
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        print(f"\nAll experiments completed!")
        print(f"Total time for all experiments: {total_time:.4f} seconds ({total_time/3600:.2f} hours)")
        
        # Print summary
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        for result in results:
            print(f"Seq Length {result['seq_length']:2d}, Pred Length {result['pred_length']:2d}: MAE={result['test_mae_avg']:.4f}, MAPE={result['test_mape_avg']:.4f}, RMSE={result['test_rmse_avg']:.4f}")
    else:
        # Run single experiment with specified parameters
        print("🏋️ 运行单个实验...")
        print(f"📊 配置: seq_length={args.seq_length}, pred_length={args.pred_length}")
        
        # Generate data for the specified seq_length and pred_length
        print(f"🔄 为 seq_length={args.seq_length}, pred_length={args.pred_length} 生成数据...")
        generate_data_for_seq_length(args.seq_length, args.pred_length)
        
        result = main_experiment()
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        print(f"\nExperiment completed!")
        print(f"Total time: {total_time:.4f} seconds ({total_time/3600:.2f} hours)")
        print(f"Results: MAE={result['test_mae_avg']:.4f}, MAPE={result['test_mape_avg']:.4f}, RMSE={result['test_rmse_avg']:.4f}")



### nvidia-smi
### srun -p 4090 --pty --gpus 1 -t 12:00:00 bash -i
### conda activate Energy-TSF
### cd /mnt/webscistorage/cc7738/ws_joella/EnergyTSF
### cd /mnt/webscistorage/cc7738/ws_joella/EnergyTSF_latest


# wandb login

# cd /mnt/webscistorage/cc7738/ws_joella/EnergyTSF/GNN/Graph-WaveNet-master-origin
# python train.py --data data/FRANCE --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 10

'''





'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 基础组件
# =========================

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x: [B, C, N, T], A: [N, N]
        return torch.einsum("ncvl,vw->ncwl", (x, A)).contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True
        )

    def forward(self, x):
        return self.mlp(x)


# =========================
# Chebyshev 谱卷积
# =========================

class ChebConv(nn.Module):
    """
    K阶Chebyshev谱域图卷积；保留alpha系数用于可解释性导出。
    """
    def __init__(self, c_in, c_out, K=3, dropout=0.0):
        super(ChebConv, self).__init__()
        assert K >= 1
        self.K = K
        self.dropout = dropout
        self.theta = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, kernel_size=(1, 1), bias=True) for _ in range(K)]
        )
        self.alpha = nn.Parameter(torch.ones(K), requires_grad=True)
        self.last_cheb_alphas = None  # 便于外部读取

    @staticmethod
    def build_laplacian(A, add_self=True, eps=1e-5):
        N = A.size(0)
        if add_self:
            A = A + torch.eye(N, device=A.device)
        deg = torch.clamp(A.sum(-1), min=eps)
        D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
        L = torch.eye(N, device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
        return L

    def forward(self, x, A, include_self=True):
        B, C, N, T = x.shape
        L = self.build_laplacian(A, add_self=include_self)
        L_tilde = 2.0 * L - torch.eye(N, device=A.device)

        Tx_list = []
        Tx0 = x
        Tx_list.append(Tx0)
        if self.K > 1:
            Tx1 = torch.einsum("vw,bcwl->bcvl", L_tilde, x)
            Tx_list.append(Tx1)
            for _k in range(2, self.K):
                Tx2 = 2 * torch.einsum("vw,bcwl->bcvl", L_tilde, Tx1) - Tx0
                Tx_list.append(Tx2)
                Tx0, Tx1 = Tx1, Tx2

        out = 0
        with torch.no_grad():
            self.last_cheb_alphas = self.alpha.detach().cpu()

        for k in range(self.K):
            out = out + self.alpha[k] * self.theta[k](Tx_list[k])

        out = F.dropout(out, self.dropout, training=self.training)
        return out


# =========================
# 空间域 GCN（支持 diffusion / 幂律）
# =========================

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2,
                 use_power=False, diag_mode="self_and_neighbor"):
        super(gcn, self).__init__()
        self.nconv = nconv()
        self.dropout = dropout
        self.order = order
        self.support_len = support_len
        self.use_power = use_power
        assert diag_mode in ["neighbor", "self_and_neighbor"]
        self.diag_mode = diag_mode

        c_total = (order * support_len + 1) * c_in
        self.mlp = linear(c_total, c_out)

        # 幂律系数（每阶一个），便于可解释
        self.power_coef = nn.Parameter(torch.ones(order), requires_grad=True)
        self.last_power_coef = None

    def _apply_diag_policy(self, A):
        if self.diag_mode == "neighbor":
            A = A.clone()
            A.fill_diagonal_(0.0)
            return A
        else:
            return A

    def _matrix_powers(self, A, K):
        powers = []
        Ak = A
        for _ in range(K):
            powers.append(Ak)
            Ak = Ak @ A
        return powers

    def forward(self, x, supports):
        with torch.no_grad():
            self.last_power_coef = self.power_coef.detach().cpu()

        out = [x]
        for A in supports:
            A_use = self._apply_diag_policy(A)
            if self.use_power: # power law
                # A^k 逐阶拼接（保持通道数 = order*support_len*c_in）
                A_pows = self._matrix_powers(A_use, self.order)
                for k_idx, Ak in enumerate(A_pows):
                    xk = self.nconv(x, Ak)
                    out.append(self.power_coef[k_idx] * xk)
            else:
                # diffusion：递推但保留每阶输出并concat
                x1 = self.nconv(x, A_use)
                out.append(x1)
                x_prev = x1
                for _k in range(2, self.order + 1):
                    x2 = self.nconv(x_prev, A_use)
                    out.append(x2)
                    x_prev = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# =========================
# NEW: 双图递推 MixProp（adj_1/adj_2 + 递推累加 + A-dropout）
# =========================

class MixPropDual(nn.Module):
    """
    两路图（adj_1 from base A, adj_2 from learnable r1），K 阶递推：
        inj_k = Conv1x1(x) * gate_k
        z_k   = adj_1 @ z_{k-1} + inj_k
        z2_k  = adj_2 @ z2_{k-1} + inj_k
        out   = z_K + z2_K
    形状：
        x:  [B, C_in, N, T]
        out:[B, C_out, N, T]
    """
    def __init__(self, c_in, c_out, K=3, droprate=0.1, temperature=1.0,
                 diag_mode='self_and_neighbor', r_dim=10, use_laplacian=False):
        super().__init__()
        assert K >= 1
        self.K = K
        self.droprate = float(droprate)
        self.temperature = float(temperature)
        assert diag_mode in ['neighbor', 'self_and_neighbor']
        self.diag_mode = diag_mode
        self.use_laplacian = use_laplacian

        # 每阶1x1 conv（线性核）
        self.k_convs = nn.ModuleList([nn.Conv2d(c_in, c_out, kernel_size=1) for _ in range(K)])
        # 每阶门控（标量）
        self.gates = nn.Parameter(torch.ones(K), requires_grad=True)

        # 自适应图的节点嵌入 r1（在前向扩展到 [N, r_dim]）
        self.r1 = nn.Parameter(torch.randn(1, r_dim), requires_grad=True)

    @staticmethod
    def _apply_diag(A, mode):
        if mode == 'neighbor':
            A = A.clone()
            A.fill_diagonal_(0.0)
            return A
        return A

    @staticmethod
    def _laplacian(A, eps=1e-6):
        N = A.size(0)
        deg = torch.clamp(A.sum(-1), min=eps)
        D_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
        L = torch.eye(N, device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
        return L

    def _build_adj1_from_A(self, A):
        # 从固定图 A 构造 adj_1：ReLU -> 温度 -> softmax(行) -> 去/留对角 -> dropout
        M = self._laplacian(A) if self.use_laplacian else A
        M = F.relu(M)
        adj_1 = torch.softmax(M / max(self.temperature, 1e-6), dim=1)  # row-stochastic
        adj_1 = self._apply_diag(adj_1, self.diag_mode)
        if self.training and self.droprate > 0:
            adj_1 = F.dropout(adj_1, p=self.droprate)
        return adj_1

    def _build_adj2_from_r1(self, N, device):
        # 由可学习节点嵌入 r1 构造 adj_2：r1 r1^T -> ReLU -> 温度 -> softmax(行) -> 去/留对角 -> dropout
        r1 = self.r1
        if r1.size(0) != N:
            r1 = r1.expand(N, -1).contiguous()
        S = r1 @ r1.t()
        S = F.relu(S)
        adj_2 = torch.softmax(S / max(self.temperature, 1e-6), dim=1)
        adj_2 = self._apply_diag(adj_2, self.diag_mode)
        if self.training and self.droprate > 0:
            adj_2 = F.dropout(adj_2, p=self.droprate)
        return adj_2

    def forward(self, x, A_base):
        """
        x: [B, C_in, N, T]
        A_base: [N, N]（来自 supports[0] 或自适应图）
        """
        B, C, N, T = x.shape
        device = x.device

        # 两路邻接
        adj_1 = self._build_adj1_from_A(A_base)
        adj_2 = self._build_adj2_from_r1(N, device)

        # 预先计算每阶注入 inj_k: [B, C_out, N, T]
        inj = [self.k_convs[k](x) * self.gates[k] for k in range(self.K)]

        # 路一：A1 递推
        z = inj[0].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)  # [B*T, N, C_out]
        for k in range(1, self.K):
            z = adj_1 @ z + inj[k].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        z = z.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous()  # [B, C_out, N, T]

        # 路二：A2 递推
        z_fix = inj[0].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        for k in range(1, self.K):
            z_fix = adj_2 @ z_fix + inj[k].permute(0, 3, 2, 1).contiguous().view(B * T, N, -1)
        z_fix = z_fix.view(B, T, N, -1).permute(0, 3, 2, 1).contiguous()

        return z + z_fix


# =========================
# Graph WaveNet 主体（含MixProp/幂律/Chebyshev开关）
# =========================

class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32, skip_channels=256,
                 end_channels=512, kernel_size=2, blocks=4, layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device

        # 从环境变量读取扩展功能
        self.use_power = os.getenv("GWN_USE_POWER", "0") == "1"
        self.use_cheby = os.getenv("GWN_USE_CHEBY", "0") == "1"
        self.cheby_K = int(os.getenv("GWN_CHEBY_K", "3"))
        self.diag_mode = os.getenv("GWN_DIAG_MODE", "self_and_neighbor")

        # NEW: MixProp 开关与参数
        self.use_mixprop = os.getenv("GWN_USE_MIXPROP", "0") == "1"
        self.mixprop_K = int(os.getenv("GWN_MIXPROP_K", "3"))
        self.adj_droprate = float(os.getenv("GWN_ADJ_DROPOUT", "0.1"))
        self.adj_temperature = float(os.getenv("GWN_ADJ_TEMP", "1.0"))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.cheb_convs = nn.ModuleList()
        self.mixprop_convs = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        # 计算感受野
        receptive_field = 1
        new_dilation = 1
        for _b in range(blocks):
            for _i in range(layers):
                receptive_field += (kernel_size - 1) * new_dilation
                new_dilation *= 2
        self.receptive_field = receptive_field

        # supports
        self.supports = supports
        self.supports_len = len(supports) if supports is not None else 0

        # 自适应邻接参数
        if gcn_bool and addaptadj:
            if aptinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True)
                self.supports_len += 1
            else:
                U, S, V = torch.svd(aptinit)
                initemb1 = U[:, :10] @ torch.diag(S[:10].pow(0.5))
                initemb2 = torch.diag(S[:10].pow(0.5)) @ V[:, :10].t()
                self.nodevec1 = nn.Parameter(initemb1.to(device), requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2.to(device), requires_grad=True)
                self.supports_len += 1

        # 构建层
        new_dilation = 1
        for _b in range(blocks):
            for _i in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation)
                )
                self.gate_convs.append(
                    nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=new_dilation)
                )
                self.residual_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))

                if gcn_bool:
                    if self.use_cheby:
                        self.cheb_convs.append(
                            ChebConv(dilation_channels, residual_channels, K=self.cheby_K, dropout=dropout)
                        )
                        self.gconv.append(None)
                        self.mixprop_convs.append(None)
                    elif self.use_mixprop:
                        self.cheb_convs.append(None)
                        self.gconv.append(None)
                        self.mixprop_convs.append(
                            MixPropDual(
                                c_in=dilation_channels,
                                c_out=residual_channels,
                                K=self.mixprop_K,
                                droprate=self.adj_droprate,
                                temperature=self.adj_temperature,
                                diag_mode=self.diag_mode,
                                r_dim=10,
                                use_laplacian=False
                            )
                        )
                    else:
                        self.gconv.append(
                            gcn(dilation_channels, residual_channels, dropout,
                                support_len=self.supports_len, order=2,
                                use_power=self.use_power, diag_mode=self.diag_mode)
                        )
                        self.cheb_convs.append(None)
                        self.mixprop_convs.append(None)
                else:
                    self.gconv.append(None)
                    self.cheb_convs.append(None)
                    self.mixprop_convs.append(None)

                new_dilation *= 2

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1), bias=True)

    def _build_adaptive_adj(self):
        # 与原实现保持一致：ReLU+softmax 得到自适应图
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        return adp

    def _collect_supports(self):
        sup = []
        if self.supports is not None:
            sup += self.supports
        if self.gcn_bool and self.addaptadj:
            sup.append(self._build_adaptive_adj())
        return sup if len(sup) > 0 else None

    def forward(self, input):
        # padding到感受野
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0

        # 收集 supports（包含自适应）
        new_supports = self._collect_supports() if self.gcn_bool else None

        layer_idx = 0
        for _b in range(self.blocks):
            for _i in range(self.layers):
                residual = x
                filt = torch.tanh(self.filter_convs[layer_idx](residual))
                gate = torch.sigmoid(self.gate_convs[layer_idx](residual))
                x = filt * gate

                # skip
                s = self.skip_convs[layer_idx](x)
                try:
                    skip = skip[:, :, :, -s.size(3):]
                except Exception:
                    skip = 0
                skip = s + skip

                # 空间聚合：优先 Cheby，其次 MixProp，否则原 gcn；若无图则用1x1残差
                if self.gcn_bool and (new_supports is not None):
                    # Chebyshev
                    if self.cheb_convs[layer_idx] is not None:
                        acc = 0
                        for A in new_supports:
                            xs = self.cheb_convs[layer_idx](x, A, include_self=(self.diag_mode == "self_and_neighbor"))
                            acc = acc + xs
                        x = acc / float(len(new_supports))
                    # MixProp
                    elif self.mixprop_convs[layer_idx] is not None:
                        # 选一张“基图”给 adj_1（优先第一张 support；若仅有自适应图也可）
                        if len(new_supports) > 0:
                            A_base = new_supports[0]
                        else:
                            A_base = self._build_adaptive_adj()
                        x = self.mixprop_convs[layer_idx](x, A_base)
                    # 原 gcn
                    elif self.gconv[layer_idx] is not None:
                        x = self.gconv[layer_idx](x, new_supports)
                    else:
                        x = self.residual_convs[layer_idx](x)
                else:
                    x = self.residual_convs[layer_idx](x)

                # 残差 + BN
                x = x + residual[:, :, :, -x.size(3):]
                x = self.bn[layer_idx](x)
                layer_idx += 1

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


##############################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # import pdb; pdb.set_trace()
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a) # x*a 
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # x = torch.sigmoid(x)
        return x


'''


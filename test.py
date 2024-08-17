import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataset, main, metrics, make_plots
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

def eval_model(dataset, model):
    targets_all, preds_all, losses_all = [], [], []
    for bid, batch in enumerate(dataset):

        pred, targ, loss = metrics.fun_test_forward(model, batch)

        targets_all.append(targ.cpu().detach().numpy())
        preds_all.append(pred.cpu().detach().numpy())
        losses_all.append(np.mean(loss.cpu().detach().numpy()))
        
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    attr_dict = {
        "targets": targets_all,
        "predictions": preds_all,
        "losses": losses_all,
    }
    return attr_dict

class tester(object):
    def __init__(self, model, test_path, run_dir, optimizer, criterion):
        super(tester, self).__init__()                
        self.model = model
        self.test_path = test_path
        self.run_dir = run_dir
        self.optimizer = optimizer
        self.criterion = criterion
        test_folders = main.get_folder_list(self.test_path)
        self.imu_freq = 100
        all_ate, all_trte, all_drte = [],[],[]

        for folder in test_folders:
        
            # Output Location
            folder_num = folder.split('/')[-1]                 # use '/' when running in linux and '\\' when running in windows
            outdir = os.path.join(self.run_dir, folder_num)
            if os.path.exists(outdir) is False:
                os.mkdir(outdir)
            
            # Load the test data
            test_basic = dataset.DataInitialization([folder], outdir, mode='test')
            test_dataset = dataset.SequenceDataset(test_basic, test_basic.get_merged_index_map())
            testset = DataLoader(test_dataset, batch_size=512, shuffle=False)

            # Run the forward pass for test data
            test_info = eval_model(testset, self.model)

            # compute the position
            trajectory_info = metrics.compute_position(test_dataset, test_info['predictions'])
            trajectory_file = os.path.join(outdir, 'trajectory.txt')
            trajectory_data = np.concatenate(
                [
                    trajectory_info['ts'].reshape(-1,1),       # (ts)
                    trajectory_info['position_prediction'],      # (x,y,z)
                    trajectory_info['position_gt']              # (x,y,z)
                ], axis=1)
            traj_column_names = ["Timestamp", "Position Prediction X", "Position Prediction Y",
                                 "Position Prediction Z", "Position GT X", "Position GT Y", "Position GT Z"]
            np.savetxt(trajectory_file, trajectory_data, delimiter=",", header=",".join(traj_column_names), comments="")

            plot_traj_dict = metrics.compute_plot_dict(self.imu_freq, test_info, trajectory_info)
            net_output_file = os.path.join(outdir, 'net_output.txt')
            net_output_data = np.concatenate(
                [
                    plot_traj_dict['pred_ts'].reshape(-1, 1),       # (ts)
                    plot_traj_dict['preds'],                        # (x,y,z)
                    plot_traj_dict['targets']                       # (x,y,z)
                ], axis=1)
            net_column_names = ["Timestamp", "Position Prediction X", "Position Prediction Y",
                                "Position Prediction Z", "Position Target X", "Position Target Y", "Position Target Z"]
            np.savetxt(net_output_file, net_output_data, delimiter=",", header=",".join(net_column_names), comments="")

            make_plots.plot_traj(plot_traj_dict, outdir)

            # ATE, T-RTE and D-RTE metrics
            # time delta is considered for an entire minute, hence multiplying the IMU freq (100) with 60 (60 seconds)
            # distance delta is 
            ate, t_rte, d_rte = metrics.compute_metrics(trajectory_info['position_prediction'], trajectory_info['position_gt'], time_delta=self.imu_freq * 60, dist_delta=1)

            # Save the metrics for the trajectory
            metrics_file = os.path.join(outdir, 'metrics.txt')
            with open(metrics_file, 'w') as file:
                file.write(f"ATE - {ate}m (UNIT - METERS)\nT-RTE - {t_rte}m (UNIT - METERS)\nD-RTE - {d_rte}m (UNIT - METERS)")

            all_ate.append(ate)
            all_drte.append(d_rte)
            all_trte.append(t_rte)
            
        # Save the average metrics for ATE, T-RTE and D-RTE
        test_metrics = os.path.join(run_dir, 'Test Metrics.txt')
        with open(test_metrics, 'w') as file:
                file.write(f"Average ATE - {float(np.mean(all_ate))}m (UNIT - METERS\nAverage T-RTE - {np.mean(all_trte)}m (UNIT - METERS)\nAverage D-RTE - {float(np.mean(all_drte))}m (UNIT - METERS)")
                
        print("COMPLETED")

from scipy.interpolate import interp1d
import pandas as pd
import random
from numpy.random import normal as gen_normal
import os
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class LoadData(object):
    def __init__(self, data_path, imu_freq, window_size):
        super().__init__()
        (
            self.ts,
            self.features,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (None, None, None, None, None, None)
        self.imu_freq = imu_freq    #100Hz
        self.interval = window_size ##100ms = 1s
        self.data_valid = False
        self.sum_dur = 0
        self.valid = False
        self.get_gt = True
        if data_path is not None:
            self.valid = self.load(data_path)      #self.valid = True

    def load(self, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        # import the SenseINS file
        file = os.path.join(data_path, 'SenseINS.h5')
        if os.path.exists(file):
            imu_all = pd.read_hdf(file, 'imu_all')
        else:
            file = os.path.join(data_path, 'SenseINS.csv')
            print(f"Opening {file}")
            if os.path.exists(file):
                imu_all = pd.read_csv(file)
            else:
                print(f"dataset_fb.py: file is not exist. {file}")
                return

        # timestamp
        if 'times' in imu_all:
            tmp_ts = np.array(imu_all[['times']].values)
        else:
            tmp_ts = np.array(imu_all[['time']].values)

        if tmp_ts.shape[0] < 1000:
            return False
        tmp_ts = np.squeeze(tmp_ts)

        # Orientation data
        tmp_vio_q = np.array(imu_all[['vio_q_w', 'vio_q_x', 'vio_q_y', 'vio_q_z']].values)
        # Ground truth
        tmp_vio_p = np.array(imu_all[['vio_p_x', 'vio_p_y', 'vio_p_z']].values)

        # gyro and acceleration
        tmp_gyro = np.array(imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values)
        tmp_accel = np.array(imu_all[['acce_x', 'acce_y', 'acce_z']].values)

        # gyro and acceleration bias
        tmp_vio_gyro_bias = np.array(imu_all[['vio_gyro_bias_x', 'vio_gyro_bias_y', 'vio_gyro_bias_z']].values)
        tmp_vio_acce_bias = np.array(imu_all[['vio_acce_bias_x', 'vio_acce_bias_y', 'vio_acce_bias_z']].values)

        # gyro and acceleration bias correction
        tmp_gyro = tmp_gyro - tmp_vio_gyro_bias[-1, :]
        tmp_acce = tmp_accel - tmp_vio_acce_bias[-1, :]

        # Timestamp sequencing
        start_ts = tmp_ts[10]
        end_ts = tmp_ts[10] + int((tmp_ts[-20]-tmp_ts[1]) * self.imu_freq) / self.imu_freq
        ts = np.arange(start_ts, end_ts, 1.0/self.imu_freq)
        self.data_valid = True
        self.sum_dur = end_ts - start_ts

        # Pre-processing the data for orientation, groundtruth and IMU measurements. 
        # Interpolation and slerping
        vio_q_slerp = Slerp(tmp_ts, Rotation.from_quat(tmp_vio_q[:, [1, 2, 3, 0]]))
        vio_r = vio_q_slerp(ts)
        vio_p = interp1d(tmp_ts, tmp_vio_p, axis=0)(ts)
        gyro = interp1d(tmp_ts, tmp_gyro, axis=0)(ts)
        acce = interp1d(tmp_ts, tmp_acce, axis=0)(ts)

        ts = ts[:, np.newaxis]
        ori_R_vio = vio_r
        ori_R = ori_R_vio
        gt_disp = vio_p[self.interval:] - vio_p[: -self.interval]

        glob_gyro = np.einsum("tip,tp->ti", ori_R.as_matrix(), gyro)
        glob_acce = np.einsum("tip,tp->ti", ori_R.as_matrix(), acce)
        glob_acce -= np.array([0.0, 0.0, 9.805])

        self.ts = ts                                                        # timestamp
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)      #gyro and acceleration
        self.orientations = ori_R.as_quat()                                 # [x, y, z, w]    #orientation
        self.gt_pos = vio_p                                                 #groundtruth position (x,y,z)
        self.gt_ori = ori_R_vio.as_quat()                                   # [x, y, z, w]
        self.targets = gt_disp
        print(f"File Location = {data_path}")
        return True

    def plot_example_trajectories(self, op_dir):
        fig = plt.figure(figsize=(10, 8))
        traj_gt_pos = np.array(self.gt_pos)        
        if traj_gt_pos.ndim == 1:
            traj_gt_pos = traj_gt_pos.reshape(-1, 3)        
        plt.plot(traj_gt_pos[:, 0], traj_gt_pos[:, 1], label='Ground Truth Position')        
        plt.title(f'Ground Truth')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(op_dir, "ground_truth.png"))
        
    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_data_valid(self):
        return self.data_valid

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos, self.gt_ori], axis=1)
    
class DataInitialization(object):
    def __init__(self, data_list, outdir=None, **kwargs):
        super(DataInitialization, self).__init__()
        self.window_size = 100
        self.step_size = 5
        self.seq_len = 10

        self.index_map = []
        self.ts, self.orientations, self.gt_pos, self.gt_ori = [], [], [], []
        self.features, self.targets = [], []
        self.valid_t, self.valid_samples = [], []
        self.data_paths = []
        self.valid_continue_good_time = 0.1
        

        self.mode = kwargs.get("mode", "train")
        sum_t = 0
        self.valid_sum_t = 0
        self.valid_all_samples = 0
        valid_i = 0
        
        # Run the LoadData class for every folder in the data.
        for i in range(len(data_list)):
            seq = LoadData(data_list[i], imu_freq=100, window_size=self.window_size)
            if self.mode == 'test':
                seq.plot_example_trajectories(op_dir = outdir)            
            if seq.valid is False:
                continue
            feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
            sum_t += seq.sum_dur
            valid_samples = 0
            index_map = []
            step_size = self.step_size
            
            for j in range(0, targ.shape[0] - (self.seq_len - 1) * self.window_size, step_size):
                    index_map.append([valid_i, j])
                    self.valid_all_samples += 1
                    valid_samples += 1

            if len(index_map) > 0:
                self.data_paths.append(data_list[i])
                self.index_map.append(index_map)
                self.features.append(feat)
                self.targets.append(targ)
                self.ts.append(aux[:, 0])
                self.orientations.append(aux[:, 1:5])
                self.gt_pos.append(aux[:, 5:8])
                self.gt_ori.append(aux[:, 8:12])
                self.valid_samples.append(valid_samples)
                valid_i += 1
            
        print(f"datasets sum time {sum_t}")

    def get_data(self):
        return self.features, self.targets, self.ts, self.orientations, self.gt_pos, self.gt_ori

    def get_index_map(self):
        return self.index_map

    def get_merged_index_map(self):
        index_map = []
        for i in range(len(self.index_map)):
            index_map += self.index_map[i]
        return index_map

class LSTMSeqToSeqDataset(Dataset):
    def __init__(self, basic_data: DataInitialization, index_map, **kwargs):
        super(LSTMSeqToSeqDataset, self).__init__()
        self.window_size = basic_data.window_size
        self.step_size = basic_data.step_size
        self.seq_len = basic_data.seq_len

        self.mode = kwargs.get("mode", "train")
        self.shuffle = False
        if self.mode == ["train", "val"]:
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False

        self.features, self.targets, self.ts, self.orientations, self.gt_pos, self.gt_ori = basic_data.get_data()  
        self.index_map = index_map
        if self.shuffle:
            random.shuffle(self.index_map)


    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        # in the world frame
        feat = self.features[seq_id][frame_id :
                                     frame_id + self.seq_len * self.window_size]
        # raw_feat = feat
        targ = self.targets[seq_id][frame_id:
                                    frame_id + self.seq_len * self.window_size:
                                    self.window_size]  # the beginning of the sequence

        feat = feat.reshape(feat.shape[0], -1)  # Reshaping to 2D

        seq_feat = []
        for i in range(self.seq_len):            
            seq_feat.append(feat[i * self.window_size:(i + 1) * self.window_size].flatten())    # Flatten the window to one long vector

        seq_feat = np.stack(seq_feat)  # Stack to get 3D tensor for LSTM
        return seq_feat.astype(np.float32), targ.astype(np.float32)

    def __len__(self):
        return len(self.index_map)

def SequenceDataset(basic_data: DataInitialization, index_map, **kwargs):
    return LSTMSeqToSeqDataset(basic_data, index_map, **kwargs)

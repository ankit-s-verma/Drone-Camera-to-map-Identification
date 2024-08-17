import numpy as np
from scipy.interpolate import interp1d
import torch

def get_sequence_smooth_loss(pred, targ):
    
    cum_gt_pos = torch.cumsum(targ, 1)
    pred_cum_pos = torch.cumsum(pred, 1)
    absolute_weight = 8.0
    loss1 = (pred - targ).pow(2)
    if len(pred.size()) == 2:
        return torch.mean(loss1)
    loss2 = absolute_weight * (pred_cum_pos[:, 1:] - cum_gt_pos[:, 1:]).pow(2)
    loss = torch.cat((loss1, loss2), 1)

    return torch.mean(loss)

def fun_test_forward(model, batch):
    feat, targ = batch
    pred = model(feat)
    output_dim = 3
    pred = pred[:, -1, ]
    targ = targ[:, -1, 0:output_dim]

    loss = get_sequence_smooth_loss(pred, targ)

    return pred, targ, loss

def compute_position(dataset, preds):
    window_time = 1     # 1 second
    imu_freq = 100      # 100Hz    
    seq_len = 10         

    dp_t = window_time
    pred_vels = preds / dp_t   
    ind = np.array([i[1] for i in dataset.index_map], dtype=np.int_)
    delta_int = int(window_time * imu_freq / 2.0)   #50
    delta_int += int((seq_len - 1) * window_time * imu_freq)
    if not (window_time * imu_freq / 2.0).is_integer():
        print("Trajectory integration point is not centered.")
    ind_intg = ind + delta_int
    ts = dataset.ts[0]
    dts = np.mean(ts[ind_intg[1:]] - ts[ind_intg[:-1]])
    pos_intg = np.zeros([pred_vels.shape[0] + 1, pred_vels.shape[1]])    
    pos_intg[0] = dataset.gt_pos[0][ind_intg[0], 0:pos_intg.shape[1]]
    pos_intg[1:] = np.cumsum(pred_vels[:, :] * dts, axis=0) + pos_intg[0]
    ts_intg = np.append(ts[ind_intg], ts[ind_intg[-1]] + dts)
    ts_in_range = ts[ind_intg[0] : ind_intg[-1]]
    pos_pred = interp1d(ts_intg, pos_intg, axis=0)(ts_in_range)
    pos_gt = dataset.gt_pos[0][ind_intg[0] : ind_intg[-1], 0:pos_intg.shape[1]]

    traj_attr_dict = {
        "ts": ts_in_range,
        "position_prediction": pos_pred,
        "position_gt": pos_gt,
    }
    
    return traj_attr_dict


def compute_plot_dict(imu_freq, test_info, traj_info):
    ts = traj_info['ts']
    pos_pred = traj_info['position_prediction']
    pos_gt = traj_info['position_gt']

    total_pred = test_info['predictions'].shape[0]
    pred_ts = (1.0 / imu_freq) * np.arange(total_pred)

    plot_traj_dict = {
        "ts": ts,
        "pos_pred": pos_pred,
        "pos_gt": pos_gt,
        "pred_ts": pred_ts,
        "preds": test_info["predictions"],
        "targets": test_info["targets"]
    }

    return plot_traj_dict

def compute_trte(pred, gt, time_delta, max_delta=-1):
    if max_delta == -1:
        max_delta = pred.shape[0]
    time_deltas = np.array([time_delta]) if time_delta > 0 else np.arange(1, min(pred.shape[0], max_delta))
    t_rtes = np.zeros((time_deltas.shape[0], 2))
    for i in range(time_deltas.shape[0]):
        # For each delta, the RTE is computed as the RMSE of endpoint drifts from fixed windows
        pred_disp = pred[time_deltas[i]:] - pred[:-time_deltas[i]]
        gt_disp = gt[time_deltas[i]:] - gt[:-time_deltas[i]]
        err = pred_disp - gt_disp
        t_rtes[i] = np.sqrt(np.mean(np.linalg.norm(err, axis=1) ** 2))       # RMSE value, if we want mse, we don't need to

    return np.mean(t_rtes)

def compute_metrics(pred, gt, time_delta, dist_delta, max_delta=-1):
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ATE is used to measure the RMSE b/w GT and Estimated Trajectory. It provides overall accuracy for estimated trajectory. The unit is meter (m)
    ate = np.sqrt(np.mean(np.linalg.norm(pred - gt, axis=1) ** 2))

    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # T-RTE measures the drift error of the trajectory over time by comparing the postion at same time interval b/w GT and Estimnated Trajectory. 
    # The unit is meter (m) for postion and seconds (s) for time interval.

    if pred.shape[0] < time_delta:
        ratio = time_delta / pred.shape[0]
        t_rte = compute_trte(pred, gt, pred.shape[0] - 1) * ratio
    else:
        t_rte = compute_trte(pred, gt, time_delta)

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # D-RTE measures the drift error of the trajectory over distance by comparing the postion at same distance interval b/w GT and Estimnated Trajectory.
    # The unit is meter (m) for postion

    distances = np.cumsum(np.sqrt(np.sum(np.diff(gt, axis=0)**2, axis=1)))
    temp_drte = []
    current_interval_end = dist_delta
    start_index = 0

    for i in range(1, len(distances)):
        if distances[i] >= current_interval_end:
            d_rte = np.sqrt(np.mean(np.sum((pred[start_index:i+1] - gt[start_index:i+1])**2, axis=1)))
            temp_drte.append(d_rte)
            current_interval_end += dist_delta
            start_index = i

    d_rte = np.mean(temp_drte) if temp_drte else 0
    
    return ate, t_rte, d_rte
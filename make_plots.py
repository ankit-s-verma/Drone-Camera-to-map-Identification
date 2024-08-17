import numpy as np
import matplotlib.pyplot as plt
import os

def plot_traj(plot_dict, outdir):
    pos_pred = plot_dict["pos_pred"]
    pos_gt = plot_dict["pos_gt"]
    pred_ts = plot_dict["pred_ts"]
    preds = plot_dict["preds"]
    targets = plot_dict["targets"]

    dpi = 90
    figsize = (16, 9)

    fig1 = plt.figure(num="ins_traj", dpi=dpi, figsize=figsize)
    targ_names = ["dx", "dy", "dz"]
    plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    plt.plot(pos_pred[:, 0], pos_pred[:, 1])
    plt.plot(pos_gt[:, 0], pos_gt[:, 1])
    plt.axis("equal")
    plt.legend(["network_pred", "Ground_truth"])
    plt.title("2D trajectory and ATE error against time")
    for i in range(preds.shape[1]):
        plt.subplot2grid((preds.shape[1], 2), (i, 1))
        plt.plot(preds[:, i])
        plt.plot(targets[:, i])
        plt.legend(["network_pred", "Ground_truth"])
        plt.title("{}".format(targ_names[i]))
    plt.tight_layout()
    plt.grid(True)
    fig1.savefig(os.path.join(outdir, "Trajectory.png"))

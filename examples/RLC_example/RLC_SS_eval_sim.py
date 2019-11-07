import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join("..", ".."))
from torchid.ssfitter import NeuralStateSpaceSimulator
from torchid.ssmodels import NeuralStateSpaceModel
from common import metrics

if __name__ == '__main__':

    #dataset_type = 'id'
    dataset_type = 'val'

    #model_type = '1step_nonoise'
    model_type = '1step_noise'
    #model_type = '64step_noise'
    #model_type = 'simerr_noise'
    plot_input = False

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']

    dataset_filename = f"RLC_data_{dataset_type}.csv"
    df_X = pd.read_csv(os.path.join("data", dataset_filename))

    time_data = np.array(df_X[COL_T], dtype=np.float32)
    # y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)
    y_var_idx = 0  # 0: voltage 1: current

    y = np.copy(x[:, [y_var_idx]])

    N = np.shape(y)[0]
    Ts = time_data[1] - time_data[0]

    n_a = 2  # autoregressive coefficients for y
    n_b = 2  # autoregressive coefficients for u
    n_max = np.max((n_a, n_b))  # delay

    std_noise_V = 0.0 * 5.0
    std_noise_I = 0.0 * 0.5
    std_noise = np.array([std_noise_V, std_noise_I])

    x_noise = np.copy(x) + np.random.randn(*x.shape) * std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:, [y_var_idx]]

    # Initialize optimization
    ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64) #NeuralStateSpaceModelLin(A_nominal*Ts, B_nominal*Ts)
    nn_solution = NeuralStateSpaceSimulator(ss_model)

    model_filename = f"model_SS_{model_type}.pkl"
    nn_solution.ss_model.load_state_dict(torch.load(os.path.join("models", model_filename)))


    # In[Validate model]
    t_val_start = 0
    t_val_end = time_data[-1]
    idx_val_start = int(t_val_start//Ts)#x.shape[0]
    idx_val_end = int(t_val_end//Ts)#x.shape[0]

    # Build fit data
    u_val = u[idx_val_start:idx_val_end]
    x_meas_val = x_noise[idx_val_start:idx_val_end]
    x_true_val = x[idx_val_start:idx_val_end]
    y_val = y[idx_val_start:idx_val_end]
    time_val = time_data[idx_val_start:idx_val_end]

    x_0 = x_meas_val[0, :]

    with torch.no_grad():
        x_sim_torch = nn_solution.f_sim(torch.tensor(x_0), torch.tensor(u_val))
        loss = torch.mean(torch.abs(x_sim_torch - torch.tensor(x_true_val)))


    # In[Plot]
    x_sim = np.array(x_sim_torch)
    if not plot_input:
        fig, ax = plt.subplots(2,1,sharex=True, figsize=(6, 5.5))
    else:
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 7.5))
    time_val_us = time_val *1e6

    if dataset_type == 'id':
        t_plot_start = 0.2e-3
    else:
        t_plot_start = 1.9e-3
    t_plot_end = t_plot_start + 0.32e-3

    idx_plot_start = int(t_plot_start//Ts)#x.shape[0]
    idx_plot_end = int(t_plot_end//Ts)#x.shape[0]


    ax[0].plot(time_val_us[idx_plot_start:idx_plot_end], x_true_val[idx_plot_start:idx_plot_end,0], 'k',  label='$v_C$')
    ax[0].plot(time_val_us[idx_plot_start:idx_plot_end], x_sim[idx_plot_start:idx_plot_end,0],'r--', label='$\hat{v}^{\mathrm{sim}}_C$')
    ax[0].legend(loc='upper right')
    ax[0].grid(True)
    ax[0].set_xlabel("Time ($\mu$s)")
    ax[0].set_ylabel("Voltage (V)")
    ax[0].set_ylim([-300, 300])

    ax[1].plot(time_val_us[idx_plot_start:idx_plot_end], np.array(x_true_val[idx_plot_start:idx_plot_end:,1]), 'k', label='$i_L$')
    ax[1].plot(time_val_us[idx_plot_start:idx_plot_end], x_sim[idx_plot_start:idx_plot_end:,1],'r--', label='$\hat i_L^{\mathrm{sim}}$')
    ax[1].legend(loc='upper right')
    ax[1].grid(True)
    ax[1].set_xlabel("Time ($\mu$s)")
    ax[1].set_ylabel("Current (A)")
    ax[1].set_ylim([-25, 25])

    if plot_input:
        ax[2].plot(time_val_us[idx_plot_start:idx_plot_end], u_val[idx_plot_start:idx_plot_end], 'k')
        #ax[2].legend(loc='upper right')
        ax[2].grid(True)
        ax[2].set_xlabel("Time ($\mu$s)")
        ax[2].set_ylabel("Input voltage $v_C$ (V)")
        ax[2].set_ylim([-400, 400])


    fig_name = f"RLC_SS_{dataset_type}_{model_type}.pdf"
    fig.savefig(os.path.join("fig", fig_name), bbox_inches='tight')

    R_sq = metrics.r_square(x_true_val, x_sim)
    print(f"R-squared metrics: {R_sq}")

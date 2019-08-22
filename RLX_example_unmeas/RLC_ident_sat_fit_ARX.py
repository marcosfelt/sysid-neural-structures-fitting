import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os
import sys
import scipy.linalg

sys.path.append(os.path.join(".."))
from torchid.arxfitter import NeuralARXSimulator
from torchid.util import RunningAverageMeter
from torchid.arxmodels import NeuralArxModel


if __name__ == '__main__':

    COL_T = ['time']
    COL_X = ['V_C', 'I_L']
    COL_U = ['V_IN']
    COL_Y = ['V_C']
    df_X = pd.read_csv(os.path.join("data", "RLC_data_sat_FE.csv"))

    t = np.array(df_X[COL_T], dtype=np.float32)
    y = np.array(df_X[COL_Y], dtype=np.float32)
    x = np.array(df_X[COL_X], dtype=np.float32)
    u = np.array(df_X[COL_U], dtype=np.float32)

    N = np.shape(y)[0]
    Ts = t[1] - t[0]
    t_fit = 2e-3
    n_fit = int(t_fit//Ts)#x.shape[0]
    num_iter = 20000
    test_freq = 100

    n_a = 2 # autoregressive coefficients for y
    n_b = 2 # autoregressive coefficients for u
    n_max = np.max((n_a, n_b)) # delay

    std_noise_V =  5.0
    std_noise_I = 0.0 * 0.5
    std_noise = np.array([std_noise_V, std_noise_I])

    x_noise = np.copy(x) + np.random.randn(*x.shape)*std_noise
    x_noise = x_noise.astype(np.float32)
    y_noise = x_noise[:,[0]]

    # Build fit data
    u_fit = u[0:n_fit]
    y_fit = y[0:n_fit]
    y_meas_fit = y_noise[0:n_fit]
    phi_fit_y = scipy.linalg.toeplitz(y_meas_fit, y_meas_fit[0:n_a])[n_max - 1:-1, :] # regressor 1
    phi_fit_u = scipy.linalg.toeplitz(u_fit, u_fit[0:n_a])[n_max - 1:-1, :]
    phi_fit = np.hstack((phi_fit_y, phi_fit_u))

    # To Pytorch tensors
    phi_fit_torch = torch.from_numpy(phi_fit)
    y_meas_fit_torch = torch.from_numpy(y_meas_fit)

    # Initialize optimization
    arx_model = NeuralArxModel(n_a=n_a, n_b=n_b, n_feat=64)
    arx_solution = NeuralARXSimulator(arx_model)
    optimizer = optim.Adam(arx_solution.arx_model.parameters(), lr=1e-4)
    end = time.time()
    loss_meter = RunningAverageMeter(0.97)

    ii = 0
    for itr in range(1, num_iter + 1):
        optimizer.zero_grad()

        # Predict
        y_pred_torch = arx_solution.f_onestep(phi_fit_torch)

        # Compute loss
        err = y_pred_torch - y_meas_fit_torch[n_max:, :]
        loss = torch.mean((err)**2)

        # Optimization step
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())

        # Print message
        if itr % test_freq == 0:
            with torch.no_grad():
                y_pred_torch = arx_solution.f_onestep(phi_fit_torch) #func(x_true_torch, u_torch)
                err = y_pred_torch - y_meas_fit_torch[n_max:, :]
                loss = torch.mean((err) ** 2)  # torch.mean(torch.sq(batch_x[:,1:,:] - batch_x_pred[:,1:,:]))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                ii += 1
        end = time.time()

    if not os.path.exists("models"):
        os.makedirs("models")
    
    #torch.save(arx_solution.arx_model.state_dict(), os.path.join("models", "model_IO_nonoise.pkl"))


    # Build validation data
    n_val = N
    u_val = u[0:n_val]
    y_val = y[0:n_val]
    y_meas_val = y_noise[0:n_val]
    phi_val_y = scipy.linalg.toeplitz(y_meas_val, y_meas_val[0:n_a])[n_max - 1:-1, :] # regressor 1
    phi_val_u = scipy.linalg.toeplitz(u_val, u_val[0:n_a])[n_max - 1:-1, :]
    phi_val = np.hstack((phi_val_y, phi_val_u))


    with torch.no_grad():
        y_seq = np.array(np.flip(y_val[0:n_a].ravel()))
        y_seq_torch = torch.tensor(y_seq)

        u_seq = np.array(np.flip(u_val[0:n_b].ravel()))
        u_seq_torch = torch.tensor(u_seq)

        u_torch = torch.tensor(u_val[n_max:,:])
        y_val_sim_torch = arx_solution.f_simerr(u_torch, y_seq_torch, u_seq_torch)

    # In[Plot]
    y_val_sim = np.array(y_val_sim_torch)
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(y_val[n_max:,0], 'b', label='True')
    ax[0].plot(y_val_sim[:,0], 'r',  label='Sim')


    ax[0].legend()
    ax[0].grid(True)

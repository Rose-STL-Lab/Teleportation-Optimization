""" Scripts for training and evaluating various meta-learning algorithms for MLP. """

import numpy as np
import pickle
import torch
from torch import nn

from gradient_descent_mlp import train_GD, train_meta_opt, test_meta_opt
from plot import plot_all

sigma = nn.LeakyReLU(0.1)
sigma_inv = nn.LeakyReLU(10)

dim = [20, 20, 20, 20] 

# do some random things first so that the wall-clock time comparison is fair
train_GD(dim, n_run=1, n_epoch=20, lr=3e-4)

# training with GD
epoch = 300
lr = 1e-4
time_arr_SGD_n, loss_arr_SGD_n, dL_dt_arr_SGD_n = \
    train_GD(dim, n_run=5, n_epoch=epoch, lr=3e-4)

# train an lstm that learns teleportation + lr
meta_opt_list, _ = train_meta_opt(dim, n_run=30, n_epoch=epoch, unroll=10, lr=lr, lr_meta=1e-3, learn_lr=True, learn_update=False, T_magnitude=0.01)
time_arr_teleport_lstm_lr_n, loss_arr_teleport_lstm_lr_n, dL_dt_arr_teleport_lstm_lr_n, lr_arr_teleport_lstm_lr_n = \
    test_meta_opt(meta_opt_list, None, dim, n_run=5, n_epoch=epoch, learn_lr=True, learn_update=False, T_magnitude=0.01)

# train an lstm that learns local update + teleportation
meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=700, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_tele=True, T_magnitude=0.01)
time_arr_lstm_update_tele_n, loss_arr_lstm_update_tele_n, dL_dt_arr_lstm_update_tele_n, lr_arr_lstm_update_tele_n = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_tele=True, T_magnitude=0.01)

# train an lstm that learns local update only
meta_opt_list, meta_opt_update = train_meta_opt(dim, n_run=600, n_epoch=100, unroll=10, lr=lr, lr_meta=1e-3, learn_tele=False)
time_arr_lstm_update_n, loss_arr_lstm_update_n, dL_dt_arr_lstm_update_n, lr_arr_lstm_update_n = \
    test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=100, learn_tele=False)


# save test results
results = (time_arr_SGD_n, loss_arr_SGD_n, dL_dt_arr_SGD_n, \
    time_arr_teleport_lstm_lr_n, loss_arr_teleport_lstm_lr_n, dL_dt_arr_teleport_lstm_lr_n, lr_arr_teleport_lstm_lr_n, \
    time_arr_lstm_update_tele_n, loss_arr_lstm_update_tele_n, dL_dt_arr_lstm_update_tele_n, lr_arr_lstm_update_tele_n, \
    time_arr_lstm_update_n, loss_arr_lstm_update_n, dL_dt_arr_lstm_update_n, lr_arr_lstm_update_n)
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)


# load and plot test results
with open('results.pkl', 'rb') as f:
    (time_arr_SGD_n, loss_arr_SGD_n, dL_dt_arr_SGD_n, \
    time_arr_teleport_lstm_lr_n, loss_arr_teleport_lstm_lr_n, dL_dt_arr_teleport_lstm_lr_n, lr_arr_teleport_lstm_lr_n, \
    time_arr_lstm_update_tele_n, loss_arr_lstm_update_tele_n, dL_dt_arr_lstm_update_tele_n, lr_arr_lstm_update_tele_n, \
    time_arr_lstm_update_n, loss_arr_lstm_update_n, dL_dt_arr_lstm_update_n, lr_arr_lstm_update_n) = pickle.load(f)

plot_all([time_arr_SGD_n, time_arr_teleport_lstm_lr_n, time_arr_lstm_update_n, time_arr_lstm_update_tele_n], \
    [loss_arr_SGD_n, loss_arr_teleport_lstm_lr_n, loss_arr_lstm_update_n, loss_arr_lstm_update_tele_n], \
    [dL_dt_arr_SGD_n, dL_dt_arr_teleport_lstm_lr_n, dL_dt_arr_lstm_update_n, dL_dt_arr_lstm_update_tele_n], \
    ['GD', 'LSTM(lr,tele)', 'LSTM(update)', 'LSTM(update,tele)'], n_epoch=30)

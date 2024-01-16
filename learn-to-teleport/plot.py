""" Helper functions for figures. """

import numpy as np
from matplotlib import pyplot as plt
import os


def plot_all(time_arr_list, loss_arr_list, dL_dt_arr_list, label_list, n_epoch=300):
    if not os.path.exists('figures'):
        os.mkdir('figures')

    # compute mean and std across multiple runs
    loss_mean_list = []
    loss_std_list = []
    time_mean_list = []
    time_std_list = []
    for i in range(len(time_arr_list)):
        time_arr_list[i] = np.array(time_arr_list[i])[:, :n_epoch]
        loss_arr_list[i] = np.array(loss_arr_list[i])[:, :n_epoch]
        dL_dt_arr_list[i] = np.array(dL_dt_arr_list[i])[:, :n_epoch]
        loss_mean_list.append(np.mean(loss_arr_list[i], axis=0))
        loss_std_list.append(np.std(loss_arr_list[i], axis=0))
        time_mean_list.append(np.mean(time_arr_list[i], axis=0))
        time_std_list.append(np.std(time_arr_list[i], axis=0))

    # plot loss vs epoch
    plt.figure()
    for i in range(len(time_arr_list)):
        plt.plot(loss_mean_list[i], linewidth=3, label=label_list[i])
    plt.gca().set_prop_cycle(None)
    for i in range(len(time_arr_list)):
        plt.fill_between(np.arange(n_epoch), loss_mean_list[i]-loss_std_list[i], loss_mean_list[i]+loss_std_list[i], alpha=0.5)
    plt.xlabel('Epoch', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    plt.yscale('log')
    plt.xticks([0, 10, 20, 30], fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.legend(fontsize=17)
    plt.savefig('figures/multi_layer_loss.pdf', bbox_inches='tight')

    # plot loss vs wall-clock time
    plt.figure()
    for i in range(len(time_arr_list)):
        plt.plot(time_mean_list[i], loss_mean_list[i], linewidth=3, label=label_list[i])
        plt.fill_between(time_mean_list[i], loss_mean_list[i]-loss_std_list[i], loss_mean_list[i]+loss_std_list[i], alpha=0.5)
    plt.xlabel('time (s)', fontsize=26)
    plt.ylabel('Loss', fontsize=26)
    max_t = np.max(time_arr_list[0])
    interval = np.round(max_t * 0.3, 2)
    plt.xticks([0, interval, interval * 2, interval * 3], fontsize= 20)
    plt.yticks(fontsize= 20)
    plt.yscale('log')
    plt.legend(fontsize=17)
    plt.savefig('figures/multi_layer_loss_vs_time.pdf', bbox_inches='tight')

    # plot loss vs dL/dt
    plt.figure()
    for i in range(len(time_arr_list)):
        plt.plot(loss_arr_list[i][-1], dL_dt_arr_list[i][-1], linewidth=3, label=label_list[i])
    plt.xlabel('Loss', fontsize=26)
    plt.ylabel('dL/dt', fontsize=26)
    plt.yscale('log')
    plt.xscale('log')
    plt.xticks(fontsize= 20)
    plt.yticks([1e1, 1e3, 1e5, 1e7], fontsize= 20)
    plt.legend(fontsize=17)
    plt.savefig('figures/multi_layer_loss_vs_gradient.pdf', bbox_inches='tight')

    return

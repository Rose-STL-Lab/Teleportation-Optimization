""" Helper functions for figures. """

import numpy as np
from matplotlib import pyplot as plt
import os
import pickle
from scipy.stats import pearsonr

def plot_optimization(opt_method_list, dataset, lr):
    x_right = 40
    for opt_method in opt_method_list:   
        if opt_method == 'Adagrad':
            y_ticks = [0.2, 0.3, 0.4, 0.5]
        elif opt_method == 'momentum':
            y_ticks = [0.1, 0.3, 0.5, 0.7]
        elif opt_method == 'RMSprop':
            y_ticks = [0.2, 0.4, 0.6, 0.8]
        elif opt_method == 'Adam':
            y_ticks = [0.3, 0.5, 0.7, 0.9]

        plt.figure()
        train_loss_list = []
        valid_loss_list = []
        time_list = []
        train_loss_teleport_list = []
        valid_loss_teleport_list = []
        time_teleport_list = []
        for run_num in range(5):
            with open('logs/optimization/{}/{}_{}_lr_{}_{}.pkl'.format(dataset, dataset, opt_method, lr, run_num), 'rb') as f:
                loss_arr_SGD, valid_loss_SGD, _, _, time_SGD, _ = pickle.load(f)
            with open('logs/optimization/{}/{}_{}_lr_{}_teleport_{}.pkl'.format(dataset, dataset, opt_method, lr, run_num), 'rb') as f:
                loss_arr_teleport, valid_loss_teleport, _, _, time_teleport, _ = pickle.load(f)
            train_loss_list.append(loss_arr_SGD)
            valid_loss_list.append(valid_loss_SGD)
            time_list.append(time_SGD)
            time_teleport_list.append(time_teleport)

            train_loss_teleport_list.append(loss_arr_teleport)
            valid_loss_teleport_list.append(valid_loss_teleport)

        time_mean = np.mean(time_list, axis=0)
        time_teleport_mean = np.mean(time_teleport_list, axis=0)

        train_loss_teleport_mean = np.mean(train_loss_teleport_list, axis=0)
        train_loss_teleport_std = np.std(train_loss_teleport_list, axis=0)
        valid_loss_teleport_mean = np.mean(valid_loss_teleport_list, axis=0)
        valid_loss_teleport_std = np.std(valid_loss_teleport_list, axis=0)

        train_loss_SGD_mean = np.mean(train_loss_list, axis=0)
        train_loss_SGD_std = np.std(train_loss_list, axis=0)
        valid_loss_SGD_mean = np.mean(valid_loss_list, axis=0)
        valid_loss_SGD_std = np.std(valid_loss_list, axis=0)


        plt.figure()
        plt.plot(train_loss_SGD_mean[:x_right], '--', linewidth=3, color='#1f77b4', label='{} train'.format(opt_method))
        plt.plot(valid_loss_SGD_mean[:x_right], '-', linewidth=3, color='#1f77b4', label='{} test'.format(opt_method))
        plt.plot(train_loss_teleport_mean[:x_right], '--', linewidth=3, color='#ff7f0e', label='{}+teleport train'.format(opt_method))
        plt.plot(valid_loss_teleport_mean[:x_right], '-', linewidth=3, color='#ff7f0e', label='{}_teleport test'.format(opt_method))

        N = len(train_loss_SGD_mean)
        plt.fill_between(np.arange(N), \
                        valid_loss_SGD_mean[:x_right] - valid_loss_SGD_std[:x_right], \
                        valid_loss_SGD_mean[:x_right] + valid_loss_SGD_std[:x_right], \
                        color='#1f77b4', alpha=0.5)
        plt.fill_between(np.arange(N), \
                        valid_loss_teleport_mean[:x_right] - valid_loss_teleport_std[:x_right], \
                        valid_loss_teleport_mean[:x_right] + valid_loss_teleport_std[:x_right], \
                        color='#ff7f0e', alpha=0.5)

        plt.xlabel('Epoch', fontsize=28)
        plt.ylabel('Loss', fontsize=28)
        plt.yscale('log')
        plt.minorticks_off()
        plt.xticks([0, 10, 20, 30, 40], fontsize= 22)
        plt.yticks(y_ticks, y_ticks, fontsize= 22)
        plt.legend(fontsize=19)
        plt.savefig('figures/optimization/{}_{}_loss_vs_epoch.pdf'.format(dataset, opt_method), bbox_inches='tight')


        fig = plt.subplots()
        plt.plot(time_mean, train_loss_SGD_mean[:x_right], '--', linewidth=3, color='#1f77b4', label='{} train'.format(opt_method))
        plt.plot(time_mean, valid_loss_SGD_mean[:x_right], '-', linewidth=3, color='#1f77b4', label='{} test'.format(opt_method))
        plt.plot(time_teleport_mean, train_loss_teleport_mean[:x_right], '--', linewidth=3, color='#ff7f0e', label='{}+teleport train'.format(opt_method))
        plt.plot(time_teleport_mean, valid_loss_teleport_mean[:x_right], '-', linewidth=3, color='#ff7f0e', label='{}+teleport test'.format(opt_method))

        N = len(train_loss_SGD_mean)
        plt.fill_between(time_mean, \
                        valid_loss_SGD_mean[:x_right] - valid_loss_SGD_std[:x_right], \
                        valid_loss_SGD_mean[:x_right] + valid_loss_SGD_std[:x_right], \
                        color='#1f77b4', alpha=0.5)
        plt.fill_between(time_teleport_mean, \
                        valid_loss_teleport_mean[:x_right] - valid_loss_teleport_std[:x_right], \
                        valid_loss_teleport_mean[:x_right] + valid_loss_teleport_std[:x_right], \
                        color='#ff7f0e', alpha=0.5)

        plt.xlabel('Time (s)', fontsize=28)
        plt.ylabel('Loss', fontsize=28)
        plt.yscale('log')
        plt.minorticks_off()
        plt.xticks(fontsize= 22)
        plt.yticks(y_ticks, y_ticks, fontsize= 22)
        plt.legend(fontsize=19)
        plt.savefig('figures/optimization/{}_{}_loss_vs_time.pdf'.format(dataset, opt_method), bbox_inches='tight')


def plot_correlation(dataset, sigma_name):
    with open('logs/correlation/{}/{}_final_W_lists/curvatures_all_{}.pkl'.format(dataset, dataset, sigma_name), 'rb') as f:
        curvature_mean_list, perturb_mean_list, valid_loss_list, train_loss_list = pickle.load(f)

    plt.figure()
    corr, _ = pearsonr(curvature_mean_list, valid_loss_list)
    plt.scatter(curvature_mean_list, valid_loss_list, label='Corr={:.3f}'.format(corr))
    plt.xlabel(r'$\psi$', fontsize=26)
    plt.ylabel('validation loss', fontsize=26)
    plt.yticks(fontsize= 20)
    if dataset == 'MNIST':
        plt.xlim(0.0005, 0.0035)
        plt.xticks([0.001, 0.002, 0.003], fontsize=20)
    elif dataset == 'FashionMNIST':
        plt.xlim(0.0005, 0.0055)
        plt.xticks([0.001, 0.003, 0.005], fontsize=20)
    else:
        plt.xticks([0.0003, 0.0006, 0.0009], fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('figures/correlation/{}_{}_loss_vs_curvature.pdf'.format(dataset, sigma_name), bbox_inches='tight')

    plt.figure()
    corr, _ = pearsonr(perturb_mean_list, valid_loss_list)
    plt.scatter(perturb_mean_list, valid_loss_list, label='Corr={:.3f}'.format(corr))
    plt.xlabel(r'$\phi$', fontsize=26)
    plt.ylabel('validation loss', fontsize=26)
    plt.yticks(fontsize= 20)
    if dataset == 'MNIST':
        plt.xticks([0.0005, 0.0006, 0.0007], fontsize=20)
    elif dataset == 'FashionMNIST':
        plt.xticks([0.00144, 0.00153, 0.00162], fontsize=20)
    else:
        plt.xticks([0.0057, 0.0060, 0.0063], fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('figures/correlation/{}_{}_loss_vs_perturbed_loss.pdf'.format(dataset, sigma_name), bbox_inches='tight')

    plt.figure()
    corr, _ = pearsonr(perturb_mean_list, curvature_mean_list)
    plt.scatter(perturb_mean_list, curvature_mean_list, label='Corr={:.3f}'.format(corr))
    plt.xlabel(r'$\phi$', fontsize=26)
    plt.ylabel(r'$\psi$', fontsize=26)
    if dataset == 'MNIST':
        plt.xticks([0.0005, 0.0006, 0.0007], fontsize=20)
        plt.ylim(0.0005, 0.0035)
        plt.yticks([0.001, 0.002, 0.003], fontsize=20)
    elif dataset == 'FashionMNIST':
        plt.xticks([0.00144, 0.00153, 0.00162], fontsize=20)
        plt.ylim(0.0005, 0.0055)
        plt.yticks([0.001, 0.003, 0.005], fontsize=20)
    else:
        plt.xticks([0.0057, 0.0060, 0.0063], fontsize=20)
        plt.yticks([0.0003, 0.0006, 0.0009], fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('figures/correlation/{}_{}_curvature_vs_perturbed_loss.pdf'.format(dataset, sigma_name), bbox_inches='tight')


def plot_sharpness_curvature(dataset, objective_list):
    x_right = 40

    for objective in objective_list:
        if objective == 'sharpness':
            variable_name = 'phi'
        else:
            variable_name = 'psi'

        train_loss_list = []
        valid_loss_list = []
        train_loss_teleport_true_list = []
        valid_loss_teleport_true_list = []
        train_loss_teleport_false_list = []
        valid_loss_teleport_false_list = []

        for run_num in range(5):
            with open('logs/generalization/{}/teleport_{}/teleport_{}_true_{}.plk'.format(dataset, objective, objective, run_num), 'rb') as f:
                train_loss, train_loss_teleport, valid_loss, valid_loss_teleport = pickle.load(f)
            train_loss_list.append(train_loss)
            train_loss_teleport_true_list.append(train_loss_teleport)
            valid_loss_list.append(valid_loss)
            valid_loss_teleport_true_list.append(valid_loss_teleport)

            with open('logs/generalization/{}/teleport_{}/teleport_{}_false_{}.plk'.format(dataset, objective, objective, run_num), 'rb') as f:
                train_loss, train_loss_teleport, valid_loss, valid_loss_teleport = pickle.load(f)
            train_loss_teleport_false_list.append(train_loss_teleport)
            valid_loss_teleport_false_list.append(valid_loss_teleport)

        train_loss_teleport_true_mean = np.mean(train_loss_teleport_true_list, axis=0)
        train_loss_teleport_true_std = np.std(train_loss_teleport_true_list, axis=0)
        valid_loss_teleport_true_mean = np.mean(valid_loss_teleport_true_list, axis=0)
        valid_loss_teleport_true_std = np.std(valid_loss_teleport_true_list, axis=0)

        train_loss_teleport_false_mean = np.mean(train_loss_teleport_false_list, axis=0)
        train_loss_teleport_false_std = np.std(train_loss_teleport_false_list, axis=0)
        valid_loss_teleport_false_mean = np.mean(valid_loss_teleport_false_list, axis=0)
        valid_loss_teleport_false_std = np.std(valid_loss_teleport_false_list, axis=0)


        # '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        plt.figure()
        plt.plot(train_loss_list[0][:x_right], '--', linewidth=3, color='black') # label='SGD train'
        plt.plot(valid_loss_list[0][:x_right], '-', linewidth=3, color='black', label='SGD') # label='SGD valid'
        plt.plot(train_loss_teleport_true_mean[:x_right], '--', linewidth=3, color='#1f77b4') #label=r'teleport(decrease $\{}$) train'.format(variable_name)
        plt.plot(valid_loss_teleport_true_mean[:x_right], '-', linewidth=3, color='#1f77b4', label=r'teleport(decrease $\{}$)'.format(variable_name)) #label=r'teleport(decrease $\{}$) valid'.format(variable_name)
        plt.plot(train_loss_teleport_false_mean[:x_right], '--', linewidth=3, color='#ff7f0e') #label=r'teleport(increase $\{}$) train'.format(variable_name)
        plt.plot(valid_loss_teleport_false_mean[:x_right], '-', linewidth=3, color='#ff7f0e', label=r'teleport(increase $\{}$)'.format(variable_name)) #label=r'teleport(increase $\{}$) valid'.format(variable_name)

        N = len(train_loss_teleport_false_mean)
        plt.fill_between(np.arange(N-1), \
                        train_loss_teleport_true_mean[:x_right] - train_loss_teleport_true_std[:x_right], \
                        train_loss_teleport_true_mean[:x_right] + train_loss_teleport_true_std[:x_right], \
                        color='#1f77b4', alpha=0.5)

        plt.fill_between(np.arange(N-1), \
                        valid_loss_teleport_true_mean[:x_right] - valid_loss_teleport_true_std[:x_right], \
                        valid_loss_teleport_true_mean[:x_right] + valid_loss_teleport_true_std[:x_right], \
                        color='#1f77b4', alpha=0.5)

        plt.fill_between(np.arange(N-1), \
                        train_loss_teleport_false_mean[:x_right] - train_loss_teleport_false_std[:x_right], \
                        train_loss_teleport_false_mean[:x_right] + train_loss_teleport_false_std[:x_right], \
                        color='#ff7f0e', alpha=0.5)

        plt.fill_between(np.arange(N-1), \
                        valid_loss_teleport_false_mean[:x_right] - valid_loss_teleport_false_std[:x_right], \
                        valid_loss_teleport_false_mean[:x_right] + valid_loss_teleport_false_std[:x_right], \
                        color='#ff7f0e', alpha=0.5)

        plt.xlabel('Epoch', fontsize=26)
        plt.ylabel('Loss', fontsize=26)
        plt.xticks([0, 20, 40], fontsize= 20)
        if dataset == 'MNIST':
            plt.yticks([0.1, 0.5, 0.9, 1.3], fontsize= 20)
        elif dataset == 'FashionMNIST':
            plt.yticks([0.3, 0.7, 1.1], fontsize= 20)
        elif dataset == 'CIFAR10':
            plt.yticks([1.4, 1.7, 2.0], fontsize= 20)
        plt.legend(fontsize=17)
        plt.savefig('figures/generalization/{}_loss_{}.pdf'.format(dataset, objective), bbox_inches='tight')
        
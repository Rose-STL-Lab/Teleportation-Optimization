""" Meta-learning algorithms for training MLPs. """

import numpy as np
import time
import torch
from torch import nn
from torch.autograd import Variable
from teleportation import teleport_MLP_random, teleport_MLP_gradient_ascent, teleport_MLP
from lstm import LSTM_tele, LSTM_tele_lr, LSTM_local_update

def detach_var(v):
    # make gradient an independent variable that is independent from the rest of the computational graph
    var = Variable(v.data, requires_grad=True)
    var.retain_grad()
    return var

def init_param(dim, seed=12345):
    # dim: list of dimensions of weight matrices. 
    # Example: [4, 5, 6, 7, 8] -> X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4
    torch.manual_seed(seed)
    W_list = []
    for i in range(len(dim) - 2):
        W_list.append(torch.rand(dim[i+2], dim[i+1], requires_grad=True))
    X = torch.rand(dim[1], dim[0], requires_grad=True)
    Y = torch.rand(dim[-1], dim[0], requires_grad=True)
    return W_list, X, Y

def W_list_to_vec(W_list):
    W_vec_all = torch.flatten(W_list[0])
    for i in range(1, len(W_list)):
        W_vec = torch.flatten(W_list[i])
        W_vec_all = torch.concat((W_vec_all, W_vec))
    return W_vec_all

def vec_to_W_list(W_vec_all, dim):
    W_list = []
    start_idx = 0
    for i in range(len(dim)-2):
        end_idx = start_idx + dim[i+2]*dim[i+1]
        W_list.append(torch.reshape(W_vec_all[start_idx:end_idx], (dim[i+2], dim[i+1])))
        start_idx = end_idx
    return W_list
    
def loss_multi_layer(W_list, X, Y, sigma=nn.LeakyReLU(0.1)):
    h = X
    for i in range(len(W_list)-1):
        h = sigma(torch.matmul(W_list[i], h))
    return 0.5 * torch.norm(Y - torch.matmul(W_list[-1], h)) ** 2

def train_epoch_GD(W_list, X, Y, lr):
    L = loss_multi_layer(W_list, X, Y)
    dL_dW_list = torch.autograd.grad(L, inputs=W_list, retain_graph=True)
    dL_dt = 0
    for i in range(len(W_list)):
        W_list[i] = W_list[i] - lr * dL_dW_list[i]
        dL_dt += torch.norm(dL_dW_list[i])**2 
    return W_list, L, dL_dt, dL_dW_list


def train_GD(dim, n_run=5, n_epoch=300, K=[5], teleport=False, random=False, lr=1e-4, lr_teleport=1e-7, T_magnitude=1.0):
    """ Run gradient descent with or without teleportation.

    Args:
        dim: list of dimensions of weight matrices. Example: [4, 5, 6, 7, 8] -> 
          X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4
        n_run: number of runs using different random seeds for initialization.
        n_epoch: number of epochs of each run.
        K: teleportation schedule. Teleportation is performed at epochs in K if teleport=True.
        teleport: True if using teleportation, False otherwise.
        random: True if teleporting using random group element, False if using optimized group element.
        lr: learning rate for gradient descent.
        lr_teleport: learning rate for gradient ascent on group element during teleportation.
        T_magnitude: frobenius norm of elements in T after normalization.
        
    Returns:
        time_arr_SGD_n: Wall-clock time at each epoch. Dimension n_run x n_epoch.
        loss_arr_SGD_n: Loss after each epoch. Dimension n_run x n_epoch.
        dL_dt_arr_SGD_n: Squared gradient norm at each epoch. Dimension n_run x n_epoch.
    """

    time_arr_SGD_n = []
    loss_arr_SGD_n = []
    dL_dt_arr_SGD_n = []

    for n in range(n_run):
        W_list, X, Y = init_param(dim, seed=n*n*12345)
        time_arr_SGD = []
        loss_arr_SGD = []
        dL_dt_arr_SGD = []

        t0 = time.time()
        for epoch in range(n_epoch):
            if teleport == True and epoch in K:
                if random:
                    W_list = teleport_MLP_random(W_list, X, T_magnitude, dim)
                else:
                    W_list = teleport_MLP_gradient_ascent(W_list, X, Y, lr_teleport, dim, loss_multi_layer, 8)

            W_list, loss, dL_dt, _ = train_epoch_GD(W_list, X, Y, lr)
            t1 = time.time()
            time_arr_SGD.append(t1 - t0)
            loss_arr_SGD.append(loss.detach().numpy())
            dL_dt_arr_SGD.append(dL_dt.detach().numpy())

        time_arr_SGD_n.append(time_arr_SGD)
        loss_arr_SGD_n.append(loss_arr_SGD)
        dL_dt_arr_SGD_n.append(dL_dt_arr_SGD)

    return time_arr_SGD_n, loss_arr_SGD_n, dL_dt_arr_SGD_n


def train_meta_opt(dim, n_run=20, n_epoch=20, unroll=5, lr=1e-4, lr_meta=1e-3, learn_lr=True, learn_tele=True, learn_update=True, T_magnitude=0.01):
    """ Run gradient descent with or without teleportation.

    Args:
        dim: list of dimensions of weight matrices. Example: [4, 5, 6, 7, 8] -> 
          X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4
        n_run: number of runs using different random seeds for initialization.
        n_epoch: number of epochs of each run.
        unroll: number of steps before each update of the parameters in the meta-optimizers
        lr: learning rate for gradient descent of the MLP parameters.
        lr_meta: learning rate for meta-optimizers.
        learn_lr: True if meta-optimizers learn lr, False otherwise.
        learn_tele: True if meta-optimizers learn teleportation, False if no teleportation is applied.
        learn_update: True if meta-optimizers learn local updates, False otherwise.
        T_magnitude: frobenius norm of elements in T after normalization.
        
    Returns:
        meta_opt_list: a meta_optimizer that outputs the group elements for teleportation
        meta_opt_update: a meta_optimizer that outputs the local update for MLP parameters. None if learn_update=False.
    """

    # initialize a list of meta_opt, one for each pair of weights (for teleportation)
    meta_opt_list = []
    optimizer_list = []
    n_meta_opt = len(dim) - 3

    for i in range(n_meta_opt):
        n_param = dim[i+1] * dim[i+2] + dim[i+2] * dim[i+3]
        if learn_update:
            meta_opt = LSTM_tele(n_param, 300, dim[i+2])
        else:
            if learn_lr==False:
                meta_opt = LSTM_tele(n_param, 20, dim[i+2])
            else:
                meta_opt = LSTM_tele_lr(n_param, 20, dim[i+2])

        meta_opt_list.append(meta_opt)
        optimizer = torch.optim.Adam(meta_opt_list[i].parameters(), lr=1e-4)
        optimizer_list.append(optimizer)

        meta_opt_list[i].train()

    # initialize a meta_opt for all weights (for update step)
    if learn_update:
        W_list, _, _ = init_param(dim, seed=12345)
        n_param = W_list_to_vec(W_list).shape[0]
        meta_opt_update = LSTM_local_update(n_param, 300, n_param)
        optimizer_update = torch.optim.Adam(meta_opt_update.parameters(), lr=lr_meta)
    else:
        meta_opt_update = None
        optimizer_update = None

    # for each of the n_run training trajectories
    for n in range(n_run):
        if n % 100 == 0:
            print("run", n)
        W_list, X, Y = init_param(dim, seed=n*n*12345-1)
        X_inv = torch.linalg.pinv(X)
        loss_sum = None
        loss_sum_all = 0.0

        # initialize LSTM hidden and cell
        hidden = []
        cell = []
        for i in range(n_meta_opt):
            cell.append(Variable(torch.zeros(2, 1, meta_opt.lstm_hidden_dim), requires_grad=True))
            hidden.append(Variable(torch.zeros(2, 1, meta_opt.lstm_hidden_dim), requires_grad=True)) 

        if learn_update: # learn local updates
            cell_update = Variable(torch.zeros(2, 1, meta_opt_update.lstm_hidden_dim), requires_grad=True)
            hidden_update = Variable(torch.zeros(2, 1, meta_opt_update.lstm_hidden_dim), requires_grad=True)

            for epoch in range(n_epoch):
                # compute loss gradients, compute local updates from meta optimizer, perform local updates for MLP parameters
                loss = loss_multi_layer(W_list, X, Y)
                dL_dW_list = torch.autograd.grad(loss, inputs=W_list, retain_graph=True)
                W_update, hidden_update, cell_update = meta_opt_update(W_list_to_vec(dL_dW_list), hidden_update, cell_update)
                W_list = vec_to_W_list(W_list_to_vec(W_list) - W_update, dim)

                loss_sum_all += loss.data
                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss

                # update meta optimizers
                if epoch % unroll == 0 and epoch != 0: 
                    for i in range(n_meta_opt):
                        optimizer_list[i].zero_grad()
                    optimizer_update.zero_grad()
                    loss_sum.backward(retain_graph=True)

                    for i in range(n_meta_opt):
                        optimizer_list[i].step()
                    optimizer_update.step()

                    loss_sum = None
                    hidden = [detach_var(v) for v in hidden]
                    cell = [detach_var(v) for v in cell]
                    hidden_update = detach_var(hidden_update)
                    cell_update = detach_var(cell_update)
                    W_list = [detach_var(v) for v in W_list]

                # compute group elements from meta optimizers and teleport MLP parameters
                if learn_tele == True:
                    g_list = []
                    for i in range(n_meta_opt):
                        g, hidden[i], cell[i] = meta_opt_list[i](dL_dW_list[i], dL_dW_list[i+1], hidden[i], cell[i])
                        g_list.append(g)
                    W_list = teleport_MLP(W_list, X, X_inv, g_list, using_T=True, T_magnitude=T_magnitude)

        else: # does not learn local updates
            for epoch in range(n_epoch):
                # one gradient descent step on MLP parameters, using learned learning rate if learn_lr=True
                if learn_lr==False or epoch == 0:
                    W_list, loss, dL_dt, dL_dW_list = train_epoch_GD(W_list, X, Y, lr)
                else:
                    learned_lr = torch.mean(torch.stack(step_size_list), dim=0)
                    W_list, loss, dL_dt, dL_dW_list = train_epoch_GD(W_list, X, Y, learned_lr)

                loss_sum_all += loss.data
                if loss_sum is None:
                    loss_sum = loss
                else:
                    loss_sum += loss

                # update meta optimizers
                if epoch % unroll == 0 and epoch != 0:
                    for i in range(n_meta_opt):
                        optimizer_list[i].zero_grad()
                    loss_sum.backward(retain_graph=True)

                    for i in range(n_meta_opt):
                        optimizer_list[i].step()

                    loss_sum = None
                    hidden = [detach_var(v) for v in hidden]
                    cell = [detach_var(v) for v in cell]
                    W_list = [detach_var(v) for v in W_list]

                # compute group elements from meta optimizers and teleport MLP parameters
                g_list = []
                step_size_list = []
                for i in range(n_meta_opt):
                    if learn_lr == True:
                        g, step_size, hidden[i], cell[i] = meta_opt_list[i](dL_dW_list[i], dL_dW_list[i+1], hidden[i], cell[i])
                    else:
                        g, hidden[i], cell[i] = meta_opt_list[i](dL_dW_list[i], dL_dW_list[i+1], hidden[i], cell[i])
                        step_size = None
                    g_list.append(g)
                    step_size_list.append(step_size)

                W_list = teleport_MLP(W_list, X, X_inv, g_list, using_T=True, T_magnitude=0.01)

    return meta_opt_list, meta_opt_update


def test_meta_opt(meta_opt_list, meta_opt_update, dim, n_run=5, n_epoch=300, lr=1e-4, learn_lr=False, learn_tele=True, learn_update=True, T_magnitude=0.01):
    """ Run gradient descent with or without teleportation.

    Args:
        meta_opt_list: a meta_optimizer that outputs the group elements for teleportation
        meta_opt_update: a meta_optimizer that outputs the local update for MLP parameters. None if learn_update=False.
        dim: list of dimensions of weight matrices. Example: [4, 5, 6, 7, 8] -> 
          X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4
        n_run: number of runs using different random seeds for initialization.
        n_epoch: number of epochs of each run.
        lr: learning rate for gradient descent of the MLP parameters.
        learn_lr: True if meta-optimizers learn lr, False otherwise.
        learn_tele: True if meta-optimizers learn teleportation, False if no teleportation is applied.
        learn_update: True if meta-optimizers learn local updates, False otherwise.
        T_magnitude: frobenius norm of elements in T after normalization.
        
    Returns:
        time_arr_teleport_n: Wall-clock time at each epoch. Dimension n_run x n_epoch.
        loss_arr_teleport_n: Loss after each epoch. Dimension n_run x n_epoch.
        dL_dt_arr_teleport_n: Squared gradient norm at each epoch. Dimension n_run x n_epoch.
        lr_arr_teleport_n: Learning rate for MLP parameters at each epoch. Dimension n_run x n_epoch.
    """

    time_arr_teleport_n = []
    loss_arr_teleport_n = []
    dL_dt_arr_teleport_n = []
    lr_arr_teleport_n = []

    n_meta_opt = len(dim) - 3
    for i in range(n_meta_opt):
        meta_opt_list[i].eval() # list of meta_opt, one for each pair of weights (for teleportation)

    if learn_update:
        meta_opt_update.eval() # meta_opt for all weights (for update step)

    for n in range(n_run):
        W_list, X, Y = init_param(dim, seed=n*n*12345)
        X_inv = torch.linalg.pinv(X)

        # initialize LSTM hidden and cell
        hidden = []
        cell = []
        for i in range(n_meta_opt):
            cell.append(Variable(torch.zeros(2, 1, meta_opt_list[0].lstm_hidden_dim), requires_grad=True))
            hidden.append(Variable(torch.zeros(2, 1, meta_opt_list[0].lstm_hidden_dim), requires_grad=True)) 

        if learn_update:
            cell_update = Variable(torch.zeros(2, 1, meta_opt_update.lstm_hidden_dim), requires_grad=True)
            hidden_update = Variable(torch.zeros(2, 1, meta_opt_update.lstm_hidden_dim), requires_grad=True)

        time_arr_teleport = []
        loss_arr_teleport = []
        dL_dt_arr_teleport = []
        lr_arr_teleport = []

        t0 = time.time()
        for epoch in range(n_epoch):
            if learn_update: # learn local updates
                # compute loss gradients, compute local updates from meta optimizer, perform local updates for MLP parameters
                loss = loss_multi_layer(W_list, X, Y)
                dL_dW_list = torch.autograd.grad(loss, inputs=W_list, retain_graph=True)
                W_update, hidden_update, cell_update = meta_opt_update(W_list_to_vec(dL_dW_list), hidden_update, cell_update)
                W_list = vec_to_W_list(W_list_to_vec(W_list) - W_update, dim)
                dL_dt = torch.norm(W_list_to_vec(W_list))**2 

                # compute group elements from meta optimizers and teleport MLP parameters
                g_list = []
                step_size_list = []
                for i in range(n_meta_opt):
                    g, hidden[i], cell[i] = meta_opt_list[i](dL_dW_list[i], dL_dW_list[i+1], hidden[i], cell[i])
                    step_size = None
                    g_list.append(g)
                    step_size_list.append(step_size)
                if learn_tele == True:
                    W_list = teleport_MLP(W_list, X, X_inv, g_list, using_T=True, T_magnitude=T_magnitude)

            else: # does not learn local updates
                # one gradient descent step on MLP parameters, using learned learning rate if learn_lr=True
                if learn_lr==False or epoch == 0:
                    W_list, loss, dL_dt, dL_dW_list = train_epoch_GD(W_list, X, Y, lr)
                else:
                    learned_lr = torch.mean(torch.stack(step_size_list), dim=0)
                    W_list, loss, dL_dt, dL_dW_list = train_epoch_GD(W_list, X, Y, learned_lr)
                    lr_arr_teleport.append(learned_lr.detach().numpy())

                # compute group elements from meta optimizers and teleport MLP parameters
                g_list = []
                step_size_list = []
                for i in range(n_meta_opt):
                    if learn_lr == True:
                        g, step_size, hidden[i], cell[i] = meta_opt_list[i](dL_dW_list[i], dL_dW_list[i+1], hidden[i], cell[i])
                    else:
                        g, hidden[i], cell[i] = meta_opt_list[i](dL_dW_list[i], dL_dW_list[i+1], hidden[i], cell[i])
                        step_size = None
                    g_list.append(g)
                    step_size_list.append(step_size)
                
                teleport_MLP(W_list, X, X_inv, g_list, using_T=True, T_magnitude=0.01)

            t1 = time.time()
            time_arr_teleport.append(t1 - t0)
            loss_arr_teleport.append(loss.detach().numpy())
            dL_dt_arr_teleport.append(dL_dt.detach().numpy())

        time_arr_teleport_n.append(time_arr_teleport)
        loss_arr_teleport_n.append(loss_arr_teleport)
        dL_dt_arr_teleport_n.append(dL_dt_arr_teleport)
        lr_arr_teleport_n.append(lr_arr_teleport)

    return time_arr_teleport_n, loss_arr_teleport_n, dL_dt_arr_teleport_n, lr_arr_teleport_n

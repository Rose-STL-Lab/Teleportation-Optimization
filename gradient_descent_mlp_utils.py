""" Functions for gradient descent and teleportations. """

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from curvature_utils import W_list_to_vec, vec_to_W_list, compute_curvature

def init_param_MLP(dim, seed=54321):
    # dim: list of dimensions of weight matrices. 
    # Example: [4, 5, 6, 7, 8] -> X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4
    torch.manual_seed(seed)
    W_list = []
    for i in range(len(dim) - 2):
        k = 1 / np.sqrt(dim[i+1]) # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        W = 2 * k * torch.rand(dim[i+2], dim[i+1], requires_grad=True) - k
        W_list.append(W)
    return W_list

def loss_multi_layer(W_list, X, Y, sigma):
    h = X
    for i in range(len(W_list)-1):
        h = sigma(torch.matmul(W_list[i], h))
    pred = torch.matmul(W_list[-1], h)
    pred = F.log_softmax(pred, dim=0)
    return F.nll_loss(torch.t(pred), Y), pred

def loss_MLP_from_vec(W_vec_all, X, Y, dim, sigma):
    W_list = vec_to_W_list(W_vec_all, dim)
    L, _ = loss_multi_layer(W_list, X, Y, sigma)
    return L

def valid_MLP(model, criterion, valid_loader):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    for data, target in valid_loader:
        batch_size = data.shape[0]
        data = torch.t(data.view(batch_size, -1))
        output = model(data)
        L = criterion(output.T, target)
        test_loss += L.item()*data.size(1)

        _, pred = torch.max(output, 0)
        test_correct += pred.eq(target.data.view_as(pred)).sum().item()

    test_loss = test_loss / len(valid_loader.sampler)
    test_correct = 100.0 * test_correct / len(valid_loader.sampler)
    return test_loss, test_correct

def train_step(x_train, y_train, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model.forward(x_train)
    loss = criterion(output.T, y_train)
    loss.backward()
    optimizer.step()
    return loss

def test_MLP(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for data, target in test_loader:
        batch_size = data.shape[0]
        data = torch.t(data.view(batch_size, -1))
        output = model(data)
        L = criterion(output.T, target)
        test_loss += L.item()*data.size(1)
        _, pred = torch.max(output, 0)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss/len(test_loader.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    return test_loss, np.sum(class_correct) / np.sum(class_total)


##############################################################
# group actions
def group_action(U, V, X, X_inv, T, sigma):
    # U, V -> U sigma(VX) sigma((I+T)VX)^+, (I+T)V

    k = list(T.size())[0]
    I = torch.eye(k)

    V_out = torch.matmul((I+T), V)
    Wh = torch.matmul(V, X)
    sigma_Wh = sigma(Wh)
    sigma_gWh = sigma(torch.matmul((I+T), Wh))
    sigma_gWh_inv = torch.linalg.pinv(sigma_gWh)
    U_out = torch.matmul(torch.matmul(U, sigma_Wh), sigma_gWh_inv)
    return U_out, V_out

def group_action_large(U, V, X, X_inv, g, g_inv, sigma):
    # U, V -> U sigma(VX) sigma(gVX)^+, gV

    k = list(g.size())[0]
    I = torch.eye(k)

    V_out = torch.matmul(g, V)
    Wh = torch.matmul(V, X)
    sigma_Wh = sigma(Wh)
    sigma_gWh = sigma(torch.matmul(g, Wh))
    sigma_gWh_inv = torch.linalg.pinv(sigma_gWh)
    U_out = torch.matmul(torch.matmul(U, sigma_Wh), sigma_gWh_inv)
    return U_out, V_out

def group_action_exp(t, U, V, X, X_inv, M, sigma):
    # U, V -> U sigma(VX) sigma(exp(tM)VX)^+, exp(tM)V

    g = torch.linalg.matrix_exp(t * M)
    g_inv = torch.linalg.pinv(g)

    V_out = torch.matmul(g, V)
    Wh = torch.matmul(V, X)
    sigma_Wh = sigma(Wh)
    sigma_gWh = sigma(torch.matmul(g, Wh))
    sigma_gWh_inv = torch.linalg.pinv(sigma_gWh)
    U_out = torch.matmul(torch.matmul(U, sigma_Wh), sigma_gWh_inv)
    return U_out, V_out

##############################################################
# first (or second) derivatives of the component of gamma corresponding to U (or V)
def compute_gamma_1_U(t, U, V, h, h_inv, M, sigma):
    func = lambda t_: group_action_exp(t_, U, V, h, h_inv, M, sigma)[0]
    gamma_1 = torch.autograd.functional.jacobian(func, t, create_graph=True)
    gamma_1 = torch.squeeze(gamma_1)
    return gamma_1

def compute_gamma_1_V(t, U, V, h, h_inv, M, sigma):
    func = lambda t_: group_action_exp(t_, U, V, h, h_inv, M, sigma)[1]
    gamma_1 = torch.autograd.functional.jacobian(func, t, create_graph=True)
    gamma_1 = torch.squeeze(gamma_1)
    return gamma_1

def compute_gamma_2_U(t, U, V, h, h_inv, M, sigma):
    func = lambda t_: compute_gamma_1_U(t_, U, V, h, h_inv, M, sigma)
    gamma_2 = torch.autograd.functional.jacobian(func, t, create_graph=True)
    gamma_2 = torch.squeeze(gamma_2)
    return gamma_2

def compute_gamma_2_V(t, U, V, h, h_inv, M, sigma):
    func = lambda t_: compute_gamma_1_V(t_, U, V, h, h_inv, M, sigma)
    gamma_2 = torch.autograd.functional.jacobian(func, t, create_graph=True)
    gamma_2 = torch.squeeze(gamma_2)
    return gamma_2

##############################################################
# teleportation

def teleport_curvature(W_list, X, Y, lr_teleport, dim, sigma, telestep=10, reverse=False):
    # reverse = True if minimizing curvature, False if maximizing curvature.
    print("before teleport", loss_multi_layer(W_list, X, Y, sigma)[0])

    X_inv = torch.linalg.pinv(X)
    h_list = [X]
    h_inv_list = [X_inv]
    for m in range(0, len(W_list)-2):
        h = sigma(torch.matmul(W_list[m], h_list[-1]))
        h_list.append(h)
        h_inv_list.append(torch.linalg.pinv(h))

    for teleport_step in range(telestep):
        layer = 1
        t = torch.zeros(1, requires_grad=True)
        M = torch.rand(dim[layer+2], dim[layer+2], requires_grad=True)

        # compute curvature using autograd
        gamma_1_U = compute_gamma_1_U(t, W_list[1+layer], W_list[0+layer], h_list[0+layer], h_inv_list[0+layer], M, sigma)
        gamma_1_V = compute_gamma_1_V(t, W_list[1+layer], W_list[0+layer], h_list[0+layer], h_inv_list[0+layer], M, sigma)

        gamma_2_U = compute_gamma_2_U(t, W_list[1+layer], W_list[0+layer], h_list[0+layer], h_inv_list[0+layer], M, sigma)
        gamma_2_V = compute_gamma_2_V(t, W_list[1+layer], W_list[0+layer], h_list[0+layer], h_inv_list[0+layer], M, sigma)
        
        gamma_1_list = []
        gamma_2_list = []
        for m in range(0, len(W_list)):
            gamma_1_list.append(torch.zeros_like(W_list[m]))
            gamma_2_list.append(torch.zeros_like(W_list[m]))
        
        gamma_1_list[0+layer] = gamma_1_U
        gamma_1_list[1+layer] = gamma_1_V
        gamma_2_list[0+layer] = gamma_2_U
        gamma_2_list[1+layer] = gamma_2_V

        kappa = compute_curvature(gamma_1_list, gamma_2_list) # curvature
        kappa_1 = torch.autograd.grad(kappa, inputs=t, create_graph=True)[0] # derivative of curvature
        
        # gradient descent/ascent on t to decrease/increase curvature
        if reverse:
            t = t - lr_teleport * kappa_1
        else:
            t = t + lr_teleport * kappa_1
        print(kappa, kappa_1, t)
        
        # transform weights using the updated t
        g = torch.linalg.matrix_exp(t * M)
        g_inv = torch.linalg.pinv(g) 
        W_list[1+layer], W_list[0+layer] = group_action_exp(t, W_list[1+layer], W_list[0+layer], h_list[0+layer], h_inv_list[0+layer], M, sigma)

        h_list = [X]
        h_inv_list = [X_inv]
        for m in range(0, len(W_list)-2):
            h = sigma(torch.matmul(W_list[m], h_list[-1]))
            h_list.append(h)
            h_inv_list.append(torch.linalg.pinv(h))

    print("after teleport", loss_multi_layer(W_list, X, Y, sigma)[0])
        
    return W_list


def teleport_sharpness(W_list, X, Y, lr_teleport, dim, sigma, telestep=10, loss_perturb_cap=2.0, reverse=False, \
    t_start=0.001, t_end=0.2, t_interval=0.001):
    # reverse = True if minimizing sharpness, False if maximizing sharpness.

    X_inv = torch.linalg.pinv(X)
    h_list = [X]
    h_inv_list = [X_inv]
    for m in range(0, len(W_list)-2):
        h = sigma(torch.matmul(W_list[m], h_list[-1]))
        h_list.append(h)
        h_inv_list.append(torch.linalg.pinv(h))

    for teleport_step in range(telestep):
        gW_list = W_list.copy()
        T = [] # list of elements of Lie algebras

        # initialize T[i] = 0 and g.W = (I+T).W
        for m in range(0, len(gW_list)-1):
            T.append(torch.zeros(dim[m+2], dim[m+2], requires_grad=True))
            gW_list[m+1], gW_list[m] = group_action(gW_list[m+1], gW_list[m], h_list[m], h_inv_list[m], T[m], sigma)

        # compute sharpness (loss_perturb_mean)
        num_t = len(np.arange(0.1, 5.0, 0.5))
        num_d = 100
        loss_perturb_mean = 0.0
        for (idx, t) in enumerate(np.arange(0.1, 1.0, 0.1)):
            for d_idx in range(num_d):
                W_vec_all = W_list_to_vec(gW_list)
                random_dir = torch.rand(W_vec_all.size()[0], requires_grad=True)
                random_dir = random_dir / torch.norm(random_dir) * t
                W_vec_all_perturb = W_vec_all + random_dir
                loss_perturb = loss_MLP_from_vec(W_vec_all_perturb, X, Y, dim, sigma)
                loss_perturb_mean += loss_perturb
        loss_perturb_mean = loss_perturb_mean / num_t / num_d
        print(teleport_step, loss_perturb_mean)
        if loss_perturb_mean > loss_perturb_cap:
            break

        # gradient descent/ascent on T to decrease/increase sharpness (loss_perturb_mean)
        dLdt_dT_list = torch.autograd.grad(loss_perturb_mean, inputs=T, create_graph=True)
        for i in range(len(T)):
            if reverse:
                T[i] = T[i] - lr_teleport * dLdt_dT_list[i]
            else:
                T[i] = T[i] + lr_teleport * dLdt_dT_list[i]

        # transform weights using the updated T
        for m in range(0, len(W_list)-1):
            W_list[m+1], W_list[m] = group_action(W_list[m+1], W_list[m], h_list[m], h_inv_list[m], T[m], sigma)

        # update the list of hidden representations h_list
        for m in range(1, len(h_list)):
            k = list(T[m-1].size())[0]
            I = torch.eye(k)
            h_list[m] = torch.matmul(I + T[m-1], h_list[m])
            h_inv_list[m] = torch.matmul(h_inv_list[m], I - T[m-1])
        
    return W_list


def teleport(W_list, X, Y, lr_teleport, dim, sigma, telestep=10, dL_dt_cap=100, random_teleport=False, reverse=False):
    # teleportation to increase gradient norm

    # print("before teleport", loss_multi_layer(W_list, X, Y, sigma)[0])
    X_inv = torch.linalg.pinv(X)
    h_list = [X]
    h_inv_list = [X_inv]
    for m in range(0, len(W_list)-2):
        h = sigma(torch.matmul(W_list[m], h_list[-1]))
        h_list.append(h)
        h_inv_list.append(torch.linalg.pinv(h))

    if random_teleport == True:
        for m in range(0, len(W_list)-1):
            g = torch.rand(dim[m+2], dim[m+2])
            g = g / torch.norm(g, p='fro', dim=None) * 0.01 + torch.eye(dim[m+2]) * 1e0
            g_inv = torch.linalg.pinv(g)
            W_list[m+1], W_list[m] = group_action_large(W_list[m+1], W_list[m], h_list[m], h_inv_list[m], g, g_inv, sigma)
        return W_list


    for teleport_step in range(telestep):
        # populate gW_list with T.W, where T=I
        gW_list = W_list.copy()
        T = []
        for m in range(0, len(gW_list)-1):
            T.append(torch.zeros(dim[m+2], dim[m+2], requires_grad=True))
            gW_list[m+1], gW_list[m] = group_action(gW_list[m+1], gW_list[m], h_list[m], h_inv_list[m], T[m], sigma)

        # compute L(T.W) and dL/d(T.W)
        L, _ = loss_multi_layer(gW_list, X, Y, sigma)
        dL_dW_list = torch.autograd.grad(L, inputs=gW_list, create_graph=True)

        # compute dL/dt=||dL/d(T.W)||^2 and d/dT dL/dt
        dL_dt = 0
        for i in range(len(gW_list)):
            dL_dt += torch.norm(dL_dW_list[i])**2 

        if dL_dt.detach().numpy() > dL_dt_cap:
            break

        # gradient ascent step on T, in the direction of d/dT dL/dt
        dLdt_dT_list = torch.autograd.grad(dL_dt, inputs=T)
        for i in range(len(T)):
            if reverse:
                T[i] = T[i] - lr_teleport * dLdt_dT_list[i]
            else:
                T[i] = T[i] + lr_teleport * dLdt_dT_list[i]

        # replace original W's with T.W, using the new T's
        for m in range(0, len(W_list)-1):
            W_list[m+1], W_list[m] = group_action(W_list[m+1], W_list[m], h_list[m], h_inv_list[m], T[m], sigma)


        for m in range(1, len(h_list)):
            k = list(T[m-1].size())[0]
            I = torch.eye(k)
            h_list[m] = torch.matmul(I + T[m-1], h_list[m])
            h_inv_list[m] = torch.matmul(h_inv_list[m], I - T[m-1])

    # print("after teleport", loss_multi_layer(W_list, X, Y, sigma)[0])
        
    return W_list

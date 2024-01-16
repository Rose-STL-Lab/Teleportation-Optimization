""" Group actions and teleportation algorithms for MLP. """

import numpy as np
import torch
from torch import nn

sigma = nn.LeakyReLU(0.1)
sigma_inv = nn.LeakyReLU(10)

def group_action_MLP(U, V, X, X_inv, T, using_T=True, sigma=nn.LeakyReLU(0.1), sigma_inv=nn.LeakyReLU(10)):
    """GL(R) group actions on a pair of matrices.

    Performs the group action in equation (8) in https://arxiv.org/pdf/2205.10637.pdf.
    U = W_m, V = W_{m-1}, X = h_{m-2}

    Args:
        U: Matrix with dimension m x k. Weight acting on sigma(VX)
        V: Matrix with dimension k x n. Weight acting on X.
        X: Matrix with dimension m x n. Output from the previous layer.
        X_inv: Matrix with dimension n x m. Inverse of X.
        T: Matrix with dimension k x k. Element in the Lie algebra of GL_k(R)
        using_T: If True, U_out = U (I-T), V_out = sigma^{-1}((I+T)sigma(VX)) X^{-1}
                 If False, U_out = U T^{-1}, V_out = sigma^{-1}(T sigma(VX)) X^{-1}
        sigma: Element-wise activation function.
        sigma_inv: Inverse of sigma.

    Returns:
        U_out: Result of g acting on U. Same dimension as U. 
        V_out: Result of g acting on V. Same dimension as V. 
    """

    k = list(T.size())[0]
    I = torch.eye(k)
    if using_T:
        U_out = torch.matmul(U, (I-T))
        V_out = sigma(torch.matmul(V, X))
        V_out = torch.matmul((I+T), V_out)
        V_out = sigma_inv(V_out)
        V_out = torch.matmul(V_out, X_inv)
    else:
        T_inv = torch.linalg.pinv(T)

        U_out = torch.matmul(U, T_inv)
        V_out = sigma(torch.matmul(V, X))
        V_out = torch.matmul(T, V_out)
        V_out = sigma_inv(V_out)
        V_out = torch.matmul(V_out, X_inv)
    return U_out, V_out

def teleport_MLP(W_list, X, X_inv, T, using_T=True, T_magnitude=None):
    """ GL(R) group actions on all layers in an MLP.

    Args:
        W_list: list of weight matrices.
        X: Data matrix, with dimension a x b. 
        X_inv: Matrix with dimension n x m. Inverse of X.
        T: list of Lie algebra elements used to transform the weight matrices
        T_magnitude: frobenius norm of elements in T

    Returns:
        W_list: Teleported weights. Same shapes as the input W_list.
    """

    # Normalize T's to the specified magnitude
    if T_magnitude != None:
        for m in range(0, len(W_list)-1):
            T[m] = T[m] / torch.norm(T[m], p='fro', dim=None) * T_magnitude

    h = X
    h_inv = X_inv
    h_inv_list = [h_inv]
    for m in range(0, len(W_list)-1):
        W_list[m+1], W_list[m] = group_action_MLP(W_list[m+1], W_list[m], h, h_inv, T[m], using_T)
        h = sigma(torch.matmul(W_list[m], h))
        h_inv = torch.linalg.pinv(h)
        h_inv_list.append(h_inv)

    return W_list

def teleport_MLP_random(W_list, X, magnitude, dim):
    """ Teleportation using random T's with specified magnitude.

    Args:
        W_list: list of weight matrices.
        X: Data matrix, with dimension a x b. 
        magnitude: frobenius norm of elements in T
        dim: list of dimensions of weight matrices. Example: [4, 5, 6, 7, 8] -> 
          X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4

    Returns:
        W_list: Teleported weights. Same shapes as the input W_list.
    """
    X_inv = torch.linalg.pinv(X)
    T = []
    for m in range(0, len(W_list)-1):
        T_m = torch.rand(dim[m+2], dim[m+2], requires_grad=True)
        T_m = T_m / torch.norm(T_m, p='fro', dim=None) * magnitude + torch.eye(dim[m+2]) * 1e0
        T.append(T_m)
    W_list = teleport_MLP(W_list, X, X_inv, T, using_T=False)
    return W_list

def teleport_MLP_gradient_ascent(W_list, X, Y, lr_teleport, dim, loss_func, step=10, sigma=nn.LeakyReLU(0.1)):
    """Teleportation on weight matrices in a multi-layer neural network, using gradient ascent.

    Args:
        W_list: list of weight matrices.
        X: Data matrix, with dimension a x b. 
        Y: Label matrix, with dimension c x b.
        lr_teleport: A scalar. Learning rate used in optimizing the group element.
        dim: list of dimensions of weight matrices. Example: [4, 5, 6, 7, 8] -> 
          X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4
        loss_func: Loss function in the optimization problem.
        step: An integer. Number of gradient ascent steps used to optimize the group
          element.
        sigma: Element-wise activation function.

    Returns:
        W_list: Teleported weights. Same shapes as the input W_list.
    """

    X_inv = torch.linalg.pinv(X)

    for teleport_step in range(step):
        # populate gW_list with T.W, where T=I
        gW_list = W_list.copy()
        T = []
        h = X
        h_inv = X_inv
        for m in range(0, len(gW_list)-1):
            T.append(torch.zeros(dim[m+2], dim[m+2], requires_grad=True))
            gW_list[m+1], gW_list[m] = group_action_MLP(gW_list[m+1], gW_list[m], h, h_inv, T[m])
            h = sigma(torch.matmul(gW_list[m], h))
            h_inv = torch.linalg.pinv(h)

        # compute L(T.W) and dL/d(T.W)
        L = loss_func(gW_list, X, Y)
        dL_dW_list = torch.autograd.grad(L, inputs=gW_list, create_graph=True)

        # compute dL/dt=||dL/d(T.W)||^2 and d/dT dL/dt
        dL_dt = 0
        for i in range(len(gW_list)):
            dL_dt += torch.norm(dL_dW_list[i])**2 
        dLdt_dT_list = torch.autograd.grad(dL_dt, inputs=T)

        # gradient ascent step on T, in the direction of d/dT dL/dt
        for i in range(len(T)):
            T[i] = T[i] + lr_teleport * dLdt_dT_list[i]

        # replace original W's with T.W, using the new T's
        W_list = teleport_MLP(W_list, X, X_inv, T)

    return W_list
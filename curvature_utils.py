""" Functions for computing curvatures. """

import numpy as np
import torch
from torch import nn

sigma = nn.LeakyReLU(0.1)
sigma_inv = nn.LeakyReLU(10)

def sigma_1(x):
    # first derivative of sigma (LeakyReLU)
    x[x > 0] = 1.0
    x[x < 0] = 0.1
    return x

def sigma_inv_1(x):
    # first derivative of sigma^{-1} (LeakyReLU)
    x[x > 0] = 1.0
    x[x < 0] = 10.0
    return x

def W_list_to_vec(W_list):
    # Flatten and concatenate all weight matrices to a vector.
    W_vec_all = torch.flatten(W_list[0])
    for i in range(1, len(W_list)):
        W_vec = torch.flatten(W_list[i])
        W_vec_all = torch.concat((W_vec_all, W_vec))
    return W_vec_all

def vec_to_W_list(W_vec_all, dim):
    # Reshape vectorized weight to matrices.
    # dim: list of dimensions of weight matrices. Example: [4, 5, 6, 7, 8] -> X: 5x4, W1:6x5, W2:7x6, W3:8x7, Y:8x4 
    W_list = []
    start_idx = 0
    for i in range(len(dim)-2):
        end_idx = start_idx + dim[i+2]*dim[i+1]
        W_list.append(torch.reshape(W_vec_all[start_idx:end_idx], (dim[i+2], dim[i+1])))
        start_idx = end_idx
    return W_list

def compute_curvature(gamma_1_list, gamma_2_list):
    """Compute curvature of gamma from its first and second derivatives. (Equation (47) in paper)

    Args:
        gamma_1_list: First derivative of curve gamma(t), d gamma / dt.
        gamma_2_list: Second derivative of curve gamma(t), d^2 gamma / dt^2.

    Returns:
        Curvature of gamma.
    """

    gamma_1_vec = W_list_to_vec(gamma_1_list)
    gamma_2_vec = W_list_to_vec(gamma_2_list)
    gamma_1_norm = torch.norm(gamma_1_vec)
    gamma_2_norm = torch.norm(gamma_2_vec)
    numerator = torch.sqrt(gamma_1_norm**2 * gamma_2_norm**2 - torch.dot(gamma_1_vec, gamma_2_vec)**2)
    denominator = gamma_1_norm**3
    return numerator / denominator

def compute_gamma_12(M_list, W_list, X):
    """Compute the first and second derivatives of curve gamma(t).
    See Equation (57) and (60) in paper. Note that the second derivative of leaky ReLU is 0.

    Args:
        M_list: List of Lie algebras (random square matrices).
        W_list: List of weight matrices.
        X: Data matrix.

    Returns:
        gamma_1_list: First derivative of curve gamma(t), d gamma / dt.
        gamma_2_list: Second derivative of curve gamma(t), d^2 gamma / dt^2.
    """

    gamma_1_list = []
    gamma_2_list = []
    for m in range(0, len(W_list)):
        gamma_1_list.append(torch.zeros_like(W_list[m]))
        gamma_2_list.append(torch.zeros_like(W_list[m]))

    h = X
    for m in range(0, len(W_list)-1):
        M = M_list[m]
        U = W_list[m+1]
        V = W_list[m]

        h_inv = torch.linalg.pinv(h)
        M_2 = torch.matmul(M, M)
        M_3 = torch.matmul(M_2, M)
        sigma_VX = sigma(torch.matmul(V, h))
        sigma_1_VX = sigma_1(torch.matmul(V, h))
        M_sigma_VX = torch.matmul(M, sigma_VX)


        gamma_1_list[m+1] += (-1) * torch.matmul(U, M)
        gamma_1_list[m] += torch.matmul(M_sigma_VX / sigma_1_VX, h_inv)

        gamma_2_list[m+1] += torch.matmul(U, M_2)
        gamma_2_list[m] += torch.matmul(torch.matmul(M_2, sigma_VX) / sigma_1_VX, h_inv)

        h = sigma(torch.matmul(V, h))

    return gamma_1_list, gamma_2_list

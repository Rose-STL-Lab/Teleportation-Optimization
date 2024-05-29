""" Compute correlation between sharpness/curvature and validation loss. """

import numpy as np
from matplotlib import pyplot as plt
import pickle
from scipy.stats import pearsonr
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from gradient_descent_mlp_utils import init_param_MLP, valid_MLP, loss_MLP_from_vec, loss_multi_layer
from models import MLP
from curvature_utils import W_list_to_vec, vec_to_W_list, compute_curvature, compute_gamma_12
from plot import plot_correlation

device = 'cpu'
dataset = 'CIFAR10' # 'MNIST', 'FashionMNIST', 'CIFAR10'
sigma_name = 'leakyrelu' # 'leakyrelu'
sigma = nn.LeakyReLU(0.1)
criterion = nn.CrossEntropyLoss()

num_run = 100
total_epoch = 40
check_epoch = 40 # compute curvature/sharpness using W_lists at this epoch.

# dataset and hyper-parameters
batch_size = 20
valid_size = 0.2
tele_batch_size = 200
if dataset == 'MNIST':
    lr = 1e-2
    t_start = 0.001
    t_end = 0.2
    t_interval = 0.01
    dim = [batch_size, 28*28, 16, 10, 10] # [batch_size, 28*28, 512, 512, 10]
    teledim = [tele_batch_size, 28*28, 16, 10, 10] # [tele_batch_size, 28*28, 512, 512, 10]
    train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transforms.ToTensor())
    test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transforms.ToTensor())
elif dataset == 'FashionMNIST':
    lr = 1e-2
    t_start = 0.0001
    t_end = 0.02
    t_interval = 0.001
    dim = [batch_size, 28*28, 16, 10, 10]
    teledim = [tele_batch_size, 28*28, 16, 10, 10]
    train_data = datasets.FashionMNIST(root = 'data', train = True, download = True, transform = transforms.ToTensor())
    test_data = datasets.FashionMNIST(root = 'data', train = False, download = True, transform = transforms.ToTensor())
elif dataset == 'CIFAR10':
    lr = 2e-2
    t_start = 0.0001
    t_end = 0.02
    t_interval = 0.001
    dim = [batch_size, 32*32*3, 128, 32, 10]
    teledim = [tele_batch_size, 32*32*3, 128, 32, 10]
    train_data = datasets.CIFAR10(root = 'data', train = True, download = True, transform = transforms.ToTensor())
    test_data = datasets.CIFAR10(root = 'data', train = False, download = True, transform = transforms.ToTensor())
else:
    raise ValueError('dataset should be one of MNIST, fashion, and CIFAR10')
    
# data loaders
if dataset in ['MNIST', 'FashionMNIST']:
    train_subset, val_subset = torch.utils.data.random_split(
            train_data, [50000, 10000], generator=torch.Generator().manual_seed(1))
    train_sampler = SequentialSampler(train_subset)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size = batch_size,
                                               sampler = train_sampler, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                             num_workers = 0)
    teleport_loader = torch.utils.data.DataLoader(train_subset, batch_size = tele_batch_size,
                                               shuffle=True, num_workers = 0)
    teleport_loader_iterator = iter(teleport_loader)
else: #CIFAR10
    train_sampler = SequentialSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                               sampler = train_sampler, num_workers = 0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                             num_workers = 0)
    teleport_loader = torch.utils.data.DataLoader(train_data, batch_size = tele_batch_size,
                                               shuffle=True, num_workers = 0)
    teleport_loader_iterator = iter(teleport_loader)


def train_step_SGD(x_train, y_train, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model.forward(x_train)
    loss = criterion(output.T, y_train)
    loss.backward()
    optimizer.step()
    return loss

def run_SGD_rand(seed):
    # run SGD with initial weights determined by seed
    W_list = init_param_MLP(dim, seed)
    loss_arr_SGD = []
    dL_dt_arr_SGD = []
    valid_loss_SGD = []
    valid_correct_SGD = []

    model = MLP(init_W_list=W_list, activation=sigma)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(total_epoch):
        epoch_loss = 0.0
        for data, label in train_loader:
            batch_size = data.shape[0]
            data = torch.t(data.view(batch_size, -1))
            loss = train_step_SGD(data, label, model, criterion, optimizer)
            epoch_loss += loss.item() * data.size(1)
        loss_arr_SGD.append(epoch_loss / len(train_loader.sampler))

        if epoch == 39:
            W_list_40 = model.get_W_list()

        if epoch in [39]:
            valid_loss, valid_correct = valid_MLP(model, criterion, test_loader)
            valid_loss_SGD.append(valid_loss)
            valid_correct_SGD.append(valid_correct)
        else:
            valid_loss_SGD.append(0)
            valid_correct_SGD.append(0)

        if epoch % 10 == 9:
            print(epoch, loss_arr_SGD[-1], valid_loss_SGD[-1], valid_correct_SGD[-1])

    results = (loss_arr_SGD, valid_loss_SGD, dL_dt_arr_SGD, valid_correct_SGD, W_list_40)
    return results


# run SGD with random seed 100 times and save
for i in range(num_run):
    print(i)
    results = run_SGD_rand((i+1)*54321)
    print(i, results[1][-1])
    with open('logs/correlation/{}/{}_final_W_lists/W_lists_{}_{}.pkl'.format(dataset, dataset, sigma_name, i), 'wb') as f:
        pickle.dump(results, f)

# compute sharpness and curvature and save
perturb_mean_list = []
curvature_mean_list = []
valid_loss_list = []
train_loss_list = []

for i in range(num_run):
    if i % 10 == 0:
        print(i)
    with open('logs/correlation/{}/{}_final_W_lists/W_lists_{}_{}.pkl'.format(dataset, dataset, sigma_name, i), 'rb') as f:
        train_loss, valid_loss, _, _, W_list  = pickle.load(f)
    
    train_loss_list.append(train_loss[check_epoch-1])
    valid_loss_list.append(valid_loss[check_epoch-1])
    W_vec_all = W_list_to_vec(W_list)    
    
    curvature_list = []
    perturb_list = []
    for curve_idx in range(200):
        # load data batch
        try:
            tele_data, tele_target = next(teleport_loader_iterator)
        except StopIteration:
            teleport_loader_iterator = iter(teleport_loader)
            tele_data, tele_target = next(teleport_loader_iterator)

        batch_size = tele_data.shape[0]
        tele_data = torch.t(tele_data.view(batch_size, -1))
        X = tele_data
        Y = tele_target

        # curvature (Equation 5)
        M_list = []
        torch.manual_seed(12345 * curve_idx)

        for m in range(0, len(W_list)-1):
            M = torch.rand(dim[m+2], dim[m+2])
            M = M / torch.norm(M, p='fro', dim=None)
            M_list.append(M)
        
        gamma_1_list, gamma_2_list = compute_gamma_12(M_list, W_list, X)
        curvature = compute_curvature(gamma_1_list, gamma_2_list).item()
        curvature_list.append(curvature)
        
        # sharpness (Equation 4)
        W_list_perturb = []
        for t in np.arange(t_start, t_end, t_interval):
            random_dir = torch.rand(W_vec_all.size()[0])
            random_dir = random_dir / torch.norm(random_dir) * t
            W_vec_all_perturb = W_vec_all + random_dir
            loss_perturb = loss_MLP_from_vec(W_vec_all_perturb, X, Y, dim, sigma)
            perturb_list.append(loss_perturb)
            
    curvature_mean_list.append(np.average(curvature_list))
    perturb_mean_list.append(np.average(perturb_list) / tele_batch_size) # correct

curvature_mean_list = np.array(curvature_mean_list)
valid_loss_list = np.array(valid_loss_list)
train_loss_list = np.array(train_loss_list)

results = (curvature_mean_list, perturb_mean_list, valid_loss_list, train_loss_list)
with open('logs/correlation/{}/{}_final_W_lists/curvatures_all_{}.pkl'.format(dataset, dataset, sigma_name), 'wb') as f:
    pickle.dump(results, f)

plot_correlation(dataset, sigma_name)
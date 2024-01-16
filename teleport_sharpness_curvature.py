""" Teleportation to change sharpness or curvature. """

import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from gradient_descent_mlp_utils import init_param_MLP, train_step, valid_MLP, teleport_curvature, teleport_sharpness
from models import MLP
from plot import plot_sharpness_curvature

device = 'cpu' #'cuda'
dataset = 'CIFAR10' # 'MNIST', 'FashionMNIST', 'CIFAR10'
objective_list = ['sharpness', 'curvature']

sigma = nn.LeakyReLU(0.1)
batch_size = 20
valid_size = 0.2
tele_batch_size = 2000

# dataset and hyper-parameters
if dataset == 'MNIST':
    lr = 1e-2
    t_start = 0.001
    t_end = 0.2
    t_interval = 0.01
    dim = [batch_size, 28*28, 16, 10, 10]
    teledim = [tele_batch_size, 28*28, 16, 10, 10]
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
    dim = [batch_size, 32*32*3, 32, 10, 10] # 32*32*3, 128, 32, 10]
    teledim = [tele_batch_size, 32*32*3, 32, 10, 10]
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


##############################################################
# run SGD without teleportation once
W_list = init_param_MLP(dim)
loss_arr_SGD = []
dL_dt_arr_SGD = []
valid_loss_SGD = []
valid_correct_SGD = []

model = MLP(init_W_list=W_list, activation=sigma)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(100):
    epoch_loss = 0.0
    for data, label in train_loader:
        batch_size = data.shape[0]
        data = torch.t(data.view(batch_size, -1))
        loss = train_step(data, label, model, criterion, optimizer)
        epoch_loss += loss.item() * data.size(1)
        
    loss_arr_SGD.append(epoch_loss / len(train_loader.sampler))
    valid_loss, valid_correct = valid_MLP(model, criterion, test_loader) #valid_loader)
    valid_loss_SGD.append(valid_loss)
    valid_correct_SGD.append(valid_correct)

    if epoch % 10 == 0:
        print(epoch, loss_arr_SGD[-1], valid_loss_SGD[-1], valid_correct_SGD[-1])

    W_list = model.get_W_list()
    with open('logs/generalization/{}/{}_SGD/{}_SGD_epoch_{}.pkl'.format(dataset, dataset, dataset, epoch), 'wb') as f:
        pickle.dump(W_list, f)

results = (loss_arr_SGD, valid_loss_SGD, dL_dt_arr_SGD, valid_correct_SGD, 0)
with open('logs/generalization/{}/{}_SGD_lr_{:e}.pkl'.format(dataset, dataset, lr), 'wb') as f:
    pickle.dump(results, f)


with open('logs/generalization/{}/{}_SGD_lr_{:e}.pkl'.format(dataset, dataset, lr), 'rb') as f:
    loss_arr_SGD, valid_loss_SGD, dL_dt_arr_SGD, _, _ = pickle.load(f)


##############################################################
# training with SGD + teleport sharpness/curvature

if dataset == 'CIFAR10':
    lr_teleport_sharpness = {True: 5e-2, False: 5e-2}
    lr_teleport_curvature = {True: 5e-2, False: 2e-1}
elif dataset == 'FashionMNIST':
    lr_teleport_sharpness = {True: 3e-1, False: 1e-1}
    lr_teleport_curvature = {True: 3e-3, False: 5e-3}
elif dataset == 'MNIST':
    lr_teleport_sharpness = {True: 5e-2, False: 5e-2}
    lr_teleport_curvature = {True: 5e-2, False: 2e-1}


start_epoch = 15 # use saved weights from the SGD run
end_epoch = 40

for objective in objective_list: 
    for reverse in [False, True]: # teleport to increase or decrease sharpness/curvature
        for run_num in range(5):
            loss_arr_teleport_rand_curvature = []
            dL_dt_arr_teleport_rand_curvature = []
            valid_loss_teleport_rand_curvature = []
            valid_correct_teleport_rand_curvature = []

            if start_epoch == 0:
                W_list = init_param_MLP(dim)
            else:
                with open('logs/generalization/{}/{}_SGD/{}_SGD_epoch_{}.pkl'.format(dataset, dataset, dataset, start_epoch), 'rb') as f:
                    W_list = pickle.load(f)
                loss_arr_teleport_rand_curvature = loss_arr_SGD[:start_epoch+1]
                valid_loss_teleport_rand_curvature = valid_loss_SGD[:start_epoch+1]

            model = MLP(init_W_list=W_list, activation=sigma)
            model.to(device)
            criterion = nn.CrossEntropyLoss() #nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)


            teleport_count = 0
            for epoch in range(start_epoch, end_epoch):
                epoch_loss = 0.0
                for data, label in train_loader:
                    batch_size = data.shape[0]
                    data = torch.t(data.view(batch_size, -1))
                    if (epoch == 20 and teleport_count < 10):# 5 for mnist, fashion
                        teleport_count += 1
                        W_list = model.get_W_list()

                        # load data batch
                        try:
                            tele_data, tele_target = next(teleport_loader_iterator)
                        except StopIteration:
                            teleport_loader_iterator = iter(teleport_loader)
                            tele_data, tele_target = next(teleport_loader_iterator)

                        # teleport
                        batch_size = tele_data.shape[0]
                        tele_data = torch.t(tele_data.view(batch_size, -1))
                        if objective == 'sharpness':
                            if (reverse == True and teleport_count < 11) or (reverse == False and teleport_count < 3): # teleport once if increasing sharpness
                                W_list = teleport_sharpness(W_list, tele_data, tele_target, lr_teleport_sharpness[reverse], teledim, sigma, \
                                                            telestep=10, reverse=reverse, t_start=t_start, t_end=t_end, t_interval=t_interval)
                        elif objective == 'curvature':
                            if (reverse == True and teleport_count < 6) or (reverse == False and teleport_count < 11): # teleport once if increasing sharpness
                                W_list = teleport_curvature(W_list, tele_data, tele_target, lr_teleport_curvature[reverse], teledim, sigma, telestep=1, reverse=reverse)
                        else:
                            raise ValueError("Teleportation objective should be either sharpness or curvature")

                        # update W_list in model
                        model = MLP(init_W_list=W_list, activation=sigma)
                        model.to(device)
                        criterion = nn.CrossEntropyLoss() #nn.MSELoss()
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                        

                    loss = train_step(data, label, model, criterion, optimizer)
                    epoch_loss += loss.item() * data.size(1)
                    
                loss_arr_teleport_rand_curvature.append(epoch_loss / len(train_loader.sampler))
                valid_loss, valid_correct = valid_MLP(model, criterion, test_loader) #valid_loader)
                valid_loss_teleport_rand_curvature.append(valid_loss)
                valid_correct_teleport_rand_curvature.append(valid_correct)

                if epoch % 1 == 0:
                    print(epoch, loss_arr_teleport_rand_curvature[-1], valid_loss_teleport_rand_curvature[-1], valid_correct_teleport_rand_curvature[-1])


            results = (loss_arr_SGD, loss_arr_teleport_rand_curvature, valid_loss_SGD, valid_loss_teleport_rand_curvature)
            with open('logs/generalization/{}/teleport_{}/teleport_{}_{}_{}.plk'.format(dataset, objective, objective, reverse,run_num), 'wb') as f:
                pickle.dump(results, f)

plot_sharpness_curvature(dataset, objective_list)

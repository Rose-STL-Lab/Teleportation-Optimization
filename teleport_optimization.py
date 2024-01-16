""" Evaluate various optimizers augmented with teleportation. """

import numpy as np
import time
from matplotlib import pyplot as plt
import pickle
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

from gradient_descent_mlp_utils import init_param_MLP, train_step, valid_MLP, teleport
from models import MLP
from plot import plot_optimization


device = 'cpu' #'cuda'
run_new = True # False if using cached results
dataset = 'MNIST' # 'MNIST', 'FashionMNIST', 'CIFAR10'
opt_method_list = ['Adagrad', 'momentum', 'RMSprop', 'Adam']

criterion = nn.CrossEntropyLoss()
sigma = nn.LeakyReLU(0.1)
batch_size = 20
valid_size = 0.2
tele_batch_size = 200

# dataset and hyper-parameters
if dataset == 'MNIST':
    lr = 1e-2
    dim = [batch_size, 28*28, 16, 10, 10]
    teledim = [tele_batch_size, 28*28, 16, 10, 10]
    train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transforms.ToTensor())
    test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transforms.ToTensor())
elif dataset == 'FashionMNIST':
    lr = 1e-2
    dim = [batch_size, 28*28, 16, 10, 10]
    teledim = [tele_batch_size, 28*28, 16, 10, 10]
    train_data = datasets.FashionMNIST(root = 'data', train = True, download = True, transform = transforms.ToTensor())
    test_data = datasets.FashionMNIST(root = 'data', train = False, download = True, transform = transforms.ToTensor())
elif dataset == 'CIFAR10':
    lr = 2e-2
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


def get_optimizer(model, opt_method, lr, dataset):
    if opt_method == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_method == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=lr)
    elif opt_method == 'momentum':
        return torch.optim.SGD(model.parameters(), lr=lr/1e1, momentum=0.9)
    elif opt_method == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr/1e2)
    elif opt_method == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr/1e2)
    else:
        raise ValueError('opt_method should be one of SGD, AdaGrad, momentum, RMSProp, and Adam')


start_epoch = 15
end_epoch = 40

if run_new == True:
    for opt_method in opt_method_list:   
        if opt_method == 'SGD':
            tele_epochs = [2]
            tele_lr = 1e-4
            tele_step = 10
        elif opt_method == 'Adagrad':
            tele_epochs = [2]
            tele_lr = 1e-4
            tele_step = 10
        elif opt_method == 'momentum':
            tele_epochs = [0]
            tele_lr = 5e-2
            tele_step = 10
        elif opt_method == 'RMSprop':
            tele_epochs = [0]
            tele_lr = 5e-2
            tele_step = 10
        elif opt_method == 'Adam':
            tele_epochs = [0]
            tele_lr = 5e-2
            tele_step = 10
        else:
            raise ValueError('opt_method should be one of SGD, AdaGrad, momentum, RMSProp, and Adam')

        for run_num in range(5):
            print(opt_method, 'run', run_num)
            
            ##############################################################
            # training with opt_method without teleportation (e.g. AdaGrad)
            W_list = init_param_MLP(dim, seed=(run_num+1)*54321)
            loss_arr_SGD = []
            dL_dt_arr_SGD = []
            valid_loss_SGD = []
            valid_correct_SGD = []
            time_SGD = []

            model = MLP(init_W_list=W_list, activation=sigma)
            model.to(device)
            optimizer = get_optimizer(model, opt_method, lr, dataset)

            t0 = time.time()
            for epoch in range(40):
                epoch_loss = 0.0
                for data, label in train_loader:
                    batch_size = data.shape[0]
                    data = torch.t(data.view(batch_size, -1)) # [20, 1, 28, 28] -> [784, 20]
                    loss = train_step(data, label, model, criterion, optimizer)
                    epoch_loss += loss.item() * data.size(1)
                    
                loss_arr_SGD.append(epoch_loss / len(train_loader.sampler))
                valid_loss, valid_correct = valid_MLP(model, criterion, test_loader)
                valid_loss_SGD.append(valid_loss)
                valid_correct_SGD.append(valid_correct)

                # print(epoch, loss_arr_SGD[-1], valid_loss_SGD[-1], valid_correct_SGD[-1])

                t1 = time.time()
                time_SGD.append(t1 - t0)

            results = (loss_arr_SGD, valid_loss_SGD, dL_dt_arr_SGD, valid_correct_SGD, time_SGD, 0)
            with open('logs/optimization/{}/{}_{}_lr_{}_{}.pkl'.format(dataset, dataset, opt_method, lr, run_num), 'wb') as f:
                pickle.dump(results, f)
            

            ##############################################################
            # training with opt_method + teleport
            W_list = init_param_MLP(dim, seed=(run_num+1)*54321)
            loss_arr_teleport = []
            dL_dt_arr_teleport = []
            valid_loss_teleport = []
            valid_correct_teleport = []
            time_teleport = []

            model = MLP(init_W_list=W_list, activation=sigma)
            model.to(device)
            optimizer = get_optimizer(model, opt_method, lr, dataset)

            teleport_count = 0
            t0 = time.time()

            for epoch in range(40):
                epoch_loss = 0.0
                for data, label in train_loader:
                    batch_size = data.shape[0]
                    data = torch.t(data.view(batch_size, -1)) # [20, 1, 28, 28] -> [784, 20]
                    if (epoch in tele_epochs and teleport_count < 8): 
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
                        tele_data = torch.t(tele_data.view(batch_size, -1))  # [tele_batch_size, 1, 28, 28] -> [784, tele_batch_size]

                        W_list = teleport(W_list, tele_data, tele_target, tele_lr, dim, sigma, telestep=tele_step, random_teleport=False, reverse=False)

                        # update W_list in model
                        model = MLP(init_W_list=W_list, activation=sigma)
                        model.to(device)
                        optimizer = get_optimizer(model, opt_method, lr, dataset)
                        

                    loss = train_step(data, label, model, criterion, optimizer)
                    epoch_loss += loss.item() * data.size(1)
                    
                loss_arr_teleport.append(epoch_loss / len(train_loader.sampler))
                valid_loss, valid_correct = valid_MLP(model, criterion, test_loader)
                valid_loss_teleport.append(valid_loss)
                valid_correct_teleport.append(valid_correct)

                # print(epoch, loss_arr_teleport[-1], valid_loss_teleport[-1], valid_correct_teleport[-1])

                t1 = time.time()
                time_teleport.append(t1 - t0)

            results = (loss_arr_teleport, valid_loss_teleport, dL_dt_arr_teleport, valid_correct_teleport, time_teleport, 0)
            with open('logs/optimization/{}/{}_{}_lr_{}_teleport_{}.pkl'.format(dataset, dataset, opt_method, lr, run_num), 'wb') as f:
                pickle.dump(results, f)

plot_optimization(opt_method_list, dataset, lr)

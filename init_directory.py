""" Scripts for setting up directories. """

import os

if not os.path.exists('data'):
    os.mkdir('data')

if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('figures/correlation'):
    os.mkdir('figures/correlation')
if not os.path.exists('figures/generalization'):
    os.mkdir('figures/generalization')
if not os.path.exists('figures/optimization'):
    os.mkdir('figures/optimization')

if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists('logs/correlation'):
    os.mkdir('logs/correlation')
if not os.path.exists('logs/generalization'):
    os.mkdir('logs/generalization')
if not os.path.exists('logs/optimization'):
    os.mkdir('logs/optimization')

dataset_list = ['MNIST', 'FashionMNIST', 'CIFAR10']
for dataset in dataset_list:
    if not os.path.exists('logs/correlation/{}'.format(dataset)):
        os.mkdir('logs/correlation/{}'.format(dataset))
    if not os.path.exists('logs/generalization/{}'.format(dataset)):
        os.mkdir('logs/generalization/{}'.format(dataset))
    if not os.path.exists('logs/optimization/{}'.format(dataset)):
        os.mkdir('logs/optimization/{}'.format(dataset))

    if not os.path.exists('logs/generalization/{}/{}_SGD'.format(dataset, dataset)):
        os.mkdir('logs/generalization/{}/{}_SGD'.format(dataset, dataset))
    if not os.path.exists('logs/generalization/{}/teleport_curvature'.format(dataset)):
        os.mkdir('logs/generalization/{}/teleport_curvature'.format(dataset))
    if not os.path.exists('logs/generalization/{}/teleport_sharpness'.format(dataset)):
        os.mkdir('logs/generalization/{}/teleport_sharpness'.format(dataset))

    if not os.path.exists('logs/correlation/{}/{}_final_W_lists'.format(dataset, dataset)):
        os.mkdir('logs/correlation/{}/{}_final_W_lists'.format(dataset, dataset))
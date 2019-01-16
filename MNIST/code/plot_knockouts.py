import pickle
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from matplotlib.colors import ListedColormap

import scipy.stats as spst

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from train_test_net_stat import Net

def excel_per_class():
    for c in range(10):
        c_acc = np.zeros((20, nets))  # (units, net_nr)
        for net_nr in range(nets):
            path = '../results/acc_class_net%d.xlsx' % net_nr
            df = pd.read_excel(path)
            c_acc[:, net_nr] = df.iloc[:, c]
        df_per_class = pd.DataFrame(c_acc)
        df_per_class.to_excel('../results/acc_per_class_%d.xlsx' % c, index=False)

def calc_drop():
    acc = pd.read_excel('../results/accuracies_class.xlsx')
    for c in range(classes):
        df = pd.read_excel('../results/acc_per_class_%d.xlsx' % c)
        df_drop = acc.iloc[c, :] - df
        print('New df', df_drop.shape)
        df_drop.to_excel('../results/drop_acc_%d.xlsx' % c)


def plot_box():
    fig, ax = plt.subplots(classes, 1, figsize=(15, 12), sharex='all')
    print('ax', ax.shape)
    for c in range(classes):
        path = '../results/drop_acc_%d.xlsx' % c
        df_acc = pd.read_excel(path)
        print('df_acc', df_acc.shape)
        ax[c].boxplot(df_acc.transpose())
        ax[c].spines['top'].set_visible(False)
        ax[c].spines['right'].set_visible(False)
        ax[c].spines['bottom'].set_visible(False)
        # ax[c].spines['left'].set_visible(False)
        ax[c].set_yticks([0, 1])
        ax[c].tick_params('x', labelsize=16, which='both', bottom=False)
        ax[c].set_ylabel(c, fontsize=16, rotation=0, labelpad=20)
    plt.xlabel('Net', fontsize=16, labelpad=20)
    plt.show()

def plot_stdv():
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    for c in range(classes):
        df = pd.read_excel('../results/drop_acc_%d.xlsx' %c)
        x = [c] * nets
        y = df.std(axis=0)
        print('y', y)
        ax.scatter(x=x, y=y)
    plt.xlabel('Class')
    plt.ylabel('Standard deviation')
    plt.xticks([i for i in range(10)])

    plt.show()

if __name__ == "__main__":
    # params
    nets = 20
    classes = 10
    calc_excel = False
    drop = False
    box = False
    stdv = True

    if calc_excel:
        excel_per_class()
    if drop:
        calc_drop()
    if box:
        plot_box()
    if stdv:
        plot_stdv()
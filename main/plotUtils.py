import os
import sys
import argparse
os.environ['KMP_DUPLICATE_LAB_OK'] = 'TRUE'
import re
import pickle as pkl
import torch.nn as nn
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

criteo_feat_names = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'C1',
                     'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
                     'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26']

avazu_feat_names = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain',
                    'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14',
                    'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']


def dir_traversal(rootDir, file_path_list):
    for root, dirs, files in os.walk(rootDir):
        for file in files:
            if file != '.DS_Store':
                file_path = os.path.join(root, file)
                print(file_path)
                file_path_list.append(file_path)
        for dir in dirs:
            dir_traversal(dir, file_path_list)


def load_operation_data(file_path, feat_names):
    with open(file_path, 'rb') as f:
        data = pkl.load(f)
    feature_num = len(feat_names)
    operation_data = np.zeros([feature_num, feature_num], dtype=int)
    index = 0
    for i in range(0, feature_num):
        for j in range(i + 1, feature_num):
            operation_data[i][j] = data[index]
            index += 1
    for i in range(0, feature_num):
        for j in range(0, i + 1):
            if (i == j):
                operation_data[i][j] = -1
            else:
                operation_data[i][j] = operation_data[j][i]
    return operation_data


def load_alpha_beta_data(file_path, feat_names):
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    activate = nn.Tanh()
    np.set_printoptions(precision=1)
    alpha = checkpoint['alpha']
    beta = checkpoint['beta']
    beta = activate(beta)
    alpha = activate(alpha)
    alpha = alpha.data.numpy()
    alpha = np.maximum(alpha, -alpha)
    alpha = np.reshape(alpha, (1, len(feat_names)))
    beta = beta.data.numpy()
    beta = np.maximum(beta, -beta)
    feature_num = len(feat_names)
    beta_matrix = np.zeros([feature_num, feature_num], dtype=float)
    index = 0
    for i in range(0, feature_num):
        for j in range(i + 1, feature_num):
            beta_matrix[i][j] = beta[index]
            index += 1
    for i in range(0, feature_num):
        for j in range(0, i + 1):
            if i == j:
                beta_matrix[i][j] = 0
            else:
                beta_matrix[i][j] = beta_matrix[j][i]
    alpha_data = pd.DataFrame(alpha)
    beta_data = pd.DataFrame(beta_matrix)
    return alpha_data, beta_data


def cal_beta_data(operation_data, beta_data, feature_num):
    beta_data = beta_data.values
    m = np.zeros_like(operation_data, dtype=float)
    for i in range(0, feature_num):
        for j in range(i + 1, feature_num):
            if beta_data[i][j] == 0.:
                m[i][j] = -1
            else:
                m[i][j] = operation_data[i][j] + beta_data[i][j]
    for i in range(0, feature_num):
        for j in range(0, i + 1):
            if (i == j):
                m[i][j] = -1
            else:
                m[i][j] = m[j][i]
    return pd.DataFrame(m)


def heat_map_plot(alpha_data, beat_data, feat_names, index, dataset, experiment_time):
    fig = plt.figure(figsize=(25, 30))
    grid = plt.GridSpec(45, 39, wspace=0.4, hspace=0.3, figure=fig)
    ax1 = plt.subplot(grid[:40, :])
    ax2 = plt.subplot(grid[40:44, :])
    # Define the gradient color bar
    beta_cdict = ['#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF',
                  '#FFFFFF',
                  '#FFE6E3', '#FCD6D2', '#FACAC5', '#FABBB4', '#FCA89F', '#FC968A', '#FC8174', '#FC6453', '#FC4732',
                  '#FC1900',
                  '#DEFFE0', '#CFFFD2', '#C2FFC6', '#B5FFBA', '#A6FFAC', '#96FF9D', '#87FF8F', '#6EFF78', '#45FF51',
                  '#00FF00',
                  '#FFF6D9', '#FFF4CF', '#FFF1C2', '#FFEDB0', '#FFE89C', '#FFE387', '#FFDD6E', '#FFD752', '#FFCC26',
                  '#FFC400',
                  '#E3E4FF', '#D1D3FF', '#C2C5FF', '#B0B4FF', '#A8ACFF', '#969BFF', '#8288FF', '#6971FF', '#3F49FC',
                  '#0000FF']
    # divide the beta data into corresponding parts
    beta_colormap = colors.ListedColormap(beta_cdict, 'indexed')
    sns.heatmap(beat_data, vmin=-1, vmax=3.9, xticklabels=feat_names, yticklabels=feat_names, square=True,
                cmap=beta_colormap,
                annot=np.array(beat_data, dtype='int'), annot_kws={'size': 16, 'weight': 'bold', 'color': 'black'}, cbar=False, ax=ax1)

    # Define the gradient color bar
    alpha_cdict = ['#FFFFFF', '#E0F8FF', '#D4F5FF', '#C4F1FF', '#B4F1FF', '#A1F1FF', '#8AEFFF', '#63EAFF', '#42E6FF',
                   '#00DDFF']
    # divide the aplha data into corresponding parts
    alpha_colormap = colors.ListedColormap(alpha_cdict, 'indexed')
    sns.heatmap(alpha_data, vmin=0, vmax=1, xticklabels=feat_names, yticklabels=False, square=True, cbar=False,
                cmap=alpha_colormap, ax=ax2)
    plt.tight_layout()

    dir_path = os.path.join('../img/', dataset, experiment_time)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    img_path = os.path.join(dir_path, str(int(index)) + '.jpg')
    plt.savefig(img_path, dpi=400)
    plt.close()


def fis_plot(dataset, feat_name, experiment_time):
    operation_type_path_list = []
    dir_traversal(os.path.join('../param', dataset, experiment_time, "evolution", 'operation_type'), operation_type_path_list)
    alpha_beta_path_list = []
    dir_traversal(os.path.join('../param', dataset, experiment_time, 'evolution', 'alpha_beta'), alpha_beta_path_list)
    for index in tqdm(range(len(alpha_beta_path_list))):
        operation_type_file_path = operation_type_path_list[index]
        alpha_beta_file_path = alpha_beta_path_list[index]
        index = re.findall(r'[0-9]+', alpha_beta_file_path)

        operation_type_data = load_operation_data(operation_type_file_path, feat_name)
        alpha_data, beta_data = load_alpha_beta_data(alpha_beta_file_path, feat_name)
        beta_data = cal_beta_data(operation_type_data, beta_data, len(feat_name))

        heat_map_plot(alpha_data, beta_data, feat_name, index[-1], dataset, experiment_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot CELS example')
    parser.add_argument('--dataset_strategy', type=str, default='criteo_1,1', help='use model',
                        choices=['criteo_1,1', 'criteo_1+1', 'criteo_1+n', 'criteo_n,1', 'criteo_n+1'])
    parser.add_argument('--datetime', type=str, help='datetime')
    args = parser.parse_args()
    dataset = args.dataset_strategy
    if dataset == 'criteo_1,1' or dataset == 'criteo_1+1' or dataset == 'criteo_1+n' or dataset == 'criteo_n,1' or dataset == 'criteo_n+1':
        feat_name = criteo_feat_names
    else:
        feat_name = avazu_feat_names

    experiment_time = args.datetime
    fis_plot(dataset, feat_name, experiment_time)

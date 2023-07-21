import logging
import time
import torch
import pickle as pkl
import pandas as pd
from data.avazuPreprocess import AvazuPreprocess
from data.featureDefiniton import SparseFeat
from utils.function_utils import build_input_features, log
from trainer.cels_s1_trainer import evolution_search
from trainer.cels_s2_trainer import model_functioning
from config.configs import General_Config
import os

feat_names = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain',
              'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15',
              'C16', 'C17', 'C18', 'C19', 'C20', 'C21']


def train(params):
    strategy = params.strategy
    param_save_dir = os.path.join('../param/avazu' + '_' + strategy, time.strftime("%Y-%m-%d-%H-%M", time.localtime()))
    if not os.path.exists(param_save_dir):
        param_save_dir_fis_type = os.path.join(param_save_dir, "evolution", "operation_type")
        param_save_dir_alphabeta = os.path.join(param_save_dir, "evolution", "alpha_beta")
        os.makedirs(param_save_dir_fis_type)
        os.makedirs(param_save_dir_alphabeta)

    log(dataset=params.dataset, model=params.model, strategy=params.strategy)
    logging.info('-' * 50)
    logging.info(str(time.asctime(time.localtime(time.time()))))
    avazuData = AvazuPreprocess()
    train_data_file_path = pkl.load(open(avazuData.train_path, 'rb'))
    test_data_file_path = pkl.load(open(avazuData.test_path, 'rb'))
    feat_size = pkl.load(open(avazuData.feature_size_file_path, 'rb'))
    fixlen_feature_columns = [SparseFeat(feat_name, feat_size[feat_name],
                                         General_Config['general']['avazu_embedding_size'])for feat_name in feat_names]
    feature_index = build_input_features(fixlen_feature_columns)
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        logging.info('cuda ready...')
        device = 'cuda:' + str(params.gpu)
    logging.info(params)
    logging.info(General_Config)
    if General_Config['general']['data'] == -1:
        data_train = pd.read_csv(train_data_file_path)
        data_test = pd.read_csv(test_data_file_path)
    else:
        data_train = pd.read_csv(train_data_file_path, nrows=General_Config['general']['data'])
        data_test = pd.read_csv(test_data_file_path, nrows=General_Config['general']['data'])

    mutation = bool(params.mutation)
    evolution_search(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                     data_train=data_train, param_save_dir=param_save_dir,
                     mutation=mutation, label_name='click', strategy=strategy,
                     embedding_size=General_Config['general']['avazu_embedding_size'],
                     device=device)

    model_functioning(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                      data_train=data_train, data_test=data_test,
                      label_name='click',
                      embedding_size=General_Config['general']['avazu_embedding_size'],
                      param_save_dir=param_save_dir,
                      device=device)
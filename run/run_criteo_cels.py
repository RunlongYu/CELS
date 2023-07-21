import time
import logging
import torch
import pickle as pkl
import pandas as pd
from data.criteoPreprocess import CriteoProcessor
from data.featureDefiniton import SparseFeat, DenseBucketFeat
from utils.function_utils import build_input_features, log
from trainer.cels_s1_trainer import evolution_search
from trainer.cels_s2_trainer import model_functioning
from config.configs import General_Config
import os


def train(params):
    strategy = params.strategy
    param_save_dir = os.path.join('../param/criteo' + '_' + strategy, time.strftime("%Y-%m-%d-%H-%M", time.localtime()))
    if not os.path.exists(param_save_dir):
        param_save_dir_fis_type = os.path.join(param_save_dir, "evolution", "operation_type")
        param_save_dir_alphabeta = os.path.join(param_save_dir, "evolution", "alpha_beta")
        os.makedirs(param_save_dir_fis_type)
        os.makedirs(param_save_dir_alphabeta)

    log(dataset=params.dataset, model=params.model, strategy=params.strategy)
    logging.info('-' * 50)
    logging.info(str(time.asctime(time.localtime(time.time()))))
    dataProcessor = CriteoProcessor()
    train_data_file_path = pkl.load(open(dataProcessor.processed_full_train_data_file_path, 'rb'))
    test_data_file_path = pkl.load(open(dataProcessor.processed_full_test_data_file_path, 'rb'))
    feat_size = pkl.load(open(dataProcessor.feature_size_file_path, 'rb'))
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    fixlen_feature_columns = [SparseFeat(feat, feat_size[feat], General_Config['general']['criteo_embedding_size'])
                              for feat in sparse_features] + \
                             [DenseBucketFeat(feat, feat_size[feat], General_Config['general']['criteo_embedding_size'])
                              for feat in dense_features]
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:' + str(params.gpu)
    feature_index = build_input_features(fixlen_feature_columns)
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
                     mutation=mutation, label_name='label', strategy=strategy,
                     embedding_size=General_Config['general']['criteo_embedding_size'],
                     device=device)

    model_functioning(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                      data_train=data_train, data_test=data_test,
                      label_name='label',
                      embedding_size=General_Config['general']['criteo_embedding_size'],
                      param_save_dir=param_save_dir,
                      device=device)

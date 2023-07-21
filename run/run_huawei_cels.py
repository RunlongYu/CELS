import time
import logging
import torch
import pickle as pkl
import pandas as pd
from data.huaweiPreProcess import HuaweiPreprocess
from data.featureDefiniton import SparseFeat
from utils.function_utils import build_input_features, log
from trainer.cels_s1_trainer import evolution_search
from trainer.cels_s2_trainer import model_functioning
from config.configs import General_Config
import os


feat_names = ['uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id',
              'dev_id', 'inter_type_cd', 'slot_id', 'spread_app_id', 'tags',
              'app_first_class', 'app_second_class', 'age', 'city', 'city_rank',
              'device_name', 'device_size', 'career', 'gender', 'net_type',
              'residence', 'his_app_size', 'his_on_shelf_time', 'app_score',
              'emui_dev', 'list_time', 'device_price', 'up_life_duration',
              'up_membership_grade', 'membership_life_duration', 'consume_purchase',
              'communication_onlinerate', 'communication_avgonline_30d', 'indu_name',
              'pt_d']


def train(params):
    strategy = params.strategy
    param_save_dir = os.path.join('../param/huawei' + '_' + strategy, time.strftime("%Y-%m-%d-%H-%M", time.localtime()))
    if not os.path.exists(param_save_dir):
        param_save_dir_fis_type = os.path.join(param_save_dir, "evolution", "operation_type")
        param_save_dir_alphabeta = os.path.join(param_save_dir, "evolution", "alpha_beta")
        os.makedirs(param_save_dir_fis_type)
        os.makedirs(param_save_dir_alphabeta)

    log(dataset=params.dataset, model=params.model, strategy=params.strategy)
    logging.info('-' * 50)
    logging.info(str(time.asctime(time.localtime(time.time()))))
    huaweiData = HuaweiPreprocess()
    train_data_file_path = pkl.load(open(huaweiData.train_path, 'rb'))
    test_data_file_path = pkl.load(open(huaweiData.test_path, 'rb'))
    feat_size = pkl.load(open(huaweiData.feature_size_file_path, 'rb'))
    fixlen_feature_columns = [SparseFeat(feat_name, feat_size[feat_name], General_Config['general']['huawei_embedding_size'])
                              for feat_name in feat_names]
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
                     mutation=mutation, label_name='label', strategy=strategy,
                     embedding_size=General_Config['general']['huawei_embedding_size'],
                     device=device)

    model_functioning(feature_columns=fixlen_feature_columns, feature_index=feature_index,
                      data_train=data_train, data_test=data_test,
                      label_name='label',
                      embedding_size=General_Config['general']['huawei_embedding_size'],
                      param_save_dir=param_save_dir,
                      device=device)

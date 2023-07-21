import os
import time
import torch
import logging
import pickle as pkl
import numpy as np
from utils.function_utils import get_feature_names
from config.configs import CELS_Config, General_Config
from utils.function_utils import get_param_sum
from model.Functioning_Model import Functioning_Model
from sklearn.metrics import log_loss, roc_auc_score


def model_functioning(feature_columns, feature_index, data_train, data_test, param_save_dir, label_name,
                      embedding_size, device='cpu'):
    logging.info('\nModel Functioning period param:')
    logging.info(CELS_Config['ModelFunctioning'])
    feature_names = get_feature_names(feature_columns)

    selected_interaction_type = pkl.load(open(os.path.join(param_save_dir, 'interaction_type-embedding_size-' + str(
        embedding_size) + '.pkl'), 'rb'))
    logging.info(selected_interaction_type)

    checkpoint = torch.load(
        os.path.join(param_save_dir,
                     'alpha_beta-c' + str(CELS_Config['CELS']['c']) + '-mu' + str(
                         CELS_Config['CELS']['mu']) + '-embedding_size' + str(
                         embedding_size) + '.pth'))
    alpha = checkpoint['alpha']
    beta = checkpoint['beta']
    logging.info(alpha)
    logging.info(beta)

    mf_model = Functioning_Model(feature_columns=feature_columns, feature_index=feature_index,
                                 selected_interaction_type=selected_interaction_type,
                                 interaction_fc_output_dim=CELS_Config['ModelFunctioning']['interaction_fc_output_dim'],
                                 alpha=alpha, beta=beta,
                                 embedding_size=embedding_size,
                                 device=device)

    train_model_input = {name: data_train[name] for name in feature_names}
    test_model_input = {name: data_test[name] for name in feature_names}
    logging.info("Model Functioning period start")
    mf_model.to(device)
    mf_model.before_train()
    start_time = time.time()
    get_param_sum(model=mf_model)
    print("Model Functioning period")
    mf_model.fit(train_model_input, data_train[label_name].values, batch_size=General_Config['general']['batch_size'],
                 epochs=General_Config['general']['epochs'],
                 validation_split=General_Config['general']['validation_split'])
    predict_result = mf_model.predict(test_model_input, 256)
    logging.info("test LogLoss:{}".format(round(log_loss(data_test[label_name].values, predict_result), 4)))
    logging.info("test AUC:{}".format(round(roc_auc_score(data_test[label_name].values, predict_result), 4)))
    end_time = time.time()
    cost_time = int(end_time - start_time)
    logging.info("Model Functioning period end")
    logging.info('Model Functioning period cost:' + str(cost_time))

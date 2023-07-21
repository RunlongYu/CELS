import os
import time
import logging
import pickle as pkl
from utils.function_utils import get_feature_names
from config.configs import CELS_Config, General_Config
from utils.function_utils import get_param_sum
from model.CELS_Model import CELS_Model
import numpy as np
from utils.function_utils import random_selected_interaction_type


def evolution_search(feature_columns, feature_index, data_train, param_save_dir, label_name, embedding_size,
                     mutation=True, device='cpu', strategy='1+1'):
    logging.info('Cognitive EvoLutionary Search period param:')
    logging.info(CELS_Config['CELS'])

    feature_names = get_feature_names(feature_columns)

    # Random initialize interaction_type
    pair_feature_len = int(len(feature_names) * (len(feature_names)-1) / 2)
    selected_interaction_type = random_selected_interaction_type(pair_feature_len)

    train_model_input = {name: data_train[name] for name in feature_names}
    logging.info('Cognitive EvoLutionary Search period start')
    cels_model = CELS_Model(feature_columns=feature_columns, feature_index=feature_index,
                            param_save_dir=param_save_dir,
                            selected_interaction_type=selected_interaction_type,
                            mutation=mutation,
                            mutation_probability=CELS_Config['CELS']['mutation_probability'],
                            embedding_size=embedding_size,
                            device=device)
    cels_model.to(device)
    cels_model.before_train()
    start_time = time.time()
    get_param_sum(model=cels_model)

    if strategy == '1,1':
        cels_model.fit_1_1(train_model_input, data_train[label_name].values,
                           batch_size=General_Config['general']['batch_size'],
                           epochs=General_Config['general']['epochs'],
                           validation_split=CELS_Config['CELS']['validation_split'])
    elif strategy == '1+1':
        cels_model.fit_1_plus_1(train_model_input, data_train[label_name].values,
                                batch_size=General_Config['general']['batch_size'],
                                epochs=General_Config['general']['epochs'],
                                validation_split=CELS_Config['CELS']['validation_split'])
    
    elif strategy == 'n,1':
        cels_model.fit_n_1(train_model_input, data_train[label_name].values,
                           batch_size=General_Config['general']['batch_size'],
                           epochs=General_Config['general']['epochs'],
                           validation_split=CELS_Config['CELS']['validation_split'])
    elif strategy == 'n+1':
        cels_model.fit_n_plus_1(train_model_input, data_train[label_name].values,
                                batch_size=General_Config['general']['batch_size'],
                                epochs=General_Config['general']['epochs'],
                                validation_split=CELS_Config['CELS']['validation_split'])

    cels_model.after_train(param_save_dir=param_save_dir)
    end_time = time.time()
    cost_time = int(end_time - start_time)
    logging.info('Cognitive EvoLutionary Search period end')
    logging.info('Cognitive EvoLutionary Search period cost:' + str(cost_time))


import os
import time
import torch
import logging
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
from tqdm import tqdm
from optimizer.gRDA import gRDA
from .baseModel import BaseModel
from layer.interactionLayer import InteractionLayer
from layer.linearLayer import NormalizedWeightedLinearLayer
from utils.function_utils import generate_pair_index, slice_arrays
from sklearn.metrics import *
from config.configs import CELS_Config, General_Config
import pickle as pkl
from utils.function_utils import random_selected_interaction_type
from torch.nn.parameter import Parameter


class CELS_Model(BaseModel):
    def __init__(self, feature_columns, feature_index, selected_interaction_type,
                 param_save_dir, embedding_size=20, mutation=True,
                 mutation_probability=0.5,
                 activation='tanh', seed=1024, device='cpu'):
        super(CELS_Model, self).__init__()
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.embedding_size = embedding_size
        self.device = device
        if device == 'cpu':
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
        self.register_buffer('pair_indexes',
                             torch.tensor(generate_pair_index(len(self.feature_columns), 2)))
        self.interaction_pair_number = len(self.pair_indexes[0])

        self.param_save_dir = param_save_dir
        self.mutation = mutation
        self.mutation_probability = mutation_probability

        self.interaction_fc_output_dim = int(CELS_Config['CELS']['interaction_fc_output_dim'])
        self.mutation_threshold = float(CELS_Config['CELS']['mutation_threshold'])
        self.mutation_step_size = int(CELS_Config['CELS']['mutation_step_size'])
        self.adaptation_hyperparameter = float(CELS_Config['CELS']['adaptation_hyperparameter'])
        self.adaptation_step_size = int(CELS_Config['CELS']['adaptation_step_size'])
        self.population_size = int(CELS_Config['CELS']['population_size'])

        # \alpha
        self.linear = NormalizedWeightedLinearLayer(feature_columns=feature_columns, feature_index=feature_index,
                                                    alpha_activation=activation, use_alpha=True,
                                                    device=device)
        # \beta
        self.interaction_operation = InteractionLayer(input_dim=embedding_size, feature_columns=feature_columns,
                                                      feature_index=feature_index, use_beta=True,
                                                      interaction_fc_output_dim=self.interaction_fc_output_dim,
                                                      selected_interaction_type=selected_interaction_type,
                                                      mutation_threshold=self.mutation_threshold,
                                                      mutation_probability=self.mutation_probability,
                                                      device=device)

    def forward(self, x, mutation=False):
        linear_out = self.linear(x)
        interation_out = self.interaction_operation(x, mutation)
        out = linear_out + interation_out
        return torch.sigmoid(out)

    def before_train(self):
        self.metrics_names = ["loss"]
        all_parameters = self.parameters()
        structure_params = {self.linear.alpha, self.interaction_operation.beta}
        net_params = [i for i in all_parameters if i not in structure_params]
        self.structure_optim = self.get_structure_optim(structure_params)
        self.net_optim = self.get_net_optim(net_params)
        self.loss_func = F.binary_cross_entropy
        self.metrics = self.get_metrics(["binary_crossentropy", "auc"])

    def get_net_optim(self, net_params):
        optimizer = optim.Adam(net_params, lr=float(CELS_Config['CELS']['net_optim_lr']))
        return optimizer

    def get_structure_optim(self, structure_params):
        optimizer = gRDA(structure_params, lr=float(CELS_Config['CELS']['gRDA_optim_lr']),
                         c=CELS_Config['CELS']['c'], mu=CELS_Config['CELS']['mu'])
        return optimizer

    def get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_

    def new(self):
        pair_feature_len = len(self.interaction_operation.selected_interaction_type)
        random_type = random_selected_interaction_type(pair_feature_len)
        model_new = CELS_Model(feature_columns=self.feature_columns, feature_index=self.feature_index,
                               param_save_dir=self.param_save_dir,
                               selected_interaction_type=random_type,
                               mutation=self.mutation,
                               mutation_probability=self.mutation_probability,
                               embedding_size=self.embedding_size, device=self.device)
        return model_new.to(self.device)

    def replace(self, new_model):
        self.load_state_dict(new_model.state_dict())
        self.interaction_operation.selected_interaction_type = new_model.interaction_operation.selected_interaction_type

    def fit_1_1(self, x=None, y=None, batch_size=None, epochs=1, initial_epoch=0, validation_split=0.,
                shuffle=True):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        do_validation = False
        if validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        model = self.train()
        loss_func = self.loss_func
        net_optim = self.net_optim
        structure_optim = self.structure_optim
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        print("Cognitive EvoLutionary Search period")
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))

        for epoch in range(initial_epoch, epochs):
            epoch_start_time = time.time()
            epoch_logs = {}
            # total_loss_epoch = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader)) as t:
                    for index, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        if self.mutation and index % self.mutation_step_size == 0:
                            # muation to generate the offspring model
                            y_pred = model(x, mutation=True).squeeze()
                        else:
                            y_pred = model(x, mutation=False).squeeze()
                        net_optim.zero_grad()
                        structure_optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        loss.backward()
                        net_optim.step()
                        structure_optim.step()
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            train_result[name].append(metric_fun(
                                y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

                        if index % 100 == 0:
                            self.after_train(self.param_save_dir, name='round' + str(int(index / 100)))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            epoch_time = int(time.time() - epoch_start_time)
            logging.info('Epoch {0}/{1}'.format(epoch + 1, epochs))

            eval_str = "{0}s".format(epoch_time)

            for name in self.metrics:
                eval_str += " - " + name + \
                            ": {0: .4f}".format(epoch_logs[name])

            if do_validation:
                for name in self.metrics:
                    eval_str += " - " + "val_" + name + \
                                ": {0: .4f}".format(epoch_logs["val_" + name])
            logging.info(eval_str)

    def fit_1_plus_1(self, x=None, y=None, batch_size=None, epochs=1, initial_epoch=0, validation_split=0.,
                     shuffle=True):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        do_validation = False
        if validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        model = self.train()
        loss_func = self.loss_func
        net_optim = self.net_optim
        structure_optim = self.structure_optim
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        print("CELS period")
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))

        for epoch in range(initial_epoch, epochs):
            epoch_start_time = time.time()
            epoch_logs = {}
            # total_loss_epoch = 0
            train_result = {}
            parent_model = self.new()
            parent_loss = float('inf')
            child_replace_parent_count = 0
            try:
                with tqdm(enumerate(train_loader)) as t:
                    for index, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        if index % self.mutation_step_size == 0:
                            # choose the better model between the current model and parent model
                            current_y_pred = model(x, mutation=False).squeeze()
                            current_loss = loss_func(current_y_pred, y.squeeze(), reduction='sum')
                            if current_loss < parent_loss:
                                parent_model.replace(self)
                                parent_loss = current_loss
                                child_replace_parent_count += 1
                            else:
                                self.replace(parent_model)

                            # adapt the mutation_probability accroding to the 1/5 successful rule
                            if index % (self.mutation_step_size * self.adaptation_step_size) == 0 and index != 0:
                                # self.replace_count(self.param_save_dir, child_replace_parent_count)
                                if child_replace_parent_count < 2:
                                    self.interaction_operation.mutation_probability *= self.adaptation_hyperparameter
                                elif child_replace_parent_count > 2:
                                    self.interaction_operation.mutation_probability /= self.adaptation_hyperparameter
                                self.mutation_probability = self.interaction_operation.mutation_probability
                                child_replace_parent_count = 0

                            # muation to generate the offspring model
                            y_pred = model(x, mutation=True).squeeze()
                        else:
                            y_pred = model(x, mutation=False).squeeze()
                        net_optim.zero_grad()
                        structure_optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        loss.backward()
                        net_optim.step()
                        structure_optim.step()
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            train_result[name].append(metric_fun(
                                y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

                        if index % 100 == 0:
                            self.after_train(self.param_save_dir, name='round' + str(int(index / 100)))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            epoch_time = int(time.time() - epoch_start_time)
            logging.info('Epoch {0}/{1}'.format(epoch + 1, epochs))

            eval_str = "{0}s".format(epoch_time)

            for name in self.metrics:
                eval_str += " - " + name + \
                            ": {0: .4f}".format(epoch_logs[name])

            if do_validation:
                for name in self.metrics:
                    eval_str += " - " + "val_" + name + \
                                ": {0: .4f}".format(epoch_logs["val_" + name])
            logging.info(eval_str)

    def fit_n_1(self, x=None, y=None, batch_size=None, epochs=1, initial_epoch=0, validation_split=0.,
                shuffle=True):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        do_validation = False
        if validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        model = self.train()
        loss_func = self.loss_func
        net_optim = self.net_optim
        structure_optim = self.structure_optim
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        print("Cognitive EvoLutionary Search period")
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))

        for epoch in range(initial_epoch, epochs):
            epoch_start_time = time.time()
            epoch_logs = {}
            # total_loss_epoch = 0
            train_result = {}

            parent_num = self.population_size
            parent_models = [self.new() for x in range(parent_num)]
            parent_loss = [float('inf') for x in range(parent_num)]
            try:
                with tqdm(enumerate(train_loader)) as t:
                    for index, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        if index % self.mutation_step_size == 0:
                            # update parent models and parent loss by replacing the worst parent model with the current model  
                            current_y_pred = model(x, mutation=False).squeeze()
                            current_loss = loss_func(current_y_pred, y.squeeze(), reduction='sum')
                            max_loss = max(parent_loss)
                            worst_parent_index = parent_loss.index(max_loss)
                            parent_models[worst_parent_index].replace(self)
                            parent_loss[worst_parent_index] = current_loss

                            # apply the crossover mechanism to the parent models
                            self.crossover(parent_models)
                            # muation to generate the offspring model
                            y_pred = model(x, mutation=True).squeeze()
                        else:
                            y_pred = model(x, mutation=False).squeeze()
                        net_optim.zero_grad()
                        structure_optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        loss.backward()
                        net_optim.step()
                        structure_optim.step()
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            train_result[name].append(metric_fun(
                                y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

                        if index % 100 == 0:
                            self.after_train(self.param_save_dir, name='round' + str(int(index / 100)))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            epoch_time = int(time.time() - epoch_start_time)
            logging.info('Epoch {0}/{1}'.format(epoch + 1, epochs))

            eval_str = "{0}s".format(epoch_time)

            for name in self.metrics:
                eval_str += " - " + name + \
                            ": {0: .4f}".format(epoch_logs[name])

            if do_validation:
                for name in self.metrics:
                    eval_str += " - " + "val_" + name + \
                                ": {0: .4f}".format(epoch_logs["val_" + name])
            logging.info(eval_str)

    def fit_n_plus_1(self, x=None, y=None, batch_size=None, epochs=1, initial_epoch=0, validation_split=0.,
                     shuffle=True):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        do_validation = False
        if validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(
                np.concatenate(x, axis=-1)),
            torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256
        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        model = self.train()
        loss_func = self.loss_func
        net_optim = self.net_optim
        structure_optim = self.structure_optim
        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        print("Cognitive EvoLutionary Search period")
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))

        for epoch in range(initial_epoch, epochs):
            epoch_start_time = time.time()
            epoch_logs = {}
            # total_loss_epoch = 0
            train_result = {}

            parent_num = self.population_size
            parent_models = [self.new() for x in range(parent_num)]
            parent_loss = [float('inf') for x in range(parent_num)]
            child_replace_parent_count = 0
            try:
                with tqdm(enumerate(train_loader)) as t:
                    for index, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        if index % self.mutation_step_size == 0:
                            # update parent models and parent loss only if the current model is better than one of the parent models
                            current_y_pred = model(x, mutation=False).squeeze()
                            current_loss = loss_func(current_y_pred, y.squeeze(), reduction='sum')
                            max_loss = max(parent_loss)
                            if current_loss < max_loss:
                                worst_parent_index = parent_loss.index(max_loss)
                                parent_models[worst_parent_index].replace(self)
                                parent_loss[worst_parent_index] = current_loss
                                child_replace_parent_count += 1

                            # adapt the mutation_probability accroding to the 1/5 successful rule
                            if index % (self.mutation_step_size * self.adaptation_step_size) == 0 and index != 0:
                                # self.replace_count(self.param_save_dir, child_replace_parent_count)
                                if child_replace_parent_count < 2:
                                    self.interaction_operation.mutation_probability *= 0.99
                                elif child_replace_parent_count > 2:
                                    self.interaction_operation.mutation_probability /= 0.99
                                self.mutation_probability = self.interaction_operation.mutation_probability
                                child_replace_parent_count = 0
                            
                            # apply the crossover mechanism to the parent models
                            self.crossover(parent_models)
                            # muation to generate the offspring model
                            y_pred = model(x, mutation=True).squeeze()
                        else:
                            y_pred = model(x, mutation=False).squeeze()
                        net_optim.zero_grad()
                        structure_optim.zero_grad()
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        loss.backward()
                        net_optim.step()
                        structure_optim.step()
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            train_result[name].append(metric_fun(
                                y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

                        if index % 100 == 0:
                            self.after_train(self.param_save_dir, name='round' + str(int(index / 100)))

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            # epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            epoch_time = int(time.time() - epoch_start_time)
            logging.info('Epoch {0}/{1}'.format(epoch + 1, epochs))

            eval_str = "{0}s".format(epoch_time)

            for name in self.metrics:
                eval_str += " - " + name + \
                            ": {0: .4f}".format(epoch_logs[name])

            if do_validation:
                for name in self.metrics:
                    eval_str += " - " + "val_" + name + \
                                ": {0: .4f}".format(epoch_logs["val_" + name])
            logging.info(eval_str)

    def after_train(self, param_save_dir, name=""):
        state = {'alpha': self.linear.alpha,
                 'beta': self.interaction_operation.beta,
                 }
        if name == "":
            # Saving the final value of alpha and beta weight 
            param_save_path = os.path.join(self.param_save_dir,
                                           'alpha_beta-c' + str(CELS_Config['CELS']['c']) + '-mu' + str(
                                               CELS_Config['CELS']['mu']) + '-embedding_size' + str(
                                               self.embedding_size) + '.pth')
        else:
            # Saving the current value of alpha and beta weight in the evolution period
            param_save_path = os.path.join(param_save_dir, "evolution", "alpha_beta",
                                           'alpha_beta-c' + str(CELS_Config['CELS']['c']) + '-mu' + str(
                                               CELS_Config['CELS']['mu']) + '-embedding_size' + str(
                                               self.embedding_size) + "_" + name + '.pth')
        torch.save(state, param_save_path)

        selected_interaction_type = self.interaction_operation.selected_interaction_type
        if name == "":
            # Saving the final value of interaction types 
            param_save_file_path = os.path.join(param_save_dir, 'interaction_type-embedding_size-' +
                                                str(self.embedding_size) + '.pkl')
        else:
            # Saving the current value of interaction types in the evolution period
            param_save_file_path = os.path.join(param_save_dir, "evolution", "operation_type",
                                                'interaction_type-embedding_size-' +
                                                str(self.embedding_size) + "_" + name + '.pkl')
        with open(param_save_file_path, 'wb') as f:
            pkl.dump(selected_interaction_type, f)

        # Saving the value of mutation_probability in the evolution period
        mutation_probability_save_file_path = os.path.join(param_save_dir, 'mutation_probability' + '.txt')
        with open(mutation_probability_save_file_path, 'a') as f:
            f.write(str(self.mutation_probability) + '\n')

    def replace_count(self, param_save_dir, count):
        save_file_path = os.path.join(param_save_dir, 'child_replace_parent_count' + '.txt')
        with open(save_file_path, 'a') as f:
            f.write(str(count) + '\n')

    def crossover(self, parent_models):
        """
        crossover mechanism: select the fittest operation (of which interaction has the largest relevance (beta)) from the population
        """
        p_model = parent_models[0]
        beta = p_model.interaction_operation.beta
        beta_vstack = beta
        interaction = p_model.interaction_operation.selected_interaction_type
        interaction_vstack = interaction
        interaction_vstack = interaction_vstack.to(self.device)
        for i in range(1, len(parent_models)):
            p_model = parent_models[i]
            beta = p_model.interaction_operation.beta
            beta_vstack = torch.vstack((beta_vstack, beta))
            interaction = p_model.interaction_operation.selected_interaction_type
            interaction = interaction.to(self.device)
            interaction_vstack = torch.vstack((interaction_vstack, interaction))

        max_beta, index = torch.max(beta_vstack, dim=0)
        self.interaction_operation.beta.weight = Parameter(max_beta)
        index = index.unsqueeze(dim=0)
        selected_interaction_type = interaction_vstack.gather(0, index)
        self.interaction_operation.selected_interaction_type = selected_interaction_type.squeeze()






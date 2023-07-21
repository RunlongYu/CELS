import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .baseModel import BaseModel
from layer.interactionLayer import InteractionLayer
from layer.linearLayer import NormalizedWeightedLinearLayer
from layer.mlpLayer import DNN
from sklearn.metrics import *
from config.configs import CELS_Config, General_Config


class Functioning_Model(BaseModel):
    def __init__(self, feature_columns, feature_index, selected_interaction_type, interaction_fc_output_dim=1,
                 alpha=None, beta=None,
                 dnn_hidden_units=CELS_Config['ModelFunctioning']['dnn_hidden_units'], dnn_dropout=0, embedding_size=20,
                 activation='tanh', seed=1024, device='cpu'):
        super(Functioning_Model, self).__init__()
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.embedding_size = embedding_size
        self.device = device
        reduce_sum = True
        if device == 'cpu':
            torch.manual_seed(seed)
        else:
            torch.cuda.manual_seed(seed)
        reduce_sum = False
        feature_num = len(feature_columns)
        interaction_pair_num = int((feature_num * (feature_num - 1)) / 2)
        self.dnn = DNN(interaction_pair_num * interaction_fc_output_dim, dnn_hidden_units,
                       dropout_rate=dnn_dropout, use_bn=False)
        self.dnn_linear = nn.Linear(
            dnn_hidden_units[-1], 1, bias=False).to(device)
        if alpha == None and beta == None:
            self.use_alpha = False
            self.use_beta = False
        else:
            self.use_alpha = True
            self.use_beta = True
        self.linear = NormalizedWeightedLinearLayer(feature_columns=feature_columns, feature_index=feature_index,
                                                    alpha=alpha, use_alpha=self.use_alpha,
                                                    alpha_activation=activation,
                                                    device=device)
        self.interaction_operation = InteractionLayer(input_dim=embedding_size, feature_columns=feature_columns,
                                                      feature_index=feature_index, beta=beta, use_beta=self.use_beta,
                                                      interaction_fc_output_dim=interaction_fc_output_dim,
                                                      selected_interaction_type=selected_interaction_type,
                                                      device=device, reduce_sum=reduce_sum)

    def forward(self, x):
        # wide part
        linear_logit = self.linear(x)
        
        # deep part
        interation_out = self.interaction_operation(x)
        interation_out = self.dnn(interation_out)
        interation_logit = self.dnn_linear(interation_out)
        
        out = linear_logit + interation_logit
        return torch.sigmoid(out)

    def before_train(self):
        self.metrics_names = ["loss"]
        all_parameters = self.parameters()
        structure_params = set([self.linear.alpha, self.interaction_operation.beta])
        net_params = [i for i in all_parameters if i not in structure_params]
        self.net_optim = self.get_net_optim(net_params)
        self.loss_func = F.binary_cross_entropy
        self.metrics = self.get_metrics(["binary_crossentropy", "auc"])

    def get_net_optim(self, net_params):
        optimizer = optim.Adam(net_params, lr=float(General_Config['general']['net_optim_lr']))
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

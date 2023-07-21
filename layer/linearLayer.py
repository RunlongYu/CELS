import torch
import torch.nn as nn
from utils.function_utils import create_embedding_matrix


class LinearLayer(nn.Module):
    def __init__(self, feature_columns, feature_index, init_std=0.0001, reduce_sum=True):
        super(LinearLayer, self).__init__()
        self.feature_index = feature_index
        self.feature_columns = feature_columns
        self.reduce_sum = reduce_sum
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, linear=True, sparse=False)

    def forward(self, X):
        embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in
            self.feature_columns]
        if self.reduce_sum:
            linear_logit = torch.sum(
                torch.cat(embedding_list, dim=-1), dim=-1, keepdim=False)
        else:
            linear_logit = torch.cat(embedding_list, dim=-1)

        return linear_logit


class NormalizedWeightedLinearLayer(nn.Module):
    def __init__(self, feature_columns, feature_index, alpha=None, use_alpha=True,
                 alpha_activation='tanh', device='cpu'):
        super(NormalizedWeightedLinearLayer, self).__init__()
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.device = device
        self.use_alpha = use_alpha
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std=0.001, linear=True, sparse=False)
        if alpha == None:
            self.alpha = self.create_structure_param(len(self.feature_columns), init_mean=0.5, init_radius=0.001)
        else:
            self.alpha = alpha
        self.activate = nn.Tanh() if alpha_activation == 'tanh' else nn.Identity()

    def create_structure_param(self, length, init_mean, init_radius):
        structure_param = nn.Parameter(
            torch.empty(length).uniform_(
                init_mean - init_radius,
                init_mean + init_radius))
        structure_param.requires_grad = True
        return structure_param

    def forward(self, X):
        embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in
            self.feature_columns]
        if self.use_alpha:
            linear_logit = torch.sum(
                torch.mul(torch.cat(embedding_list, dim=1).squeeze(-1), (self.activate(self.alpha))), dim=-1,
                keepdim=True)
        else:
            linear_logit = torch.sum(
                torch.cat(embedding_list, dim=1).squeeze(-1), dim=-1,
                keepdim=True)
        return linear_logit

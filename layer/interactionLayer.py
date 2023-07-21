import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.function_utils import create_embedding_matrix, generate_pair_index

Interaction_Types = [
    'pointwise_addition',
    'hadamard_product',
    'concatenation_layer',
    'generalized_product'
]

Interation_Operations_Dict = {
    'pointwise_addition': lambda input_dim: PointWiseAddition(input_dim=input_dim),
    'hadamard_product': lambda input_dim: HadamardProduct(input_dim=input_dim),
    'concatenation_layer': lambda input_dim: Concatenation(input_dim=input_dim),
    'generalized_product': lambda input_dim: GeneralizedProduct(input_dim=input_dim)
}


class MixedOp(nn.Module):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(MixedOp, self).__init__()
        self.input_dim = input_dim
        self._ops = nn.ModuleList()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        for interaction_type in Interaction_Types:
            op = Interation_Operations_Dict[interaction_type](input_dim=input_dim)
            self._ops.append(op)
        if output_dim == input_dim:
            self.use_fc = False
        else:
            self.use_fc = True
            self.fc = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x1, x2, weights):
        sum = torch.zeros((x1.shape[0], x1.shape[1], self.output_dim)).to(self.device)
        for index, op in enumerate(self._ops):
            interaction_result = op(x1, x2)
            if self.use_fc:
                interaction_result = self.fc(interaction_result)
            interaction_type_weight = weights[:, index].unsqueeze(1)
            sum += torch.mul(interaction_result, interaction_type_weight)
        return sum


class InteractionLayer(nn.Module):
    def __init__(self, input_dim, feature_columns, feature_index, selected_interaction_type,
                 mutation_threshold=0.2, mutation_probability=0.5,
                 beta=None, use_beta=True,
                 interaction_fc_output_dim=1,
                 beta_activation='tanh', device='cpu',
                 reduce_sum=True):
        super(InteractionLayer, self).__init__()
        self.input_dim = input_dim
        self.feature_columns = feature_columns
        self.feature_index = feature_index
        self.selected_interaction_type = selected_interaction_type
        self.device = device
        self.reduce_sum = reduce_sum
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std=0.001, sparse=False)
        self.register_buffer('pair_indexes',
                             torch.tensor(generate_pair_index(len(self.feature_columns), 2)))
        self.interaction_pair_number = len(self.pair_indexes[0])
        self.use_beta = use_beta
        if beta == None:
            self.beta = self.create_structure_param(len(selected_interaction_type), init_mean=0.5, init_radius=0.001)
        else:
            self.beta = beta
        self.batch_norm = torch.nn.BatchNorm1d(len(selected_interaction_type), affine=False, momentum=0.01, eps=1e-3)
        self.mixed_operation = MixedOp(input_dim=input_dim, output_dim=interaction_fc_output_dim, device=device)
        self.activate = nn.Tanh() if beta_activation == 'tanh' else nn.Identity()

        self.mask_weight = self.generate_mask_weight()
        self.mutation_threshold = mutation_threshold
        self.mutation_probability = mutation_probability

    def forward(self, X, mutation=False):
        embedding_list = [self.embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for feat in
            self.feature_columns]
        embed_matrix = torch.cat(embedding_list, dim=1)
        feat_i, feat_j = self.pair_indexes
        embed_i = torch.index_select(embed_matrix, 1, feat_i)
        embed_j = torch.index_select(embed_matrix, 1, feat_j)
        if mutation:
            self.interaction_mutation()
        interaction_matrix = self.mixed_operation(embed_i, embed_j, self.mask_weight)
        normed_interaction_matrix = self.batch_norm(interaction_matrix)
        if self.use_beta:
            weighted_interaction_matrix = torch.mul(normed_interaction_matrix, self.activate(self.beta.unsqueeze(-1)))
        else:
            weighted_interaction_matrix = normed_interaction_matrix
        if self.reduce_sum:
            return torch.sum(torch.flatten(weighted_interaction_matrix, start_dim=-2, end_dim=-1), dim=-1, keepdim=True)
        else:
            return torch.flatten(weighted_interaction_matrix, start_dim=-2, end_dim=-1)

    def generate_mask_weight(self):
        mask_weight = torch.zeros((self.interaction_pair_number, len(Interation_Operations_Dict)))
        for index, interaction_type in enumerate(self.selected_interaction_type):
            mask_weight[index][interaction_type] = 1.
        return mask_weight.to(self.device)

    def create_structure_param(self, length, init_mean, init_radius):
        structure_param = nn.Parameter(
            torch.empty(length).uniform_(
                init_mean - init_radius,
                init_mean + init_radius))
        structure_param.requires_grad = True
        return structure_param

    @torch.no_grad()
    def interaction_mutation(self):
        for index in range(self.interaction_pair_number):
            if abs(self.beta[index]) < self.mutation_threshold:
                if random.random() < self.mutation_probability:
                    # randomly choose a different interaction type and reset the beta of the interaction 
                    mutation_interaction_type = (self.selected_interaction_type[index] + random.randint(1, 3)) % 4
                    self.mask_weight[index][self.selected_interaction_type[index]] = 0.
                    self.selected_interaction_type[index] = mutation_interaction_type
                    self.mask_weight[index][mutation_interaction_type] = 1.
                    self.beta[index] = 0.5


class PointWiseAddition(nn.Module):
    def __init__(self, input_dim):
        super(PointWiseAddition, self).__init__()
        self.input_dim = input_dim

    def forward(self, x1, x2):
        return torch.add(x1, x2)


class HadamardProduct(nn.Module):
    def __init__(self, input_dim):
        super(HadamardProduct, self).__init__()
        self.input_dim = input_dim

    def forward(self, x1, x2):
        return torch.mul(x1, x2)


class Concatenation(nn.Module):
    def __init__(self, input_dim):
        super(Concatenation, self).__init__()
        self.fc = nn.Linear(input_dim * 2, input_dim, bias=False)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        x = self.fc(x)
        x = F.relu(x)
        return x


class GeneralizedProduct(nn.Module):
    def __init__(self, input_dim):
        super(GeneralizedProduct, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x1, x2):
        x = torch.mul(x1, x2)
        x = self.fc(x)
        return x

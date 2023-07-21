import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, dropout_rate=0, use_bn=False):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')
        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

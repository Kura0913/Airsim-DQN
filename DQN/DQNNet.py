import torch
import torch.nn as nn

class SubModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SubModule, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        x = self.leaky_relu(self.bn(self.fc(x)))
        x = self.dropout(x)
        return x

class DQNNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_repeats=3):
        super(DQNNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.initial_fc = nn.Linear(state_dim, hidden_dim)
        self.initial_bn = nn.BatchNorm1d(hidden_dim)
        
        self.submodules = nn.ModuleList([SubModule(hidden_dim, hidden_dim) for _ in range(num_repeats)])
        
        self.final_fc = nn.Linear(hidden_dim, action_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = self.leaky_relu(self.initial_bn(self.initial_fc(x)))
        x = self.dropout(x)
        
        for submodule in self.submodules:
            x = submodule(x)
        
        return self.final_fc(x).squeeze(0)
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, activation):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dense1_bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.dense2_bn = nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.activation = activation

    def forward(self, x):

            theoutput = F.relu(self.fc1(x))
            out = self.fc3(F.relu(self.fc2(theoutput))
            
        return out

# FIXME:
def get_network(cfg: DictConfig) -> nn.Module:
    return Net()
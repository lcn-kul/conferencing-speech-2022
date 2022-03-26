import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.model.config import Config


class PoolAtt(nn.Module):
    '''
    PoolAtt: Attention-Pooling module.
    '''

    def __init__(self, config: Config):
        super().__init__()

        self.linear1 = nn.Linear(config.dim_head_in, 1)
        self.linear2 = nn.Linear(config.dim_head_in, 1)

    def forward(self, x: Tensor):

        att = self.linear1(x)
        att = att.transpose(2, 1)
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x)
        x = x.squeeze(1)

        x = self.linear2(x)

        return x

class PoolAttFF(torch.nn.Module):
    '''
    PoolAttFF: Attention-Pooling module with additonal feed-forward network.
    '''         
    def __init__(self, config: Config):
        super().__init__()
        
        self.linear1 = nn.Linear(config.dim_head_in, 2*config.dim_head_in)
        self.linear2 = nn.Linear(2*config.dim_head_in, 1)
        
        self.linear3 = nn.Linear(config.dim_head_in, 1)
        
        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: Tensor):

        att = self.linear2(self.dropout(self.activation(self.linear1(x))))
        att = att.transpose(2,1)
        att = F.softmax(att, dim=2)
        x = torch.bmm(att, x) 
        x = x.squeeze(1)
        
        x = self.linear3(x)
        
        return x  


class HeadWrapper(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        # self.norm = nn.BatchNorm1d(config.dim_head_in)
        self.pooling = PoolAttFF(config)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:

        # x = self.norm(x)
        x = self.pooling(x)
        x = self.sigmoid(x)
        return x

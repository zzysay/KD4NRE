"""
The base network.
"""

import torch.nn as nn
from model.base_encoder.cgcn import GCNRelationModel


class BaseNetwork(nn.Module):

    def __init__(self, opt, emb_matrix=None):
        print(">> Current Model: BaseNetwork")
        super(BaseNetwork, self).__init__()
        self.base_model = GCNRelationModel(opt=opt, emb_matrix=emb_matrix)

    def forward(self, inputs):
        loss_part, aux_part = self.base_model(inputs)
        return loss_part, aux_part


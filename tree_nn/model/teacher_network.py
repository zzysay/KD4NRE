"""
The teacher network.
"""

from torch import nn
from utils import constant, torch_utils

from model.base import BaseModel
from model.base_network import BaseNetwork


class TeacherNetwork(nn.Module):

    def __init__(self, opt, emb_matrix=None):
        print(">> Current Model: TeacherNetwork")
        super(TeacherNetwork, self).__init__()

        if opt['base_mode'] == 0:  # start from scratch
            print("   Do not use base model.")
            self.inst_encoder = BaseNetwork(opt=opt, emb_matrix=emb_matrix)
        else:
            self.base_model_file = opt['save_dir'] + '/' + opt['base_id'] + '/best_model.pt'
            self.base_opt = torch_utils.load_config(self.base_model_file)
            if opt['base_mode'] == 1:  # load & fine tune
                print("   Fine-tune base model.")
                inst_base_model = BaseModel(self.base_opt)
                inst_base_model.load(self.base_model_file)
                self.inst_encoder = inst_base_model.model
            elif opt['base_mode'] == 2:  # load & fix pre-trained
                print("   Fix pre-trained base model.")
                inst_base_model = BaseModel(self.base_opt)
                inst_base_model.load(self.base_model_file)
                inst_base_model = inst_base_model.model
                for param in inst_base_model.parameters():
                    param.requires_grad = False
                inst_base_model.eval()
                self.inst_encoder = inst_base_model
            else:
                print('Illegal Parameter (base_mode).')
                assert False

        self.rel_matrix = nn.Embedding(opt['num_class'], opt['num_class'], padding_idx=constant.LABEL_TO_ID['no_relation'])
        self.opt = opt
        self.init_weights()

    def init_weights(self):
        self.rel_matrix.weight.data[1:, :].uniform_(-1.0, 1.0)  # keep padding dimension to be 0

    def forward(self, inst_inputs, inst_rel=None):
        if inst_rel is None:
            _inputs, pattern_dist, inst_rel = inst_inputs[:10], inst_inputs[10], inst_inputs[11]
        else:
            _inputs, pattern_dist = inst_inputs[:10], inst_inputs[10]
        rel_weight = self.rel_matrix(inst_rel)  # [batch_size, rel_dim]
        pattern_logits = rel_weight * pattern_dist
        (inst_logits, inst_pooling_output), inst_aux_part = self.inst_encoder(_inputs)

        final_logits = inst_logits + pattern_logits
        return final_logits, inst_logits, pattern_logits, self.rel_matrix, inst_pooling_output

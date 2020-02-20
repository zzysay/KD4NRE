"""
The student network.
"""

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from utils import constant, torch_utils

from model.base import BaseModel
from model.base_network import BaseNetwork


class StudentNetwork(nn.Module):

    def __init__(self, opt, emb_matrix=None):
        print(">> Current Model: StudentNetwork")
        super(StudentNetwork, self).__init__()

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

        self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim_attn'])
        self.ner_emb = nn.Embedding(constant.NER_NUM, opt['ner_dim_attn'])
        self.attn_layer = MultiAspectAttention(opt)
        self.final_linear = nn.Linear(2 * opt['hidden_dim'], opt['num_class'])
        self.opt = opt
        self.init_weights()

    def init_weights(self):

        self.pe_emb.weight.data.uniform_(-1.0, 1.0)
        self.ner_emb.weight.data.uniform_(-1.0, 1.0)

        init.xavier_uniform_(self.final_linear.weight, gain=1)
        self.final_linear.bias.data.fill_(0)

    def forward(self, inst_inputs):
        _inputs = inst_inputs[:10]
        inst_mask, inst_ner, subj_pos, obj_pos, subj_end, obj_end = inst_inputs[1], inst_inputs[3], inst_inputs[6], inst_inputs[7], inst_inputs[12], inst_inputs[13]
        batch_size, seq_len = inst_mask.size()

        inst_loss_part, inst_aux_part = self.inst_encoder(_inputs)
        inst_logits, inst_pooling_output = inst_loss_part
        inst_hidden, inst_vectors, inst_outputs = inst_aux_part

        # convert all negative PE numbers to positive indices.  e.g., -2 -1 0 1 will be mapped to 98 99 100 101
        subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)  # subj_pos [batch_size, seq_len]
        obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
        pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=-1)

        # ner features
        subj_ner_inputs = torch.gather(inst_ner, 1, subj_end.view(-1, 1).repeat(1, seq_len))
        obj_ner_inputs = torch.gather(inst_ner, 1, subj_end.view(-1, 1).repeat(1, seq_len))
        ner_features = torch.cat((self.ner_emb(subj_ner_inputs), self.ner_emb(obj_ner_inputs)), dim=-1)

        attn_logits, attn_hidden = self.attn_layer(inst_outputs, inst_mask, inst_hidden, pe_features, ner_features)

        final_logits = inst_logits + attn_logits
        return final_logits, inst_logits, attn_logits, inst_pooling_output


class MultiAspectAttention(nn.Module):
    """
    A multi-aspect attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + W_1f_pe + W_2f_ner)
    where x is the input, q is the query, and f_pe / f_ner is additional position / NER features.
    """

    def __init__(self, opt):
        super(MultiAspectAttention, self).__init__()
        self.input_size = opt['hidden_dim']
        self.query_size = opt['hidden_dim']
        self.pe_feature_size = 2 * opt['pe_dim_attn']
        self.ner_feature_size = 2 * opt['ner_dim_attn']
        self.attn_size = opt['attn_dim']

        self.ulinear = nn.Linear(self.input_size, self.attn_size, bias=False)
        self.vlinear = nn.Linear(self.query_size, self.attn_size, bias=False)
        self.pe_wlinear = nn.Linear(self.pe_feature_size, self.attn_size, bias=False)
        self.ner_wlinear = nn.Linear(self.ner_feature_size, self.attn_size, bias=False)
        self.tlinear = nn.Linear(self.attn_size, 1, bias=False)

        self.attn_linear = nn.Linear(opt['hidden_dim'], opt['num_class'])
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.vlinear.weight.data.normal_(std=0.001)
        self.pe_wlinear.weight.data.normal_(std=0.001)
        self.ner_wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_()
        # self.tlinear.weight.data.normal_(std=0.001)

        init.xavier_uniform_(self.attn_linear.weight, gain=1)
        self.attn_linear.bias.data.fill_(0)

    def forward(self, x, x_mask, q, pe_f, ner_f):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, _ = x.size()

        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(batch_size, seq_len, self.attn_size)
        q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(batch_size, self.attn_size).unsqueeze(1).expand(batch_size, seq_len, self.attn_size)
        pe_f_proj = self.pe_wlinear(pe_f.view(-1, self.pe_feature_size)).contiguous().view(batch_size, seq_len, self.attn_size)
        ner_f_proj = self.ner_wlinear(ner_f.view(-1, self.ner_feature_size)).contiguous().view(batch_size, seq_len, self.attn_size)
        projs = [x_proj, q_proj, pe_f_proj, ner_f_proj]

        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)).view(batch_size, seq_len)

        # mask padding
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=1)
        # weighted average input vectors
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)  # [batch_size, hidden_dim]
        logits = self.attn_linear(outputs)
        logits = F.leaky_relu(logits)

        return logits, outputs

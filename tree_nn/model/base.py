"""
A trainer class for base encoder.
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import torch_utils
from model.base_network import BaseNetwork


class BaseModel(object):
    """ A wrapper class for the training and evaluation of models. """

    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = BaseNetwork(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    @staticmethod
    def unpack_batch(batch, cuda):
        # 0 words, 1 masks, 2 pos, 3 ner, 4 deprel, 5 head, 6 subj_positions, 7 obj_positions,
        # 8 subj_type, 9 obj_type, 10 pattern_dist, 11 rels, 12 subj_end, 13 obj_end, 14 orig_idx

        if cuda:
            inputs = [Variable(b.cuda()) for b in batch[:10]]
            labels = Variable(batch[11].cuda())
        else:
            inputs = [Variable(b) for b in batch[:10]]
            labels = Variable(batch[11])
        tokens = batch[0]
        head = batch[5]
        subj_pos = batch[6]
        obj_pos = batch[7]
        lens = batch[1].eq(0).long().sum(1).squeeze()
        return inputs, labels, tokens, head, subj_pos, obj_pos, lens

    def update(self, batch):
        """ Run a step of forward and backward model update. """
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = self.unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        (logits, pooling_output), aux_part = self.model(inputs)
        loss = self.criterion(logits, labels)

        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.opt['conv_l2'] * self.model.conv_l2()
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = self.unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[-1]

        # forward
        self.model.eval()
        (logits, pooling_output), aux_part = self.model(inputs)
        loss = self.criterion(logits, labels)

        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs)))]
        return predictions, probs, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

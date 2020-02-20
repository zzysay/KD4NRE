"""
A trainer class for teacher network.
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import torch_utils
from model.teacher_network import TeacherNetwork


class TeacherModel(object):
    """ A wrapper class for the training and evaluation of models. """

    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = TeacherNetwork(opt=opt, emb_matrix=emb_matrix)
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
            inputs = [Variable(b.cuda()) for b in batch[:-1]]
            labels = Variable(batch[11].cuda())
        else:
            inputs = [Variable(b) for b in batch[:-1]]
            labels = Variable(batch[11])
        tokens = batch[0]
        head = batch[5]
        subj_pos = batch[6]
        obj_pos = batch[7]
        lens = batch[1].eq(0).long().sum(1).squeeze()
        return inputs, labels, tokens, head, subj_pos, obj_pos, lens

    def tsd_loss(self, final_logits):
        """ calculate top score difference as a extra loss. """
        k = 3  # k > 2
        final_logits = F.softmax(final_logits, dim=1)
        batch_size, num_class = final_logits.size()
        total_loss = 0
        for i in range(batch_size):
            top_logits = final_logits[i].topk(k)[0]
            max_logit = top_logits[0]
            secondary_logits = 0
            for j in range(1, k):
                secondary_logits += (top_logits[j] / k)
            total_loss += (max_logit - secondary_logits)
        return total_loss / batch_size

    def update(self, batch):
        """ Run a step of forward and backward model update. """
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = self.unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        tch_final_logits, tch_inst_logits, tch_pattern_logits, tch_rel_matrix, tch_pooling_output = self.model(inputs)
        ground_loss = self.criterion(tch_final_logits, labels)
        top_score_loss = self.tsd_loss(tch_final_logits)
        loss = ground_loss + top_score_loss if self.opt['use_tsd'] else ground_loss

        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.opt['conv_l2'] * self.model.conv_l2()
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (tch_pooling_output ** 2).sum(1).mean()

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        ground_loss_val = ground_loss.data.item()
        top_score_loss_val = top_score_loss.data.item()
        return loss_val, ground_loss_val, top_score_loss_val

    def predict(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = self.unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[-1]

        # forward
        self.model.eval()
        tch_final_logits, _, _, _, _ = self.model(inputs)
        loss = self.criterion(tch_final_logits, labels)

        probs = F.softmax(tch_final_logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(tch_final_logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs)))]
        return predictions, probs, loss.data.item()

    def predict_all(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = self.unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[-1]

        # forward
        self.model.eval()
        tch_final_logits, tch_inst_logits, tch_pattern_logits, rel_matrix, _ = self.model(inputs)

        # fianl_loss = self.criterion(tch_final_logits, labels)
        final_probs = F.softmax(tch_final_logits, dim=1).data.cpu().numpy().tolist()
        final_predictions = np.argmax(tch_final_logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, final_predictions, final_probs = [list(t) for t in
                                                 zip(*sorted(zip(orig_idx, final_predictions, final_probs)))]

        # inst_loss = self.criterion(tch_inst_logits, labels)
        inst_probs = F.softmax(tch_inst_logits, dim=1).data.cpu().numpy().tolist()
        inst_predictions = np.argmax(tch_inst_logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, inst_predictions, inst_probs = [list(t) for t in
                                               zip(*sorted(zip(orig_idx, inst_predictions, inst_probs)))]

        # pattern_loss = self.criterion(tch_pattern_logits, labels)
        pattern_probs = F.softmax(tch_pattern_logits, dim=1).data.cpu().numpy().tolist()
        pattern_predictions = np.argmax(tch_pattern_logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, pattern_predictions, pattern_probs = [list(t) for t in
                                                 zip(*sorted(zip(orig_idx, pattern_predictions, pattern_probs)))]
        return final_predictions, inst_predictions, pattern_predictions, final_probs, inst_probs, pattern_probs

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

"""
A trainer class for student network.
"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import torch_utils
from model.student_network import StudentNetwork


class StudentModel(object):
    """ A wrapper class for the training and evaluation of models. """

    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = StudentNetwork(opt=opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def prepare(self, pre_outputs):
        pst_outputs = list()
        for output in pre_outputs:
            output_logits = torch.from_numpy(output)
            output_logits = output_logits.cuda() if self.opt['cuda'] else output_logits
            output_logits = Variable(output_logits, requires_grad=False)
            pst_outputs.append(output_logits)
        return pst_outputs if len(pst_outputs) > 1 else pst_outputs[0]

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

    def update(self, batch):
        """ Run a step of forward and backward model update. """
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = self.unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        stu_final_logits, stu_inst_logits, stu_reread_logits, stu_pooling_output = self.model(inputs)
        loss = self.criterion(stu_final_logits, labels)
        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.opt['conv_l2'] * self.model.conv_l2()
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (stu_pooling_output ** 2).sum(1).mean()

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def update_with_teacher(self, batch, tch_outputs=None, base_mode=0):
        """ Run a step of forward and backward model update. """
        inputs, ground_labels, tokens, head, subj_pos, obj_pos, lens = self.unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()

        stu_final_logits, stu_inst_logits, stu_attn_logits, stu_pooling_output = self.model(inputs)
        ground_loss = self.criterion(stu_final_logits, ground_labels)
        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            ground_loss += self.opt['conv_l2'] * self.model.conv_l2()
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            ground_loss += self.opt['pooling_l2'] * (stu_pooling_output ** 2).sum(1).mean()

        tch_final_logits, tch_inst_logits, tch_pattern_logits = self.prepare(tch_outputs)
        teacher_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(stu_final_logits / self.opt['temperature'], dim=1),
                                                           F.softmax(tch_final_logits / self.opt['temperature'], dim=1))

        if base_mode == 0 or base_mode == 1:  # start from scratch | load & fine tune
            hint_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(stu_inst_logits, dim=1), F.softmax(tch_inst_logits, dim=1)) + \
                        nn.KLDivLoss(reduction='batchmean')(F.log_softmax(stu_attn_logits, dim=1), F.softmax(tch_pattern_logits, dim=1))
        if base_mode == 2:  # load & fix pre-trained
            hint_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(stu_attn_logits, dim=1), F.softmax(tch_pattern_logits, dim=1))

        loss = (1 - self.opt['lambda_kd']) * ground_loss + self.opt['lambda_kd'] * (teacher_loss + self.opt['lambda_ht'] * hint_loss)

        # backward
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val, ground_loss.data.item(), teacher_loss.data.item(), hint_loss.data.item(), tch_pattern_logits.data.cpu().numpy(), stu_attn_logits.data.cpu().numpy()

    def predict(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = self.unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[-1]

        # forward
        self.model.eval()
        stu_final_logits, _, _, _ = self.model(inputs)
        loss = self.criterion(stu_final_logits, labels)

        probs = F.softmax(stu_final_logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(stu_final_logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx, predictions, probs)))]
        return predictions, probs, loss.data.item()

    def predict_all(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = self.unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[-1]

        # forward
        self.model.eval()
        stu_final_logits, stu_inst_logits, stu_reread_logits, _ = self.model(inputs)

        # fianl_loss = self.criterion(stu_final_logits, labels)
        final_probs = F.softmax(stu_final_logits, dim=1).data.cpu().numpy().tolist()
        final_predictions = np.argmax(stu_final_logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, final_predictions, final_probs = [list(t) for t in zip(*sorted(zip(orig_idx, final_predictions, final_probs)))]

        # inst_loss = self.criterion(stu_inst_logits, labels)
        inst_probs = F.softmax(stu_inst_logits, dim=1).data.cpu().numpy().tolist()
        inst_predictions = np.argmax(stu_inst_logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, inst_predictions, inst_probs = [list(t) for t in zip(*sorted(zip(orig_idx, inst_predictions, inst_probs)))]

        # aux_loss = self.criterion(stu_reread_logits, labels)
        aux_probs = F.softmax(stu_reread_logits, dim=1).data.cpu().numpy().tolist()
        aux_predictions = np.argmax(stu_reread_logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, aux_predictions, aux_probs = [list(t) for t in zip(*sorted(zip(orig_idx, aux_predictions, aux_probs)))]
        return final_predictions, inst_predictions, aux_predictions, final_probs, inst_probs, aux_probs

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

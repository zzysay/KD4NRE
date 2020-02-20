"""
Train the student network on TACRED.
"""

import os
import time
import math
import numpy as np
from shutil import copyfile
from datetime import datetime

from config import Configure
from utils.vocab import Vocab
from data.loader import DataLoader
from utils import torch_utils, scorer, constant, helper

from model.base import BaseModel
from model.teacher import TeacherModel
from model.student import StudentModel

# make opt
opt = vars(Configure.c_gcn())
opt['num_class'] = len(constant.LABEL_TO_ID)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False, pattern_file=opt['pattern_file'])
dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True, pattern_file=opt['pattern_file'])
test_batch = DataLoader(opt['data_dir'] + '/test.json', opt['batch_size'], opt, vocab, evaluation=True, pattern_file=opt['pattern_file'])

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1\ttest_loss\ttest_f1")

# print model info
helper.print_config(opt)

# model
base_outputs = ['placeholder' for _ in range(0, len(train_batch))]
if opt['base_mode'] == 3:
    base_outputs = []
    base_model_file = opt['save_dir'] + '/' + opt['base_id'] + '/best_model.pt'
    print("Loading base model from {}".format(base_model_file))
    base_opt = torch_utils.load_config(base_model_file)
    base_model = BaseModel(opt=base_opt)
    base_model.load(base_model_file)
    base_model.model.eval()
    for _, batch in enumerate(train_batch):
        inputs = [b.cuda() for b in batch[:10]] if opt['cuda'] else [b for b in batch[:10]]
        base_logits, _, _ = base_model.model(inputs)
        base_outputs.append([base_logits.data.cpu().numpy()])

teacher_outputs = []
if opt['use_teacher']:
    teacher_model_file = opt['save_dir'] + '/' + opt['teacher_id'] + '/best_model.pt'
    print("Loading teacher model from {}".format(teacher_model_file))
    teacher_opt = torch_utils.load_config(teacher_model_file)
    teacher_model = TeacherModel(opt=teacher_opt)
    teacher_model.load(teacher_model_file)
    teacher_model.model.eval()
    for _, batch in enumerate(train_batch):
        inputs = [b.cuda() for b in batch[:-1]] if opt['cuda'] else [b for b in batch[:-1]]
        tch_final_logits, tch_inst_logits, tch_match_logits, _, _ = teacher_model.model(inputs)
        teacher_outputs.append([tch_final_logits.data.cpu().numpy(),
                                tch_inst_logits.data.cpu().numpy(),
                                tch_match_logits.data.cpu().numpy()])

student_model = StudentModel(opt=opt, emb_matrix=emb_matrix)

id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
kd_para_str = '> epoch {}/{}:  lambda_kd: {:.6f}'
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f}, ground_loss = {:.6f}, teacher_loss = {:.6f}, hint_loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

if opt['teacher_anneal']:
    print('>> max_lkd:', opt['lambda_kd'], 'grade:', opt['anneal_grade'])
    max_lkd = current_lkd = opt['lambda_kd']
    grade_epoch = math.ceil(opt['num_epoch'] / opt['anneal_grade'])  # num epoch of each grade
    grade_span_lkd = max_lkd / (opt['anneal_grade'] - 1)  # span of each grade
    print(grade_epoch, grade_span_lkd)
else:
    grade_epoch = grade_span_lkd = 0
    current_lkd = opt['lambda_kd']

# start training
max_dev_f1, max_dev_id = 0, 0
max_test_f1, max_test_id = 0, 0
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0

    if opt['teacher_anneal']:
        if epoch != 1 and (epoch % grade_epoch == 1 or grade_epoch == 1):
            current_lkd = max(0.0, current_lkd - grade_span_lkd)
            opt['lambda_kd'] = current_lkd
        print(kd_para_str.format(epoch, opt['num_epoch'], opt['lambda_kd']))

    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1

        if opt['use_teacher']:
            loss, ground_loss, teacher_loss, hint_loss, tch_match_logits, stu_aux_logits = student_model.update_with_teacher(batch, tch_outputs=teacher_outputs[i], base_mode=opt['base_mode'])
            train_loss += loss
            if global_step % opt['log_step'] == 0:
                duration = time.time() - start_time
                print(format_str.format(datetime.now(), global_step, max_steps, epoch, opt['num_epoch'], loss, ground_loss, teacher_loss, hint_loss, duration, current_lr))
        else:
            loss = student_model.update(batch)
            train_loss += loss
            if global_step % opt['log_step'] == 0:
                duration = time.time() - start_time
                print(format_str.format(datetime.now(), global_step, max_steps, epoch, opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    dev_loss = 0
    for _, batch in enumerate(dev_batch):
        preds, _, loss = student_model.predict(batch)
        predictions += preds
        dev_loss += loss
    predictions = [id2label[p] for p in predictions]
    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions)

    max_dev_f1, max_dev_id = (dev_f1, epoch) if max_dev_f1 < dev_f1 else (max_dev_f1, max_dev_id)
    train_loss = train_loss / train_batch.num_examples * opt['batch_size']  # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']

    # eval on test
    print("Evaluating on test set...")
    predictions = []
    test_loss = 0
    for _, batch in enumerate(test_batch):
        preds, _, loss = student_model.predict(batch)
        predictions += preds
        test_loss += loss
    predictions = [id2label[p] for p in predictions]
    test_p, test_r, test_f1 = scorer.score(test_batch.gold(), predictions)

    max_test_f1, max_test_id = (test_f1, epoch) if max_test_f1 < test_f1 else (max_test_f1, max_test_id)
    test_loss = test_loss / test_batch.num_examples * opt['batch_size']
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}, test_loss = {:.6f}, test_f1 = {:.4f}".format(epoch, train_loss, dev_loss, dev_f1, test_loss, test_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1, test_loss, test_f1))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    student_model.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)

    # lr schedule
    if len(dev_f1_history) > opt['decay_epoch'] and dev_f1 <= dev_f1_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        student_model.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print("Training ended with {} epochs.".format(opt['num_epoch']))
print("Maximal dev  F1 {:.3f}% on epoch {}.".format(max_dev_f1 * 100, max_dev_id))
print("Maximal test F1 {:.3f}% on epoch {}.".format(max_test_f1 * 100, max_test_id))

"""
Configurations (parameters might not be optimal).
"""

import torch
import random
import argparse
import numpy as np


class Configure:

    @staticmethod
    def c_gcn():
        parser = argparse.ArgumentParser()
        # data and rnn
        parser.add_argument('--data_dir', type=str, default='dataset/tacred')
        parser.add_argument('--vocab_dir', type=str, default='dataset/vocab')
        parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
        parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
        parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
        parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
        parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
        parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
        parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
        parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
        parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
        parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
        parser.add_argument('--no-lower', dest='lower', action='store_false')
        parser.set_defaults(lower=False)
        parser.add_argument('--na_prob', type=float, default=0.5, help='the init probability of NA')

        parser.add_argument('--prune_k', default=1, type=int, help='Prune the dependency tree to <= K distance off the dependency path; set to -1 for no pruning.')
        parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
        parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
        parser.add_argument('--pooling_l2', type=float, default=0.003, help='L2-penalty for all pooling output.')  # 1
        parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
        parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

        parser.add_argument('--no-rnn', dest='rnn', action='store_false', help='Do not use RNN layer.')
        parser.add_argument('--rnn_hidden', type=int, default=200, help='RNN hidden state size.')
        parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
        parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

        # attention
        parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
        parser.add_argument('--pe_dim_attn', type=int, default=30, help='Position encoding dimension.')
        parser.add_argument('--ner_dim_attn', type=int, default=30, help='NER encoding dimension.')

        # training
        parser.add_argument('--lr', type=float, default=0.3, help='Applies to sgd and adagrad.')
        parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
        parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
        parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='sgd', help='Optimizer: sgd, adagrad, adam or adamax.')
        parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
        parser.add_argument('--batch_size', type=int, default=32, help='Training batch size.')  # 50
        parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
        parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
        parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
        parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
        parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
        parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
        parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

        # device
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
        parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

        # supervision
        parser.add_argument('--base_mode', default=0, type=int, help='0: off, 1: fine-tune, 2: fix-pretrained')
        parser.add_argument('--base_id', type=str, default='base', help='Model ID of base')
        parser.add_argument('--teacher', dest='use_teacher', action='store_true', help='Use teacher for supervision.')
        parser.add_argument('--no-teacher', dest='use_teacher', action='store_false')
        parser.add_argument('--teacher_id', type=str, default='teacher', help='Model ID of teacher')
        parser.set_defaults(use_teacher=True)
        parser.add_argument('--pattern', dest='use_pattern', action='store_true', help='Use pattern for supervision.')
        parser.add_argument('--no-pattern', dest='use_pattern', action='store_false')
        parser.set_defaults(use_pattern=True)
        parser.add_argument('--pattern_file', type=str, default='../bipartite/output/type2prob.json')
        parser.add_argument('--pattern_num', type=int, default=27, help='Num of pattern.')
        parser.add_argument('--load_mode', dest='use_pretrained', action='store_true', help='Use pretrained base model as encoder.')

        # knowledge distillation
        parser.add_argument('--tsd', dest='use_tsd', action='store_true', help='Use top score different loss when training.')
        parser.add_argument('--no-tsd', dest='use_tsd', action='store_false')
        parser.set_defaults(use_tsd=True)
        parser.add_argument('--anneal', dest='teacher_anneal', action='store_true', help='Use teacher annealing when training.')
        parser.add_argument('--no-anneal', dest='teacher_anneal', action='store_false')
        parser.set_defaults(teacher_anneal=True)
        parser.add_argument('--anneal_grade', type=int, default=100, help='Num for teacher annealing from 1 to 0.')
        parser.add_argument('--lambda_kd', type=float, default=1.0, help='Weight factor between ground truth and teacher prediction')
        parser.add_argument('--lambda_ht', type=float, default=1.8, help='Weight factor for hint learning')
        parser.add_argument('--temperature', type=float, default=1., help='Temperature for Knowledge Distillation.')

        # load
        parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
        parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

        args = parser.parse_args()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if args.cpu:
            args.cuda = False
        elif args.cuda:
            torch.cuda.manual_seed(args.seed)

        return args

    @staticmethod
    def eval():
        parser = argparse.ArgumentParser()

        parser.add_argument('model_id', type=str, default='00', help='Directory of the model.')
        parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
        parser.add_argument('--data_dir', type=str, default='dataset/tacred')
        parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
        parser.add_argument('--out', type=str, default='results.txt', help="Save model predictions to this dir.")
        parser.add_argument('--na_prob', type=float, default=0.5, help='the init probability of NA')

        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
        parser.add_argument('--cpu', action='store_true')

        args = parser.parse_args()
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if args.cpu:
            args.cuda = False
        elif args.cuda:
            torch.cuda.manual_seed(args.seed)

        return args

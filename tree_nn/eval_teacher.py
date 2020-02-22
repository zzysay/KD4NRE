"""
Run evaluation with saved models for teacher network.
"""

import json
from config import Configure
from utils.vocab import Vocab
from data.loader import DataLoader
from model.teacher import TeacherModel
from utils import torch_utils, scorer, constant, helper

args = Configure.eval()

# load opt
model_file = 'saved_models/' + args.model_id + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
teacher_model = TeacherModel(opt)
teacher_model.load(model_file)

# load vocab
vocab_file = 'saved_models/' + args.model_id + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True, pattern_file=opt['pattern_file'])

helper.print_config(opt)
id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])

final_predictions, inst_predictions, pattern_predictions = [], [], []
all_final_probs, all_inst_probs, all_pattern_probs = [], [], []
for i, b in enumerate(batch):
    final_preds, inst_preds, pattern_preds, final_probs, inst_probs, pattern_probs = teacher_model.predict_all(b)
    final_predictions += final_preds
    inst_predictions += inst_preds
    pattern_predictions += pattern_preds
    all_final_probs += final_probs
    all_inst_probs += inst_probs
    all_pattern_probs += pattern_probs
final_predictions = [id2label[p] for p in final_predictions]
inst_predictions = [id2label[p] for p in inst_predictions]
pattern_predictions = [id2label[p] for p in pattern_predictions]
print('\n >> Final Prediction:')
_, _, _ = scorer.score(batch.gold(), final_predictions, verbose=True)
print('\n >> Instance Prediction:')
_, _, _ = scorer.score(batch.gold(), inst_predictions, verbose=True)
print('\n >> Pattern Prediction:')
_, _, _ = scorer.score(batch.gold(), pattern_predictions, verbose=True)

# save probability scores
# if len(args.out) > 0:
#     outfile = 'saved_models/' + args.model_id + '/' + args.out
#     with open(outfile, 'w') as fw:
#         for f_prob, i_prob, p_prob in zip(all_final_probs, all_inst_probs, all_pattern_probs):
#             fw.write(json.dumps([round(p, 4) for p in f_prob]))
#             fw.write('\r\n')
#             fw.write(json.dumps([round(p, 4) for p in i_prob]))
#             fw.write('\r\n')
#             fw.write(json.dumps([round(p, 4) for p in p_prob]))
#             fw.write('\r\n')
#             fw.write('\r\n')
#     print("Prediction scores saved to {}.".format(outfile))
print("Evaluation ended.")

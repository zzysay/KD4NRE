"""
Run evaluation with saved models for base encoder.
"""

import json
from config import Configure
from utils.vocab import Vocab
from model.base import BaseModel
from data.loader import DataLoader
from utils import torch_utils, scorer, constant, helper

args = Configure.eval()

# load opt
model_file = 'saved_models/' + args.model_id + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
opt['na_prob'] = args.na_prob
base_model = BaseModel(opt)
base_model.load(model_file)

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

predictions = []
all_probs = []
for i, b in enumerate(batch):
    preds, probs, _ = base_model.predict(b)
    predictions += preds
    all_probs += probs
predictions = [id2label[p] for p in predictions]
_, _, _ = scorer.score(batch.gold(), predictions, verbose=True)

# save probability scores
# if len(args.out) > 0:
#     outfile = 'saved_models/' + args.model_id + '/' + args.out
#     with open(outfile, 'w') as fw:
#         for prob in all_probs:
#             fw.write(json.dumps([round(p, 4) for p in prob]))
#             fw.write('\r\n')
#     print("Prediction scores saved to {}.".format(outfile))
print("Evaluation ended.")

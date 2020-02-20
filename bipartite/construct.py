"""
Construct the bipartite graph.
"""

import json
from bipartite import constant


class ConstructByType:

    def __init__(self):
        """ initialize file path. """
        self.train_file = '../tree_nn/dataset/tacred/train.json'
        # self.dev_file = 'dataset/tacred/dev.json'
        # self.test_file = 'dataset/tacred/test.json'

        self.type2rel = dict()
        self.rel2type = dict()
        self.type2prob = dict()

        self.type2rel_file = 'output/type2rel.json'
        self.rel2type_file = 'output/rel2type.json'
        self.type2prob_file = 'output/type2prob.json'

    def statistics(self):
        """ statistics type number """
        with open(self.train_file) as infile:
            data = json.load(infile)

        _num = 0
        _positive_num = 0

        subj_type = set()
        obj_type = set()
        for sample in data:
            # print(_num, _positive_num)
            _num += 1

            relation = sample['relation']
            if relation == 'no_relation' or relation == 'Other':
                continue
            _positive_num += 1

            subj_type.add(sample['subj_type'])
            obj_type.add(sample['obj_type'])

        print(len(subj_type), subj_type)  # 2
        print(len(obj_type), obj_type)  # 17

    def get_type_pattern(self):
        """ generate original type-relation mapping (type2rel, rel2type). """
        with open(self.train_file) as infile:
            data = json.load(infile)

        _num = 0
        _na_num = 0

        for sample in data:
            # print(_num, _positive_num)
            _num += 1

            relation = sample['relation']
            if relation == 'no_relation' or relation == 'Other':
                _na_num += 1
                continue

            subj_type = sample['subj_type']
            obj_type = sample['obj_type']

            pattern = 'SUBJ-' + subj_type + '|:|' + 'OBJ-' + obj_type

            if relation in self.rel2type:
                if pattern in self.rel2type[relation]:
                    self.rel2type[relation][pattern] = self.rel2type[relation][pattern] + 1
                else:
                    self.rel2type[relation][pattern] = 1
            else:
                self.rel2type[relation] = {pattern: 1}

            if pattern in self.type2rel:
                if relation in self.type2rel[pattern]:
                    self.type2rel[pattern][relation] = self.type2rel[pattern][relation] + 1
                else:
                    self.type2rel[pattern][relation] = 1
            else:
                self.type2rel[pattern] = dict({relation: 1})

        # print('>> total_sentence_num na_num:', _num, _na_num)

    def sort(self):
        """ sort patterns or relations by probability. """
        for rel in self.rel2type:
            pattern_dict = self.rel2type[rel]
            pattern_list = sorted(pattern_dict.items(), key=lambda v: v[1], reverse=True)
            self.rel2type[rel] = {k: v for k, v in pattern_list}

        for pattern in self.type2rel:
            rel_dict = self.type2rel[pattern]
            rel_list = sorted(rel_dict.items(), key=lambda v: v[1], reverse=True)
            self.type2rel[pattern] = {k: v for k, v in rel_list}

    def normalization(self):
        """ normalize the distribution for each pattern and relation. """
        for rel in self.rel2type:
            count = 0
            for pattern in self.rel2type[rel]:
                count += self.rel2type[rel][pattern]
            for pattern in self.rel2type[rel]:
                num = self.rel2type[rel][pattern]
                self.rel2type[rel][pattern] = (num, num / count)

        for pattern in self.type2rel:
            count = 0
            for rel in self.type2rel[pattern]:
                count += self.type2rel[pattern][rel]
            for rel in self.type2rel[pattern]:
                num = self.type2rel[pattern][rel]
                self.type2rel[pattern][rel] = (num, num / count)
        # print('>> original_pattern_num:', len(self.type2rel))

    def get_probability(self):
        """ calculate probabilities for each pattern. """
        for pattern in self.type2rel:
            rel_dict = self.type2rel[pattern]
            prob_list = [0 for _ in range(len(constant.LABEL_TO_ID))]
            for rel in rel_dict:
                prob_list[constant.LABEL_TO_ID[rel]] = round(rel_dict[rel][1], 6)
            self.type2prob[pattern] = prob_list

    def write_json(self):
        """ write all json into files. """
        with open(self.rel2type_file, 'w', encoding='utf8') as fw:
            json.dump(self.rel2type, fw, ensure_ascii=False, indent=4)
        with open(self.type2rel_file, 'w', encoding='utf8') as fw:
            json.dump(self.type2rel, fw, ensure_ascii=False, indent=4)
        with open(self.type2prob_file, 'w', encoding='utf8') as fw:
            json.dump(self.type2prob, fw, ensure_ascii=False, indent=4)
        print("> Done.")


if __name__ == '__main__':
    cbt = ConstructByType()
    # cbt.statistics()
    cbt.get_type_pattern()
    cbt.sort()
    cbt.normalization()
    cbt.get_probability()
    cbt.write_json()

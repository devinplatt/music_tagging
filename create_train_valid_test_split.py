#!/usr/bin/env python
import argparse
from collections import Counter, defaultdict
import csv
import operator
import os
import random

parser = argparse.ArgumentParser(description='Create a train/valid/test split for Magnatagatune.')
parser.add_argument('--annotations',
                    help='The path to the annotations_final.csv file.')
parser.add_argument('--output_dir',
                    help='The directory in which to store output files (train/valid/test splits).')

args = parser.parse_args()
annotations_final_fname = args.annotations
output_dir = args.output_dir

rows_tag = []
with open(annotations_final_fname) as annotationscsv:
    csvreader = csv.reader(annotationscsv, delimiter='\t')
    for row in csvreader:
        rows_tag.append(row)
tags = rows_tag[0][1:-1]
bad_audio_paths = [
    '6/norine_braun-now_and_zen-08-gently-117-146.mp3',
    '8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3',
    '9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.mp3'
]
paths = dict()
tag_dict = defaultdict(list)
for row in rows_tag[1:]:
    path = row[-1]
    if path in bad_audio_paths:
        continue
    key = int(row[0])
    paths[key] = path
    for i in range(len(tags)):
        if int(row[i+1]) != 0:  # it == 1
            tag_dict[key].append(tags[i])
tags_inverse = dict()
for index, tag in enumerate(tags):
    tags_inverse[tag] = index
examples = []
for key, tag_list in tag_dict.items():
    for tag in tag_list:
        examples.append((key, tag))
c = Counter([x[1] for x in examples])
top_tags = [x[0] for x in c.most_common()[:50]]
top_tags = sorted(top_tags)
top_tags_inverse = {i: tag for i, tag in enumerate(top_tags)}
print(top_tags)
train_valid_test_examples = [ex for ex in examples if ex[1] in top_tags]
train_valid_test_keys = list(set(ex[0] for ex in train_valid_test_examples))
print(len(train_valid_test_examples))
print(len(train_valid_test_keys))


def get_mp3_stem(fname):
    s = fname.split('-')
    s = '-'.join(s[:-2])
    return s


def train_valid_test_split(keys, percent_train):
    key_paths = [paths[key] for key in keys]
    stems = list(set([get_mp3_stem(fname) for fname in key_paths]))
    random.shuffle(stems)
    # Here, number_train includes validation data.
    number_train = int(len(stems) * percent_train)
    number_valid = int(number_train / 10)
    train_stems = set(stems[number_valid:number_train])
    valid_stems = set(stems[:number_valid])
    test_stems = set(stems[number_train:])
    print('number_train: {}'.format(number_train))
    print('stems : {}'.format(len(stems)))
    train_keys = set(fname for fname in key_paths if get_mp3_stem(fname) in train_stems)
    valid_keys = set(fname for fname in key_paths if get_mp3_stem(fname) in valid_stems)
    test_keys = set(fname for fname in key_paths if get_mp3_stem(fname) in test_stems)
    return train_keys, valid_keys, test_keys 


def pairs2xy(pairs):
    random.shuffle(pairs)
    # X is the train file paths
    Xl = [paths[ex[0]] for ex in pairs]
    yl = [ex[1] for ex in pairs]
    return Xl, yl

train_keys, valid_keys, test_keys = train_valid_test_split(train_valid_test_keys,
                                                           percent_train=0.9)
train_examples = [ex for ex in train_valid_test_examples if paths[ex[0]] in train_keys]
valid_examples = [ex for ex in train_valid_test_examples if paths[ex[0]] in valid_keys]
test_examples = [ex for ex in train_valid_test_examples if paths[ex[0]] in test_keys]
X_train, y_train = pairs2xy(train_examples)
X_valid, y_valid = pairs2xy(valid_examples)
X_test, y_test = pairs2xy(test_examples)
print('train_keys: {}'.format(len(train_keys)))
print('valid_keys: {}'.format(len(valid_keys)))
print('test_keys: {}'.format(len(test_keys)))
print('train_examples: {}'.format(len(train_examples)))
print('valid_examples: {}'.format(len(valid_examples)))
print('test_examples: {}'.format(len(test_examples)))

# We save fname-tag examples for train, valid, test.
# We also save the label_map of numerical value <-> tag name
labelmap_fname = os.path.join(output_dir, 'labelmap.txt')
train_fname = os.path.join(output_dir, 'train.tsv')
valid_fname = os.path.join(output_dir, 'valid.tsv')
test_fname = os.path.join(output_dir, 'test.tsv')
with open(labelmap_fname, 'w') as fo:
    for tag in top_tags:
        fo.write(tag + '\n')
with open(train_fname, 'w') as fo:
    for x, y in zip(X_train, y_train):
        fo.write(x + '\t' + y + '\n')
with open(valid_fname, 'w') as fo:
    for x, y in zip(X_valid, y_valid):
        fo.write(x + '\t' + y + '\n')
with open(test_fname, 'w') as fo:
    for x, y in zip(X_test, y_test):
        fo.write(x + '\t' + y + '\n')


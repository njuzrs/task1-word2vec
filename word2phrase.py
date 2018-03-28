# -*- coding: utf-8 -*-
"""
Created on Sat Feb 03 16:08:30 2018

@author: zrssch
"""
'''
this file processes the dataset and extract phrase as a unit instead word
'''

from collections import defaultdict
import argparse
from itertools import tee, izip_longest

def filter_vocab(vocab, min_count):
    # remove word fewer than min_count
    return dict((k, v) for k, v in vocab.iteritems() if v >= min_count)
    
def pairwise(iterable):
    # combine near words together as a pair
    a, b = tee(iterable)
    next(b)
    return izip_longest(a, b)
    
def learn_vocab_from_train_iter(train_iter):
    vocab = defaultdict(int)
    train_words = 0
    for line in train_iter:
        if line == []:   
          continue        
        for pair in pairwise(line):
            vocab[pair[0]] += 1
            if None not in pair:
                vocab[pair] += 1
            train_words += 1
    return vocab, train_words
    
def train_model(train_iter, min_count=5, threshold=100.0, sep='_'):
    vocab_iter, train_iter = tee(train_iter)
    vocab, train_words = learn_vocab_from_train_iter(vocab_iter)
    print("Done Vocab", len(vocab), train_words)
    vocab = filter_vocab(vocab, min_count)
    print("Filtered Vocab", len(vocab))
    cnt = 0
    for line in train_iter:
        out_sentence = []
        pairs = pairwise(line)
        for pair in pairs:
            pa = vocab.get(pair[0])
            pb = vocab.get(pair[1])
            pab = vocab.get(pair)

            if all((pa, pb, pab)):
                score = float(pab - min_count) / pa / pb * train_words
            else:
                score = 0.0
            if score > threshold:
                next(pairs)
                out_sentence.append(sep.join(pair))
                cnt += 1
            else:
                out_sentence.append(pair[0])
        yield out_sentence
    print('cnt: ', cnt)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default="ptb.train.txt")
    parser.add_argument('--output_file', type=str, default="ptbphrase.train.txt")
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--sep", type=str, default='_')
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.05)
    args = parser.parse_args()
    
    fr = open(args.train_file, 'r')
    train_file = (line.strip().split() for line in fr.readlines())
    out = train_file
    for i in range(args.iters):
        this_thresh = max(args.threshold - (i * args.discount * args.threshold), 0.0)
        print("Iteration: %d Threshold: %6.2f" % (i, this_thresh))
        out = train_model(out, min_count=args.min_count, threshold=this_thresh, sep=args.sep)
    out_fh = open(args.output_file, 'w')
    for row in out:
        out_fh.write(' '.join(row) + '\n')
    out_fh.close()
    fr.close()

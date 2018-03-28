# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 20:12:38 2018

@author: zrssch
"""

import numpy as np
import argparse
import time
import struct
import random

EXP_TABLE_SIZE = 1000
MAX_EXP = 6

class wordinfo:
    # the information about words
    def __init__(self, word):
        self.word = word
        self.count = 0
        self.path = None # the path to the word
        self.code = None # huffman encode
        
        
class vocab:
    def __init__(self, filename, min_count):
        vocabulary = []
        word2index = {}
        words_count = 0
        # construct vocabulary contained with wordinfo, a dict that word to its index in vocabulary   
        fr = open(filename, 'r')
        for line in fr.readlines():
            words = line.split()
            for word in words:
                if word not in word2index:
                    word2index[word] = len(vocabulary)
                    vocabulary.append(wordinfo(word))
                vocabulary[word2index[word]].count += 1 
                words_count += 1
        fr.close()
        self.vocabulary = vocabulary
        self.word2index = word2index
        self.words_count = words_count
        print("total words in file: ", self.words_count)
        print("the size of vocabulary: ", len(self.vocabulary))
        self.sortvocab(min_count) # make the words count less than min_count to <unk> and sorted
    
    def __getitem__(self, i):
        return self.vocabulary[i]
        
    def __len__(self):
        return len(self.vocabulary)
        
    def __iter__(self):
        return iter(self.vocabulary)
        
    def __contains__(self, i):
        return i in self.word2index
        
    def index(self, words):
        temp = []
        for word in words:
            if word in self.word2index:
                temp.append(self.word2index[word])
            else:
                temp.append(self.word2index['<unk>'])
        return temp
        
    def sortvocab(self, min_count):
        sortedvocab = []
        unk_count = 0
        if '<unk>' in self.word2index:
            sortedvocab.append(wordinfo('<unk>'))
            sortedvocab[0].count = self.vocabulary[self.word2index['<unk>']].count
        else:
            sortedvocab.append(wordinfo('<unk>'))
            
        for word in self.vocabulary:
            if word.word != '<unk>':
                if word.count < min_count:
                    sortedvocab[0].count += word.count
                    unk_count += 1
                else:
                    sortedvocab.append(word)
        print("the number of words less than min_count: ", unk_count)            
        sortedvocab.sort(key=lambda word: word.count, reverse=True) 
        ind = {}
        for i, word in enumerate(sortedvocab):
            ind[word.word] = i
        self.vocabulary = sortedvocab
        self.word2index = ind
        
    def huffman(self):
        vocab_size = len(self)
        count = [t.count for t in self] + [1e15]*(vocab_size - 1)
        parent = [0] * (2*vocab_size - 2)
        binary = [0] * (2*vocab_size - 2)
        
        pos1 = vocab_size - 1
        pos2 = vocab_size
        
        for i in xrange(vocab_size - 1):
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min1 = pos1
                    pos1 -= 1
                else:
                    min1 = pos2
                    pos2 += 1
            else:
                min1 = pos2
                pos2 += 1
                
            if pos1 >= 0:
                if count[pos1] < count[pos2]:
                    min2 = pos1
                    pos1 -= 1
                else:
                    min2 = pos2
                    pos2 += 1
                    
            else:
                min2 = pos2
                pos2 += 1
                
            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = vocab_size + i
            parent[min2] = vocab_size + i
            binary[min2] = 1
            
        root_idx = 2*vocab_size - 2
        for i, word in enumerate(self):
            path = []
            code = []
            
            node_idx = i
            while node_idx < root_idx:
                if node_idx >= vocab_size:
                    path.append(node_idx)
                code.append(binary[node_idx])
                node_idx = parent[node_idx]
            path.append(root_idx)
            
            word.path = [j- vocab_size for j in path[::-1]]
            word.code = code[::-1]
        

class negtable:
    def __init__(self, vocab):
        norm = sum([word.count**0.75 for word in vocab])
        table_size = 1e8
        table = np.zeros(table_size, dtype=np.uint32)
        i = 0
        prob = 0
        for j, word in enumerate(vocab):
            prob += float(word.count**0.75)/norm
            while i < table_size and float(i) /table_size < prob:
                table[i] = j
                i += 1
        self.table = table
    
    def sample(self, count):
        ind = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in ind]


def save(vocab, syn0, fo, binary):
    print('Saving model to', fo)
    dim = len(syn0[0])
    if binary:
        fo = open(fo, 'wb')
        fo.write('%d %d\n' % (len(syn0), dim))
        fo.write('\n')
        for word, vector in zip(vocab, syn0):
            fo.write('%s ' % word.word)
            for s in vector:
                fo.write(struct.pack('f', s))
            fo.write('\n')
    else:
        fo = open(fo, 'w')
        fo.write('%d %d\n' % (len(syn0), dim))
        for word, vector in zip(vocab, syn0):
            w = word.word
            vector_str = ' '.join([str(s) for s in vector])
            fo.write('%s %s\n' % (w, vector_str))
    fo.close()
     
def train(vocab, filename, alpha, win, cbow, syn0, syn1, dim, neg, table, subsample, subsample_rate, fo, binary, epochs):
    start_alpha_ori = alpha
    for ep in range(epochs):
        start_alpha = alpha
        word_count = 0
        fr = open(filename, 'r')
        txt = fr.readlines()
        random.shuffle(txt)
        for line in txt:
            line = line.strip()
            if not line:
                continue
            sentence = []
            temp = vocab.index(line.split())
            
            if subsample:
                for i in temp:
                    f = float(vocab.vocabulary[i].count)/vocab.words_count
                    p = 1 - np.sqrt(subsample_rate/f)
                    if p<=0:
                        sentence.append(i)
                    else:
                        ran = np.random.randint(low=1, high=1e8)
                        if ran > p*(1e8):
                            sentence.append(i)
            else:
                sentence = vocab.index(line.split())
                
            if len(sentence)<2:
                continue
            for pos, word in enumerate(sentence):
                if word_count%10000 == 0:
                    alpha = start_alpha*(1- float(word_count)/vocab.words_count)
                    alpha = max(start_alpha_ori*0.0001, alpha)
                    print("alpha: %f process: %d of %d (%.2f%%)"%(alpha, word_count, vocab.words_count, float(word_count) / vocab.words_count * 100))
                
                current_win = np.random.randint(low=1, high=win+1)
                context_start = max(pos-current_win, 0)
                context_end = min(pos+current_win+1, len(sentence))
                context = sentence[context_start:pos] + sentence[pos+1:context_end]
                
                if cbow:
                    neu1 = np.mean(np.array([syn0[c] for c in context]), axis=0)
                    neu1e = np.zeros(dim)
                    
                    if neg > 0:
                        classifiers = [(word, 1)] + [(target, 0) for target in table.sample(neg)]
                    else:
                        classifiers = zip(vocab[word].path, vocab[word].code)
                        
                    for target, label in classifiers:
                        z = np.dot(neu1, syn1[target])
                        # print neu1, syn1[target], z
                        if z <= -MAX_EXP:
                            p = 0.0
                        elif z>= MAX_EXP:
                            p = 1.0
                        else:
                            p = sigmoid[int((float(z)+6)*EXP_TABLE_SIZE/(2*MAX_EXP))]
                        g = alpha * (label - p)
                        neu1e += g * syn1[target]
                        syn1[target] += g * neu1
                    
                    for context_word in context:
                        syn0[context_word] += neu1e
                        
                else:
                    for context_word in context:
                        neu1e = np.zeros(dim)
                        if neg > 0:
                            classifiers = [(word, 1)] + [(target, 0) for target in table.sample(neg)]
                        else:
                            classifiers = zip(vocab[word].path, vocab[word].code)
                        
                        for target, label in classifiers:
                            z = np.dot(syn0[context_word], syn1[target])
                            if z <= -MAX_EXP:
                                p = 0.0
                            elif z>= MAX_EXP:
                                p = 1.0
                            else:
                                p = sigmoid[int((float(z)+6)*EXP_TABLE_SIZE/(2*MAX_EXP))]
                            g = alpha*(label - p)
                            neu1e = g * syn1[target]
                            syn1[target] += g * syn0[context_word]
                        syn0[context_word] += neu1e
                    
                word_count += 1
        fr.close()
    save(vocab, syn0, fo, binary)
        
        
def init_net(dim, vocab_size):
    syn0 = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(vocab_size, dim))
    syn1 = np.zeros(shape=(vocab_size, dim))
    return (syn0, syn1)
                      

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training filename', dest='filename', default='ptb.train.txt', type=str)
    parser.add_argument('-model', help='Output model file', dest='fo', default='vec.txt', type=str)
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=0, type=int)
    parser.add_argument('-negative', help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax', dest='neg', default=25, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=100, type=int)
    parser.add_argument('-alpha', help='Starting alpha', dest='alpha', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int) 
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5, type=int)
    parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0, type=int)
    parser.add_argument('-subsample', help='>0 for subsample when train', dest='subsample', default=1, type=int)
    parser.add_argument('-subsample_rate', help='subsample_rate (subsample must > 0)', dest='subsample_rate', default=5e-3, type=float)
    parser.add_argument('-epochs', help='number of epoches for train', dest='epochs', default=1, type=int)
    args = parser.parse_args()
    
    # compute sigmoid value
    sigmoid = []
    for i in range(EXP_TABLE_SIZE):
        tmp = np.exp((float(2*i)/EXP_TABLE_SIZE - 1)* MAX_EXP)
        sigmoid.append(tmp/(tmp +1))
    
    vocabulary = vocab(args.filename, args.min_count)
    syn0, syn1 = init_net(args.dim, len(vocabulary))
    
    table = None
    if args.neg>0:
        table = negtable(vocabulary)
    else:
        vocabulary.huffman()
    
    t1 = time.time()
    train(vocabulary, args.filename, args.alpha, args.win, args.cbow, syn0, syn1, 
          args.dim, args.neg, table, args.subsample, args.subsample_rate, args.fo, args.binary, args.epochs)
    t2 = time.time()
    print('complete training and saving: ', (t2-t1)/60, 'minutes')
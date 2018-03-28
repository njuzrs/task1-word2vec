# -*- coding: utf-8 -*-
"""
Created on Sun Feb 04 16:19:44 2018

@author: zrssch
"""
'''
this file visualize the word2vec by tensorboard
'''

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import os

LOG_DIR = "log"
fr = open('subsp5e3_10ep_cbow.txt', 'r')
size = 100
words = []
embeddings = []
for line in fr.readlines()[1:]:
    line = line.strip().split()
    assert len(line) == size+1
    words.append(line[0])
    embeddings.append([float(i) for i in line[1:]])
fr.close()

words = np.array(words)
embeddings = np.array(embeddings)
np.savetxt(os.path.join(LOG_DIR, 'metadata.tsv'), words, fmt='%s')

session = tf.InteractiveSession()
embedding_var = tf.Variable(embeddings, name='embedding')
tf.global_variables_initializer().run()

saver = tf.train.Saver()
saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), global_step=0)

summary_writer = tf.summary.FileWriter(LOG_DIR)

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
projector.visualize_embeddings(summary_writer, config)

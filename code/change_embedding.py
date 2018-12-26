# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 14:46:19 2018

@author: lanwangbj
"""
import numpy as np
from tqdm import tqdm
def change_embedding(embedname):
    '''
    This function can be used by calling 'dat = change_embedding('parag')/('wiki')/('anything')' to change your embedding.
    And dat would be a dictionary with word-keys and embedding-list-values.
    '''
    embeddings_index = {}
    if embedname == 'parag': # 3 min
        f = open('./embeddings/paragram_300_sl999/paragram_300_sl999.txt','r', encoding='utf-8', newline='\n', errors='ignore')
        for line in tqdm(f):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            embeddings_index[word] = np.asarray(tokens[1:], dtype='float32')
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index
    elif embedname == 'wiki': # 1 min
        f = open('./embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
        for line in tqdm(f):
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            embeddings_index[word] = np.asarray(tokens[1:], dtype='float32')
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index
    else: # 3.55 min 
        f = open('./embeddings/glove.840B.300d/glove.840B.300d.txt','r', encoding='UTF-8')
        for line in tqdm(f):
            values = line.split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index

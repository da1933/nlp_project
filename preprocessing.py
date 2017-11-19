from __future__ import print_function

import json
import random
import numpy as np
from collections import Counter
import pickle as pickle
import scipy.stats

import operator
import re
import sys

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import random
from random import randint

def load_data(path): #load SNLI words
    '''
    Constructs 4 dictionaries with the same key values across the dictionaries
    '''
    #data = []
    label_dict = {'neutral':0,'contradiction':1,'entailment':2}
    excluded = 0
    hypothesis = {}
    premise = {}
    label = {}
    label_enc = {}
    with open(path, 'r') as f:
        #need this so indexing is continuous when sentences are skipped over
        idx = 0
        for i,line in enumerate(f):
            obj = json.loads(line)
            #skip these rows per readme
            if obj["gold_label"] == '-':
                excluded += 1
            else:
                label[idx] = obj["gold_label"]
                label_enc[idx] = label_dict[obj["gold_label"]]
                premise[idx] = obj["sentence1"]
                hypothesis[idx] = obj["sentence2"]
                idx += 1
    print('%s excluded' %excluded)
    return hypothesis, premise, label, label_enc


def load_embeddings(path,words_to_load,emb_dim): #load pre-trained GloVe embeddings
    with open(path) as f:
        loaded_embeddings = np.zeros((words_to_load, emb_dim))
        words = {}
        idx2words = {}
        ordered_words = []
        for i, line in enumerate(f):
            if i >= words_to_load: 
                break
            s = line.split()
            loaded_embeddings[i, :] = np.asarray(s[1:])
            words[s[0]] = i
            idx2words[i] = s[0]
            ordered_words.append(s[0])

    return loaded_embeddings, words, idx2words, ordered_words

            
def add_tokens_da(idx_mapping, embeddings, emb_dim):
    '''
    This function increases the index of the word to index mapping for GloVe so that
    0: padding index
    1: unk
    2: BoS
    Sentences are padded at the end
    '''
    words_cnt = Counter(idx_mapping)
    increment = Counter(dict.fromkeys(idx_mapping, 3))
    words_cnt = words_cnt + increment
    words_cnt['<PAD_IDX>'] = 0
    words_cnt['<UNK>'] = 1
    words_cnt['<BoS>'] = 2
    
    #insert embeddings for tokens
    #<BoS>
    embed = np.insert(embeddings,[0],np.random.rand(300),axis=0)
    #<UNK>
    embed = np.insert(embed,[0],np.random.rand(300),axis=0) 
    #<PAD_IDX>
    embed = np.insert(embed,[0],np.zeros(300),axis=0)
    
    return words_cnt, embed

def add_tokens_nti(idx_mapping, embeddings, emb_dim):
    '''
    This function increases the index of the word to index mapping for GloVe so that
    0: padding index
    1: unk
    Sentences are padded in front
    '''
    words_cnt = Counter(idx_mapping)
    increment = Counter(dict.fromkeys(idx_mapping, 3))
    words_cnt = words_cnt + increment
    words_cnt['<PAD_IDX>'] = 0
    words_cnt['<UNK>'] = 1
    
    #insert embeddings for tokens
    #<UNK>
    embed = np.insert(embeddings,[0],np.random.rand(300),axis=0) 
    #<PAD_IDX>
    embed = np.insert(embed,[0],np.zeros(300),axis=0)
    
    return words_cnt, embed


def clean_words(text_list): # Removes characters and makes all words lowercase
    unwanted_chars = ['\\','.',',','/','\'s']
    for i,word in enumerate(text_list):
        for ch in unwanted_chars:
            if ch in text_list[i]:
                text_list[i] = text_list[i].replace(ch,'')
            text_list[i] = text_list[i].lower()

            
def tokenize_da(text_dict, idx_mapping, pad_len):
    '''
    text_dict: dictionary with index as key, sentence as value
    returns dictionary with the index as key, sentenece mapped to index as value, and padded to pad_len
    '''
    tokenized_data = {}
    for i in range(len(text_dict.keys())):
        text_list = text_dict[i].split()
        clean_words(text_list)
        text_idx = []
        for word in text_list:
            try:
                text_idx.append(idx_mapping[word])
            except KeyError:
                #UNK token
                random_UNK_embedding = random.randint(1,101) # 1-100 is the range of UNK embedding idxs
                text_idx.append(random_UNK_embedding)
                continue
        #insert BoS token
        text_idx.insert(0,2)
        if len(text_idx) > pad_len:
            text_idx = text_idx[:pad_len]
        text_idx = np.concatenate((text_idx,np.zeros(max(pad_len-len(text_idx),0))))
                                    
        tokenized_data[i] = np.array(text_idx).astype(int)
    return tokenized_data

def tokenize_nti(text_dict, idx_mapping, pad_len):
    '''
    text_dict: dictionary with index as key, sentence as value
    returns dictionary with the index as key, sentenece mapped to index as value, and padded to pad_len
    '''
    tokenized_data = {}
    for i in range(len(text_dict.keys())):
        text_list = text_dict[i].split()
        clean_words(text_list)
        text_idx = []
        for word in text_list:
            try:
                text_idx.append(idx_mapping[word])
            except KeyError:
                #UNK token
                random_UNK_embedding = random.randint(1,101) # 1-100 is the range of UNK embedding idxs
                text_idx.append(random_UNK_embedding)
                continue
        #insert BoS token
        if len(text_idx) > pad_len:
            text_idx = text_idx[:pad_len]
        text_idx = np.concatenate((np.zeros(max(pad_len-len(text_idx),0)),text_idx))
                                    
        tokenized_data[i] = np.array(text_idx).astype(int)
    return tokenized_data
    
def word2vec(sent,embeddings):
    return [embeddings[idx].astype(np.float32) for idx in sent]

def preprocess4(sent):
	return ' '.join([x.strip() for x in re.split('(\W+)?', sent) if x.strip()])

def batch(dataset, indexes):
	return [dataset[i] for i in indexes]

# -*- coding: utf-8 -*-

import os
import csv
import pickle
import random
import numpy as np
from pytorch_pretrained_bert import BertTokenizer

def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './fastText/cc.zh.300.vec'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

class Tokenizer4Bert:
    def __init__(self, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)

    def text_to_sequence(self, text):
        words = text.split()
        sequence = [self.tokenizer.vocab[w] if w in self.tokenizer.vocab else self.tokenizer.vocab['[UNK]'] for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def _tokenize_chinese_text(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            output.append(' ')
            output.append(char)
            output.append(' ')
        else:
            output.append(char)
    return ''.join(output)

def _is_chinese_char(cp):
    """Checks whether cp is the codepoint of a CJK character."""
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  
        (cp >= 0x3000 and cp <= 0x303f) or # punctuation
        (cp >= 0x3400 and cp <= 0x4DBF) or  
        (cp >= 0x20000 and cp <= 0x2A6DF) or  
        (cp >= 0x2A700 and cp <= 0x2B73F) or  
        (cp >= 0x2B740 and cp <= 0x2B81F) or  
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  
        return True
    return False

class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
            f_csv = csv.reader(fin)
            for i, row in enumerate(f_csv):
                if i == 0:
                    continue
                else:
                    text_raw = _tokenize_chinese_text(row[1])
                text += text_raw + ' '
            fin.close()
                
        return text

    @staticmethod
    def __read_data__(fname, bert_tokenizer=None):
        all_data = []
        fin = open(fname, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
        f_csv = csv.reader(fin)
        for i, row in enumerate(f_csv):
            if i == 0:
                continue
            else:
                text_raw = row[1]
                if row[2] != '':
                    entities = row[2].split(';')
                else:
                    continue
                if row[4] != '':
                    key_entities = row[4].split(';')
                else:
                    key_entities = []
                for entity in entities:
                    if entity == '':
                        continue
                    aspect = entity
                    text_left, _, text_right = [s for s in text_raw.partition(entity)]
                    text_left = _tokenize_chinese_text(text_left)
                    text_right = _tokenize_chinese_text(text_right)
                    aspect = _tokenize_chinese_text(aspect)
                    _text_indices = bert_tokenizer.text_to_sequence(text_left+' '+aspect+' '+text_right)
                    _aspect_indices = bert_tokenizer.text_to_sequence(aspect)
                    bert_text_indices = bert_tokenizer.text_to_sequence('[CLS]') + _text_indices + \
                         bert_tokenizer.text_to_sequence('[SEP]') + _aspect_indices + bert_tokenizer.text_to_sequence('[SEP]')
                    bert_segment_indices = [0] * (len(_text_indices) + 2) + [1] * (len(_aspect_indices) + 1)
                    if entity in key_entities:
                        negative = 1
                    else:
                        negative = 0
                    data = {
                        'bert_text_indices': bert_text_indices,
                        'bert_segment_indices': bert_segment_indices,
                        'negative': negative,
                    }

                    all_data.append(data)
        fin.close()  
        return all_data

    def __init__(self, dataset='finance', embed_dim=300):
        print("preparing {0} dataset...".format(dataset))
        fname = {
            'finance': {
                'train': './datasets/Clean_Train_Data.csv',
                'test': './datasets/Clean_Test_Data.csv'
            },
        }
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        
        bert_tokenizer = Tokenizer4Bert('bert-base-chinese')
        #exit(0)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], bert_tokenizer))

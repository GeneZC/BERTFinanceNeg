# -*- coding: utf-8 -*-


import os
import csv
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from data_utils import _tokenize_chinese_text, Tokenizer4Bert
from models import BERT_SPC
from pytorch_pretrained_bert import BertModel

class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
        print("loading {0} tokenizer...".format(opt.dataset))
        self.bert_tokenizer = Tokenizer4Bert('bert-base-chinese')

        self.model_list = []
        for i, model_name in enumerate(opt.model_name_list):
            print('loading model {0}... '.format(model_name))
            bert = BertModel.from_pretrained('bert-base-chinese')
            model = nn.DataParallel(opt.model_class_list[i](bert, opt).to(opt.device))
            model.load_state_dict(torch.load(opt.state_dict_path_list[i]))
            # switch model to evaluation mode
            model.eval()
            self.model_list.append(model)
        
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, fname):
        fin = open(fname, 'r', encoding='utf-8-sig', newline='\n', errors='ignore')
        fin_csv = csv.reader(fin)
        fout = open('submission.csv', 'w', encoding='utf-8-sig', newline='')
        fout_csv = csv.writer(fout)
        header = ['id','negative','key_entity']
        fout_csv.writerow(header)
        for i, row in enumerate(fin_csv):
            if i == 0:
                continue
            else:
                key_entities = []
                uid = row[0]
                text_raw = row[1]
                entities = row[2].split(';')
                for entity in entities:
                    if entity == '' or entity == ' ':
                        break
                    aspect = entity
                    text_left, _, text_right = [s for s in text_raw.partition(entity)]
                    text_left = _tokenize_chinese_text(text_left)
                    text_right = _tokenize_chinese_text(text_right)
                    aspect = _tokenize_chinese_text(aspect)
                    _text_indices = self.bert_tokenizer.text_to_sequence(text_left+' '+aspect+' '+text_right)
                    _aspect_indices = self.bert_tokenizer.text_to_sequence(aspect)
                    bert_text_indices = [(self.bert_tokenizer.text_to_sequence('[CLS]') + _text_indices +\
                         self.bert_tokenizer.text_to_sequence('[SEP]') + _aspect_indices + self.bert_tokenizer.text_to_sequence('[SEP]'))[:512]]
                    bert_segment_indices = [([0] * (len(_text_indices) + 2) + [1] * (len(_aspect_indices) + 1))[:512]]
                    data = {
                        'bert_text_indices': torch.tensor(bert_text_indices),
                        'bert_segment_indices': torch.tensor(bert_segment_indices),
                    }
                    preds = []
                    for i, inputs_cols in enumerate(self.opt.inputs_cols_list):
                        t_inputs = [data[col].to(self.opt.device) for col in inputs_cols]
                        with torch.no_grad():
                            t_outputs = self.model_list[i](t_inputs)
                        t_preds = t_outputs.argmax(dim=1).cpu().numpy()[0]
                        preds.append(t_preds)
                    preds = max(preds, key=preds.count)
                    if preds == 1:
                        key_entities.append(entity)
                if len(key_entities) == 0:
                    fout_csv.writerow([uid, '0', ''])
                else:
                    fout_csv.writerow([uid, '1', ';'.join(key_entities)])
        fin.close()
        fout.close()

if __name__ == '__main__':
    model_classes = {
        'bert': BERT_SPC
    }
    dataset = 'finance'
    # set your trained models here
    model_state_dict_paths = {
        'bert': 'state_dict/bert_'+dataset+'.pkl',
    }
    input_colses = {
        'bert': ['bert_text_indices', 'bert_segment_indices'],
    }
    class Option(object): pass
    opt = Option()
    opt.model_name_list = ['bert']
    opt.model_class_list = [model_classes[model_name] for model_name in opt.model_name_list]
    opt.inputs_cols_list = [input_colses[model_name] for model_name in opt.model_name_list]
    opt.dataset = dataset
    opt.state_dict_path_list = [model_state_dict_paths[model_name] for model_name in opt.model_name_list]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.polarities_dim = 3
    opt.dropout = 0.3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = Inferer(opt)
    inf.evaluate('./datasets/Clean_Test_Data.csv')

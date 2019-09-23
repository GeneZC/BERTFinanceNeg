# -*- coding: utf-8 -*-

import math
import random
import torch

class BucketIterator(object):
    def __init__(self, data, batch_size, shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x['bert_text_indices']))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_trunc_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    @staticmethod
    def pad_trunc_data(batch_data):
        batch_bert_text_indices = []
        batch_bert_segment_indices = []
        batch_negative = []
        bert_max_len = max([len(t['bert_text_indices']) for t in batch_data])
        for item in batch_data:
            bert_text_indices, bert_segment_indices, negative = \
                item['bert_text_indices'], item['bert_segment_indices'], item['negative']
            bert_text_padding = [0] * (bert_max_len - len(bert_text_indices))
            bert_segment_padding = [0] * (bert_max_len - len(bert_segment_indices))
            batch_bert_text_indices.append((bert_text_indices + bert_text_padding)[:512])
            batch_bert_segment_indices.append((bert_segment_indices + bert_segment_padding)[:512])
            batch_negative.append(negative)
        return { 
                'bert_text_indices': torch.tensor(batch_bert_text_indices), 
                'bert_segment_indices': torch.tensor(batch_bert_segment_indices), 
                'negative': torch.tensor(batch_negative), 
                }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]

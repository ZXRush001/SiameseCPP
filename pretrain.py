import os
import re

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
import pickle
from termcolor import colored
from transformers import BertModel, BertTokenizer
import random
import pandas as pd
from sklearn.model_selection import train_test_split
torch.cuda.set_device(1)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def genData(file, max_len):
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23}
    with open(file, 'r') as inf:
        lines = inf.read().splitlines()

    long_pep_counter = 0
    pep_codes = []
    labels = []
    pep_seq = []
    max_seq_len = 70
    for pep in lines:
        pep, label = pep.split(",")
        labels.append(int(label))
        input_seq = ' '.join(pep)
        # input_seq = input_seq + ' [PAD]'*(max_seq_len-len(pep))
        input_seq = re.sub(r"[UZOB]", "X", input_seq)
        pep_seq.append(input_seq)
        if not len(pep) > max_len:
            current_pep = []
            for aa in pep:
                current_pep.append(aa_dict[aa])
            pep_codes.append(torch.tensor(current_pep))
            #pep2label.append((current_pep,torch.tensor(int(label))))
        else:
            long_pep_counter += 1
    print("length > 63:", long_pep_counter)
    data = rnn_utils.pad_sequence(pep_codes, batch_first=True)

    return data, torch.tensor(labels), pep_seq



class MyDataSet(Data.Dataset):
    def __init__(self, data, label, seq):
        self.data = data
        self.label = label
        self.seq = seq

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.seq[idx]










if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('../prot_bert_bfd',
                                              do_lower_case=False)
    bert = BertModel.from_pretrained("../prot_bert_bfd")

    train_data, train_label, train_seq = genData("../train915.csv", 63)
    test_data, test_label, test_seq = genData("../test915.csv", 63)

    seq = train_seq + test_seq
#Unsupervised learning

    seq2vec = dict()
    for pep in seq:
        pep_str = "".join(pep)
        pep_text = tokenizer.tokenize(pep_str)
        pep_tokens = tokenizer.convert_tokens_to_ids(pep_text)
        tokens_tensor = torch.tensor([pep_tokens])
        with torch.no_grad():
            encoder_layers = bert(tokens_tensor)
            out_ten = torch.mean(encoder_layers.last_hidden_state, dim=1)
            out_ten = out_ten.numpy().tolist()
            seq2vec[pep] = out_ten

    with open('../seq2vec_CPP.emb', 'w') as g:
        g.write(json.dumps(seq2vec))


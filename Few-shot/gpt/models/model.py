import os
from typing import List, Union

import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from transformers import AutoConfig, AutoModel, GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoModel, GPTNeoForSequenceClassification

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_size=1024, layers=2, bidirectional=True, dropout=0, ag=False):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout,)

        self.linear = nn.Linear(hidden_size*2, 4 if ag else 2)


    def forward(self, padded_texts, lengths):
        texts_embedding = self.embedding(padded_texts)
        packed_inputs = pack_padded_sequence(texts_embedding, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed_inputs)
        forward_hidden = hn[-1, :, :]
        backward_hidden = hn[-2, :, :]
        concat_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        output = self.linear(concat_hidden)
        return output

    def predict(self,):
        pass

class BERT(nn.Module):
    def __init__(self, model_path: str, mlp_layer_num: int, class_num:int=6, hidden_dim:float=1024):
        super(BERT, self).__init__()
        self.mlp_layer_num = mlp_layer_num

        self.config = AutoConfig.from_pretrained('EleutherAI/gpt-neo-1.3B')
        self.hidden_size = self.config.hidden_size
        self.tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
        self.bert = GPTNeoForSequenceClassification.from_pretrained('EleutherAI/gpt-neo-1.3B', num_labels=6)
        self.bert.config.pad_token_id = self.bert.config.eos_token_id


    def forward(self, input_ids,attention_mask):
        bert_output = self.bert(input_ids,attention_mask=attention_mask)
        logits = bert_output.logits   # batch_size, 768
        feature = bert_output[0]

        return logits, feature


    


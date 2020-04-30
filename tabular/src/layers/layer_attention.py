"""class implements prototype layer"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.layers.layer_base import LayerBase
from src.classes.utils import apply_mask
import math

class LayerAttention(LayerBase):
    """implements prototype layer"""
    def __init__(self, input_dim, embedding_dim, output_dim, gpu=-1, equal_weight = False):
        super(LayerAttention, self).__init__(gpu)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.equal_weight = equal_weight
        self.gpu = gpu
        self.projection = nn.Linear(in_features = embedding_dim, out_features = input_dim, bias = False)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k) 
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
        
    def forward(self, rnn_output, mention_embedding = None, mask = None):
        # x should be shape: batch_size x max_seq_len x hidden_dim
        if self.equal_weight:
            # needs is_mention_mask
            batch_size = rnn_output.shape[0]
            max_seq_len = rnn_output.shape[1]
            is_mention_input = apply_mask(rnn_output, is_mention_mask, gpu=self.gpu)
            summed_mention_vectors = torch.sum(is_mention_input, dim=1)
            num_words_per_mention = torch.sum(is_mention_mask, dim=1).reshape(batch_size, 1).expand_as(summed_mention_vectors)
            avg_mention_vectors = summed_mention_vectors / num_words_per_mention
            return avg_mention_vectors
        
        else:
            # needs word_sequences_mask
            query = self.projection(mention_embedding) # shape should be : batch_size x max_seq_len x embedding_dim
            return self.attention(query = query, 
                                key = rnn_output, 
                                value = rnn_output, 
                                mask = mask) 







        
        

"""class implements prototype layer"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.layers.layer_base import LayerBase
from src.classes.utils import apply_mask
import math

class LayerPooler(LayerBase):
    """implements prototype layer"""
    def __init__(self, input_dim, gpu=-1, pooling_type = None):
        super(LayerPooler, self).__init__(gpu)
        assert input_dim % 2 == 0, "need an even number for the RNN hidden dim"
        self.input_dim = input_dim
        self.gpu = gpu
        self.pooling_type = pooling_type
        self.output_dim = input_dim
        # if self.pooling_type == "average":
        #     self.output_dim = input_dim        
        # elif self.pooling_type == "max":
        #     self.output_dim = input_dim
        # elif self.pooling_type == "final_embeddings":
        #     self.output_dim = input_dim
        if self.pooling_type == "attention":            
            self.mlp = nn.Sequential(
                nn.Linear(in_features = input_dim, out_features = input_dim, bias = True),
                nn.Tanh())
            self.query_vector = nn.Parameter(torch.rand(1,input_dim)*2-1) # match the tanh range -- equiv. to sigmoid right?

        if self.pooling_type == 'self-attention':
            self.self_attention_weight = nn.Sequential(
                nn.Linear(in_features = input_dim, out_features = input_dim, bias = True),
                nn.Tanh()
                )
        

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
        output = torch.matmul(p_attn, value).view(-1,self.output_dim)
        p_attn = p_attn.squeeze()
        return output, p_attn

        
    def forward(self, rnn_output, word_sequences_mask = None):
        # rnn_output should be shape: batch_size x max_seq_len x hidden_dim
        batch_size = rnn_output.shape[0]
        max_seq_len = rnn_output.shape[1]
        masked_rnn_output = apply_mask(rnn_output, word_sequences_mask, gpu=self.gpu)

        if self.pooling_type == "average":
            summed_rnn_output = torch.sum(masked_rnn_output, dim=1)
            num_words_per_mention = torch.sum(word_sequences_mask, dim=1).reshape(batch_size, 1).expand_as(summed_rnn_output)
            avg_mention_vectors = summed_rnn_output / num_words_per_mention
            return avg_mention_vectors
        
        elif self.pooling_type == "max":
            max_rnn_output, _ = torch.max(masked_rnn_output, dim = 1)
            return max_rnn_output

        elif self.pooling_type == "final_embeddings":
            # surely there was a better way to do this
            half_input_dim = int(self.input_dim / 2)
            last_forward_embedding_idx = torch.LongTensor([sum(seq)-1 for seq in word_sequences_mask]) # get positions of last word
            last_forward_mask = torch.zeros(batch_size,max_seq_len).scatter_(1, last_forward_embedding_idx.view(-1,1), 1) # make one hot
            last_forward_mask = last_forward_mask.cuda()
            last_forward_mask = last_forward_mask.unsqueeze(-1).expand_as(rnn_output[:,:,:half_input_dim]) # expand along embedding dimension
            last_forward_embeddings = torch.sum(rnn_output[:,:,:half_input_dim] * last_forward_mask,dim=1) 
            last_backward_embeddings = rnn_output[:,0, half_input_dim:]
            return torch.cat((last_forward_embeddings, last_backward_embeddings), dim = 1)

        elif self.pooling_type == "attention":
            mlp_output = self.mlp(rnn_output)
            context_embedding, p_attn = self.attention(query = self.query_vector, 
                                key = mlp_output, 
                                value = rnn_output, 
                                mask = word_sequences_mask) 
            return context_embedding

        # elif self.pooling_type == 'self-attention':
        #     Wrnn = self.self_attention_weight(rnn_output)
        #     context_embedding, p_attn = self.attention(query = rnn_output, 
        #                         key = Wrnn, 
        #                         value = rnn_output, 
        #                         mask = word_sequences_mask) 
        #     import ipdb; ipdb.set_trace()
        #     return context_embedding            









        
        

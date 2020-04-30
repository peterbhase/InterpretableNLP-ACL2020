"""Vanilla recurrent network model for sequences tagging."""
import torch
import torch.nn as nn
from src.models.tagger_base import TaggerBase
from src.layers.layer_word_embeddings import LayerWordEmbeddings
from src.layers.layer_bivanilla import LayerBiVanilla
from src.layers.layer_bilstm import LayerBiLSTM
from src.layers.layer_bigru import LayerBiGRU
from src.layers.layer_attention import LayerAttention


class TaggerAttentive(TaggerBase):
    """TaggerBiRNN is a Vanilla recurrent network model for sequences tagging."""
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, batch_size=1, rnn_hidden_dim=100,
                 freeze_word_embeddings=False, dropout_ratio=0.5, rnn_type='LSTM', gpu=-1,
                 latent_dim = None):
        super(TaggerAttentive, self).__init__(word_seq_indexer, tag_seq_indexer, gpu, batch_size)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.freeze_embeddings = freeze_word_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu
        self.word_embeddings_layer = LayerWordEmbeddings(word_seq_indexer, gpu, freeze_word_embeddings)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.latent_dim = latent_dim
        if rnn_type == 'GRU':
            self.birnn_layer = LayerBiGRU(input_dim=self.word_embeddings_layer.output_dim,
                                          hidden_dim=rnn_hidden_dim,
                                          gpu=gpu)
        elif rnn_type == 'LSTM':
            self.birnn_layer = LayerBiLSTM(input_dim=self.word_embeddings_layer.output_dim,
                                           hidden_dim=rnn_hidden_dim,
                                           gpu=gpu)
        elif rnn_type == 'Vanilla':
            self.birnn_layer = LayerBiVanilla(input_dim=self.word_embeddings_layer.output_dim+self.char_cnn_layer.output_dim,
                                           hidden_dim=rnn_hidden_dim,
                                           gpu=gpu)
        else:
            raise ValueError('Unknown rnn_type = %s, must be either "LSTM" or "GRU"')

        # equal weight attention 
        self.attention = LayerAttention(input_dim = self.birnn_layer.output_dim,
                                        embedding_dim = self.word_embeddings_layer.output_dim,
                                        output_dim = self.birnn_layer.output_dim,
                                        gpu = gpu)

        # dimension reduction
        if latent_dim is not None:
            self.dim_red = nn.Sequential(
                nn.Linear(in_features=self.attention.output_dim + self.word_embeddings_layer.output_dim, 
                          out_features=latent_dim),
                nn.Sigmoid()
            )            
            self.dim_red.apply(self.inititialize_random_projection)

            lin_layer_in = latent_dim
        else:
            lin_layer_in = self.attention.output_dim + self.word_embeddings_layer.output_dim

        # We add an additional class that corresponds to the zero-padded values not to be included to the loss function
        self.lin_layer = nn.Linear(in_features=lin_layer_in, out_features=class_num + 1)
        
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)
        self.nll_loss = nn.NLLLoss(ignore_index=0) # "0" target values actually are zero-padded parts of sequences

    def forward(self, word_sequences):
        # entity representation
        entity_embed = self.word_embeddings_layer(word_sequences)

        # context representation
        word_mask = self.get_mask_from_word_sequences(word_sequences = word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = z_word_embed # z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, word_mask) # shape: batch_size x max_seq_len x rnn_dim*2
        attention_out, _ = self.attention(rnn_output_h, mention_embedding = entity_embed, mask = word_mask)
        
        # concat entity + context representations
        concat_rep = torch.cat((entity_embed, attention_out), dim = -1)

        if self.latent_dim is not None:
            latent_h = self.dim_red(concat_rep)            
        else:
            latent_h = concat_rep # shape: batch_size x max_seq_len x latent_dim

        z_rnn_out = self.apply_mask(self.lin_layer(latent_h), word_mask) # shape: batch_size x max_seq_len x class_num + 1
        y = self.log_softmax_layer(z_rnn_out.permute(0, 2, 1))       
        
        return y # shape: batch_size x class_num + 1 x max_seq_len

    def initialize_from_pretrained(self, pretrained_path):
        print("Initializing model weights from model at %s" % pretrained_path)
        pretrained_model = torch.load(pretrained_path)
        state_dict = pretrained_model.state_dict()
        
        keys = [key for key in state_dict.keys()]
        for key in keys:
            if key.startswith('lin_layer'):
                del state_dict[key]

        self.load_state_dict(state_dict, strict = False)  


    def inititialize_random_projection(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0, std = 1 / (m.out_features ** 1/2) )

    def get_loss(self, word_sequences_train_batch, tags_train_batch):
        # if positions is None, the non-mention words are used in the loss (should be predicted to be tag 'non-mention')
        outputs_tensor_train_batch = self.forward(word_sequences = word_sequences_train_batch)
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tags_train_batch)
        loss = self.nll_loss(outputs_tensor_train_batch, targets_tensor_train_batch)
        return loss

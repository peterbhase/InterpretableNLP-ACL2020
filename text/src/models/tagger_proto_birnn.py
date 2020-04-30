"""Vanilla recurrent network model for sequences tagging."""
import torch
import torch.nn as nn
from src.models.tagger_base import TaggerBase
from src.layers.layer_word_embeddings import LayerWordEmbeddings
from src.layers.layer_bilstm import LayerBiLSTM
from src.layers.layer_proto import LayerProto
from src.layers.layer_pooler import LayerPooler
from src.classes.utils import *
import numpy as np
import math
import time
from sklearn.cluster import KMeans
from anchor.anchor.utils import perturb_sentence


class TaggerProtoBiRNN(TaggerBase):
    """TaggerBiRNN is a Vanilla recurrent network model for sequences tagging."""
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, batch_size=1, rnn_hidden_dim=100,
                 freeze_word_embeddings=False, dropout_ratio=0.5, rnn_type='GRU', gpu=-1,
                 num_prototypes_per_class=6, proto_dim = None, pretrained_path = None, max_pool_protos = False,
                 pooling_type = 'attention', similarity_epsilon = 1e-4, hadamard_importance = False,
                 similarity_function_name = 'log_inv_distance'):
        super(TaggerProtoBiRNN, self).__init__(word_seq_indexer, tag_seq_indexer, gpu, batch_size)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.freeze_embeddings = freeze_word_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu        
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes = class_num * num_prototypes_per_class
        self.proto_dim = proto_dim
        self.max_pool = max_pool_protos
        self.hadamard_importance = hadamard_importance

        # parameters
        self.prototypes_shape = (self.num_prototypes, self.proto_dim, 1) # the last dimension is 1 since the prototype vectors are used as a conv1d filter weight
        self.prototypes = nn.Parameter(torch.rand(self.prototypes_shape))

        # layers
        self.word_embeddings_layer = LayerWordEmbeddings(word_seq_indexer, gpu, freeze_word_embeddings)            

        self.birnn_layer = LayerBiLSTM(input_dim=self.word_embeddings_layer.output_dim,
                                       hidden_dim=rnn_hidden_dim,
                                       gpu=gpu)
        self.pooler = LayerPooler(input_dim = self.birnn_layer.output_dim, gpu=gpu, pooling_type = pooling_type)
   
        self.dim_red = nn.Sequential(
            nn.Linear(in_features=self.pooler.output_dim, out_features=proto_dim),
            nn.Sigmoid()
        )            
        
        self.proto_layer = LayerProto(input_dim=self.proto_dim, prototypes=self.prototypes, num_classes = class_num,
                    num_prototypes_per_class = num_prototypes_per_class, gpu=gpu, max_pool = max_pool_protos, 
                    similarity_epsilon = similarity_epsilon, hadamard_importance = hadamard_importance,
                    similarity_function_name = similarity_function_name)
        
        self.lin_layer = nn.Linear(in_features=self.proto_layer.output_dim, out_features=class_num, bias = False)
                
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)
        self.nll_loss = nn.NLLLoss() 

        # init weights and set grad reqs
        self._initialize_weights()
        self._set_grad_reqs()
        

    def forward(self, word_sequences):
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask) # shape: batch_size x max_seq_len x hidden_dim*2
        pooled_output_h = self.pooler(rnn_output_h, mask) # shape: batch_size x hidden_dim*2
        latent_h = self.dim_red(pooled_output_h) # shape: batch_size x proto_dim
        proto_output_h, distances = self.proto_layer(latent_h) # proto_output shape: batch_size x num_features
        z_out = self.lin_layer(proto_output_h) # shape: batch_size x class_num 
        y = self.log_softmax_layer(z_out) # shape: batch_size x  class_num
        return y

    def get_logprobs_and_distances(self, word_sequences):
        '''distances needed for the loss, but .forward used throughout this codebase. so this function appears in main.py'''
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask) # shape: batch_size x max_seq_len x hidden_dim*2
        pooled_output_h = self.pooler(rnn_output_h, mask) # shape: batch_size x hidden_dim*2
        latent_h = self.dim_red(pooled_output_h) # shape: batch_size x proto_dim
        proto_output_h, distances = self.proto_layer(latent_h) # proto_output shape: batch_size x num_features
        z_out = self.lin_layer(proto_output_h) # shape: batch_size x class_num 
        y = self.log_softmax_layer(z_out) # shape: batch_size x  class_num    
        return y, distances

    def get_logits(self, word_sequences):
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask) # shape: batch_size x max_seq_len x hidden_dim*2
        pooled_output_h = self.pooler(rnn_output_h, mask) # shape: batch_size x hidden_dim*2
        latent_h = self.dim_red(pooled_output_h) # shape: batch_size x proto_dim
        proto_output_h, distances = self.proto_layer(latent_h) # proto_output shape: batch_size x num_features
        z_out = self.lin_layer(proto_output_h) # shape: batch_size x class_num 
        return z_out

    def push_forward(self, word_sequences):
        '''used in push step'''
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask) # shape: batch_size x max_seq_len x hidden_dim*2
        pooled_output_h = self.pooler(rnn_output_h, mask) # shape: batch_size x hidden_dim*2
        latent_h = self.dim_red(pooled_output_h) # shape: batch_size x proto_dim
        proto_output_h, distances = self.proto_layer(latent_h) # proto_output shape: batch_size x num_features
        return latent_h, distances # latent shape: batch_size x proto_dim. distances shape: batch_size x num_prototypes

    def get_proto_output(self, word_sequences):
        '''used when gathering similarity scores, e.g. in self.explain_instance'''
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask) # shape: batch_size x max_seq_len x hidden_dim*2
        pooled_output_h = self.pooler(rnn_output_h, mask) # shape: batch_size x hidden_dim*2
        latent_h = self.dim_red(pooled_output_h) # shape: batch_size x proto_dim
        proto_output_h, distances = self.proto_layer(latent_h) # proto_output shape: batch_size x num_features
        return proto_output_h, distances

    def initialize_from_pretrained(self, pretrained_path):
        print("Initializing model weights from model at %s" % pretrained_path)
        pretrained_model = torch.load(pretrained_path)
        state_dict = pretrained_model.state_dict()
        
        # delete classifier weights
        del state_dict['lin_layer.weight']
        del state_dict['lin_layer.bias']
        # delete dim reduction weights if they don't match pretrained model's dimensionality
        if state_dict['dim_red.0.weight'].shape != self.dim_red[0].weight.shape:
            print("Dimension reduction matrix from pretrained model does not match this model's in shape!")
            del state_dict['dim_red.0.weight']
            del state_dict['dim_red.0.bias']
        
        self.load_state_dict(state_dict, strict = False)


    def get_lin_layer_l1(self):
        '''returns l1 penalty on off-prototype-class-connection weights in lin_layer'''
        if self.max_pool:
            identity = torch.eye(self.class_num).cuda()
            mask = 1 - identity
            masked_weight = self.lin_layer.weight * mask
        else:
            identity = torch.eye(self.class_num).cuda()
            repeated_identity = identity.unsqueeze(2).repeat(1,1,self.num_prototypes_per_class).\
                                    view(self.class_num, -1)
            mask = 1 - repeated_identity
            masked_weight = self.lin_layer.weight * mask        
        return masked_weight.norm(p=1)           

    def get_sep_loss(self, distances, word_sequences, targets_tensor):
        ''' 
        return the mean distance between each instance and its closest off-class prototype 
        distances should be shape: batch_size x num_prototypes
        need to mask the on-class prototype distances

        don't penalize distances once they're at least 4
        '''

        # mask the on-class prototype distances (set to 1e9)
        batch_size = targets_tensor.shape[0]
        for i in range(batch_size):
            target = targets_tensor[i].item()
            onclass_prototype_idx = np.arange(target * self.num_prototypes_per_class, (target+1) * self.num_prototypes_per_class)
            mask = torch.ones(batch_size, self.num_prototypes)
            mask[i, onclass_prototype_idx] = 1e9
            mask = mask.cuda()
            distances = distances * mask

        closest_distances, _ = torch.min(distances, dim = 1)
        max_to_penalize = 4 * torch.ones_like(closest_distances)
        stacked_distances_and_caps = torch.stack((closest_distances,max_to_penalize),dim=0)
        capped_closest_distances, _ = torch.min(stacked_distances_and_caps,dim=0)
        neg_avg_closest_distance = -torch.mean(capped_closest_distances)
        return neg_avg_closest_distance
      

    def get_loss(self, args, word_sequences_train_batch, tag_sequences_train_batch):
        # defunct, loss now calculated in main.py
        outputs_tensor_train_batch_one_hot, distances = self.get_logprobs_and_distances(word_sequences_train_batch)
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
        cross_entropy = self.nll_loss(outputs_tensor_train_batch_one_hot, targets_tensor_train_batch)

        sep_loss = self.get_sep_loss(distances, word_sequences_train_batch, targets_tensor_train_batch)
        lin_layer_reg = self.get_lin_layer_l1()

        loss = cross_entropy + 1/10 * lin_layer_reg + 1/100 * sep_loss

        return loss


    def freeze_unfreeze_parameters(self, epoch, args):
        # linear layer
        set_to = (epoch >= args.unfreeze_lin_layer)
        self.lin_layer.weight.requires_grad = set_to # there is no lin_layer.bias

        if args.hadamard_importance:
            assert args.unfreeze_lin_layer > 999, "Using hadamard weighting in proto_layer, should keep tagger.lin_layer frozen as an identity matrix (i.e. set to at least 999)"
        
        # every other layer
        set_to = (epoch >= args.unfreeze_feature_extractor)
        self.word_embeddings_layer.requires_grad = set_to
        for m in self.birnn_layer.rnn.modules():
            if hasattr(m,'weight'):
                m.requires_grad = set_to
            if hasattr(m,'bias'):
                m.requires_grad = set_to
        for m in self.pooler.modules():
            if hasattr(m,'weight'):
                m.requires_grad = set_to
            if hasattr(m,'bias'):
                m.requires_grad = set_to        
        for m in self.dim_red.modules():
            if hasattr(m,'weight'):
                m.requires_grad = set_to
            if hasattr(m,'bias'):
                m.requires_grad = set_to


    def _initialize_weights(self):
        
        def _initialize_random_projection(m):
            if type(m) == nn.Linear:
                torch.nn.init.normal_(m.weight, mean=0, std = 1 / (m.out_features ** 1/2) )  

        def _initialize_lin_layer(self):               
            if self.max_pool:
                identity = torch.eye(self.class_num)
                self.lin_layer.weight.data.copy_(identity)
            else: 
                identity = torch.eye(self.class_num)
                repeated_identity = identity.unsqueeze(2).repeat(1,1,self.num_prototypes_per_class).\
                                        view(self.class_num, -1)                                
                self.lin_layer.weight.data.copy_(repeated_identity)

        _initialize_lin_layer(self)        
        self.dim_red.apply(_initialize_random_projection) 


    def _set_grad_reqs(self):
        self.word_embeddings_layer.requires_grad = False
        for m in self.modules():        
            if hasattr(m,'weight'):
                m.requires_grad = False
            if hasattr(m,'bias'):
                m.requires_grad = False
        self.prototypes.requires_grad = True


    def initialize_prototypes_empirical(self, word_sequences, tag_sequences, batch_size = 10):
        '''initialize prototypes for each class by k-means on that classes latent representations (class given by labels)'''
        print("Initializing prototypes empirically")
        self.eval()
        class2vecs = dict()
        class_id_list = [i for i in range(0, self.class_num)]        

        for i in class_id_list:
            class2vecs[i] = []
        
        # for each batch, get vecs and ids, add vecs to class2vecs based on ids
        batch_num = math.floor(len(word_sequences) / batch_size)
        if len(word_sequences) > 0 and len(word_sequences) < batch_size:
            batch_num = 1

        start_time = time.time()
        
        for n in range(batch_num):
            i = n*batch_size
            if n < batch_num - 1:
                j = (n + 1)*batch_size
            else:
                j = len(word_sequences)
    
            batch = word_sequences[i:j]
            targets = self.tag_seq_indexer.items2idx(tag_sequences[i:j])
            mask = self.get_mask_from_word_sequences(batch)            
            latents, distances = self.push_forward(batch) # latents: batch_size x proto_dim
            for k in range(len(batch)):                
                class_id = targets[k]
                latent_vec = latents[k, :].detach().cpu().numpy()
                class2vecs[class_id].append(latent_vec)

        print('gathering latent vecs took %.1f seconds' % (time.time() - start_time))

        # there are a variety of ways to move the data from kmeans.cluster_centers_ to self.prototypes, of course
        # but copying with splices directly to self.prototypes_[idx,:,:] was silently failing, so we preallocate and .copy_ all at once
        new_prototypes = torch.zeros(0, self.proto_dim, 1)        
        for i in class_id_list:
            class_data = np.array(class2vecs[i])
            proto_idx = [idx for idx in range(self.num_prototypes_per_class*(i-1),\
                                                self.num_prototypes_per_class*i)]
            kmeans = KMeans(n_clusters = self.num_prototypes_per_class)
            kmeans.fit(class_data)

            centers = torch.Tensor(kmeans.cluster_centers_).view(self.num_prototypes_per_class, self.proto_dim, 1)
            new_prototypes = torch.cat((new_prototypes, centers), 0)        

        self.prototypes.data.copy_(new_prototypes)


    def explain_instance(self, word_sequence, saliency_type = 'directional', neighbors_obj = None, language_model = None, tokenizer = None,
                            counterfactual_method = 'unk', decision_boundary = False):
        # word_sequence should be text with tokens separated by spaces or list of tokens
        self.eval()

        # adjust formatting for string inputs
        if type(word_sequence) is str:
            word_sequence = word_sequence.split()

        # local vars
        classes = self.tag_seq_indexer.idx2items([0,1])
        text = ' '.join(word_sequence)
        m = 10 # multiple for scaling the logits. easier to read values on the integer rather than decimal scale

        # original prediction and logits
        orig_logits = self.get_logits([word_sequence]).squeeze().detach().cpu()
        pred = np.argmax(orig_logits)
        predicted_tag = classes[pred]
        opposite_tag = classes[1-pred]
        
        # local vars
        prototype_dict = self.prototype_dict
        n_protos_per_class = self.num_prototypes_per_class
        max_activation = self.proto_layer.similarity_score(torch.zeros(1)).item()

        # proto output 
        proto_output, distances = self.get_proto_output([word_sequence])
        proto_output = proto_output.detach().cpu().squeeze()
        distances = distances.detach().cpu().squeeze()
        prototype_id = np.argmin(distances).item()
        activated_prototype = prototype_dict[prototype_id]
        similarity_score = self.proto_layer.similarity_score(torch.min(distances)).item()

        # get signed evidence measure
        signed_evidence_str = '+%.2f' % (m*similarity_score) if activated_prototype.tag == 'pos' else '-%.2f' % (m*similarity_score)

        # proto and input salience 
        orig_explanation = self.prototype_saliency_map(word_sequence, 
                                                    saliency_type = saliency_type, 
                                                    prototype_id = prototype_id, 
                                                    neighbors_obj = neighbors_obj,
                                                    language_model = language_model,
                                                    tokenizer = tokenizer,
                                                    counterfactual_method = counterfactual_method)


        # if not going to look at a perturbation of the opposite predicted class, then go ahead and assemble the explanation
        explanation = "Most activated prototype (label: %s) | evidence: %s \n" % (predicted_tag, signed_evidence_str) + \
                activated_prototype.to_str() + '\n----\n' + \
                "Informative words in input:\n" + \
                orig_explanation
      
        return explanation



    def prototype_saliency_map(self, word_sequence, prototype_id, saliency_type = 'directional', num_perturbations = 1000, neighbors_obj = None, 
                                counterfactual_method = 'unk', language_model = None, tokenizer = None):
        '''
        get feature importance scores for prototype model using word omission with counterfactual_method approach
        '''

        # prep tagger
        self.zero_grad()
        self.eval()
        
        # local variables
        n_protos_per_class = self.num_prototypes_per_class
        class_id = prototype_id // n_protos_per_class
        start = time.time()
        m = 10 # multiple for word importance values and logits. easier to read values on the integer scale than decimal.
        
        # should clean this up with arguments, since this is computed in .explain_instance. for now, recompute
        predicted_tag = self.predict_tags_from_words([word_sequence], constrain_to_classes = None, quiet = True)[0]

        # get valid_idx for words to sample in expected_word method
        if counterfactual_method == 'expected_word':
            all_vocab = [force_ascii(tokenizer._convert_id_to_token(i)) for i in range(tokenizer.vocab_size)]
            valid_vocab = [word for word in all_vocab if word in self.word_seq_indexer.item2idx_dict]
            valid_idx = np.argwhere([word in self.word_seq_indexer.item2idx_dict for word in all_vocab]).reshape(-1)

        # forward pass
        logits = self.get_logits([word_sequence])
        selected_logit = torch.max(logits) if class_id is None else logits[0,class_id]  
        selected_logit = selected_logit.detach().cpu()

        # need class ids to possibly negate the importance values later
        neg_class_id = self.tag_seq_indexer.item2idx_dict['neg']
        pred_class_id = torch.argmax(logits.view(-1)).item()
        explain_class_id = pred_class_id if class_id is None else class_id

        # get avg. difference in selected_logit and the class logit obtained from perturbed inputs (perturbed at a specific word)
        logit_differences = np.zeros(len(word_sequence))

        for slot_id in range(len(word_sequence)):

            # if 'unk' or 'neighbors', fill the slot with either the <unk> vector or neighboring words in embedding space (according to counterfactual_method)
            if counterfactual_method != 'expected_word':
                counterfactual_sequences = replace_word(word_sequence, slot_id, neighbors_obj, 
                                                        tagger_word_dict = self.word_seq_indexer.item2idx_dict, method = counterfactual_method)
                counterfactual_logits = self.get_logits(counterfactual_sequences).detach().cpu()
                mean_logit = torch.mean(counterfactual_logits[:,explain_class_id])
                
                logit_differences[slot_id] = selected_logit - mean_logit

            # if 'expected_word', find the expected logit over p(x_i | x_{-i})
            elif counterfactual_method == 'expected_word':
                expected_logit = expected_score(word_sequence = word_sequence, 
                                            mask_position = slot_id, 
                                            class_id = explain_class_id, 
                                            tagger = self, 
                                            language_model = language_model, 
                                            tokenizer = tokenizer, 
                                            vocab = valid_vocab, 
                                            valid_idx = valid_idx)
                logit_differences[slot_id] = selected_logit - expected_logit

        # set importance metric
        importance_metric = logit_differences

        # quick fix so that saliency maps are consistently directional between classes. positive values always positive sentiment, negative numbers always negative sentiment
        neg_class_id = self.tag_seq_indexer.item2idx_dict['neg']
        explain_class_id = self.tag_seq_indexer.item2idx_dict[predicted_tag] if class_id is None else class_id
        if explain_class_id == neg_class_id and (saliency_type == 'directional' or saliency_type == 'counterfactual'):
            importance_metric = -importance_metric   

        # scale by 10 for readability
        importance_metric = m * importance_metric 
                
        # get highlighted words
        importance_str = saliency_list_values(word_sequence, importance_metric, saliency_type, print_flat = False)

        # print("Prototype explanation took %.2f seconds" % (time.time() - start))

        return importance_str


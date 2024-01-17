"""Vanilla recurrent network model for sequences tagging."""
import torch
import torch.nn as nn
from src.models.tagger_base import TaggerBase
from src.layers.layer_proto import LayerProto
from src.classes.utils import *
from src.classes.prototype import Prototype
import heapq
import numpy as np
import math
import time
from sklearn.cluster import KMeans


class TaggerProtoMLP(TaggerBase):
    """TaggerBiRNN is a Vanilla recurrent network model for sequences tagging."""
    def __init__(self, data_encoder, tag_seq_indexer, class_num, proto_dim, batch_size=1, input_dim = 115,
                 dropout_ratio=0.2, gpu=-1, hidden_dim = 50,
                 num_prototypes_per_class=6, pretrained_path = None, max_pool_protos = False,
                 similarity_epsilon = 1e-4, hadamard_importance = False,
                 similarity_function_name = 'gaussian', dim_red_bool = False):
        super(TaggerProtoMLP, self).__init__(data_encoder, tag_seq_indexer, gpu, batch_size)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.input_dim = input_dim
        self.dropout_ratio = dropout_ratio
        self.gpu = gpu        
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.num_prototypes_per_class = num_prototypes_per_class
        self.num_prototypes = class_num * num_prototypes_per_class
        self.proto_dim = proto_dim
        self.max_pool = max_pool_protos
        self.hadamard_importance = hadamard_importance
        self.dim_red_bool = dim_red_bool

        # prototype parameters
        self.prototypes_shape = (self.num_prototypes, self.proto_dim, 1) # the last dimension is 1 since the prototype vectors are used as a conv1d filter weight
        self.prototypes = nn.Parameter(torch.rand(self.prototypes_shape))

        # feature extractor
        hidden_sizes = [hidden_dim, hidden_dim]
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(in_features=hidden_sizes[0], out_features=hidden_sizes[1]),
            nn.ReLU()
        )
         
        # dimension reduction
        self.dim_red = nn.Sequential(
            nn.Linear(in_features=hidden_sizes[1], out_features=proto_dim),
            nn.Sigmoid()
        )            
        
        # prototype layer
        self.proto_layer = LayerProto(input_dim=self.proto_dim, prototypes=self.prototypes, num_classes = class_num,
                    num_prototypes_per_class = num_prototypes_per_class, gpu=gpu, max_pool = max_pool_protos, 
                    similarity_epsilon = similarity_epsilon, hadamard_importance = hadamard_importance,
                    similarity_function_name = similarity_function_name)
        
        # final layer (set to identity matrix in this code iteration)
        self.lin_layer = nn.Linear(in_features=self.proto_layer.output_dim, out_features=class_num, bias = False)
            
        # quantities for loss
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)
        self.nll_loss = nn.NLLLoss() 

        # init weights and set grad reqs
        self._initialize_weights()
        self._set_grad_reqs()
        

    def forward(self, X_dense):
        X_sparse = self.dense2sparse(X_dense)
        hidden = self.feature_extractor(X_sparse)
        latents = self.dim_red(hidden)
        proto_output_h, distances = self.proto_layer(latents) # proto_output shape: batch_size x num_features
        z_out = self.lin_layer(proto_output_h) # shape: batch_size x class_num 
        y = self.log_softmax_layer(z_out) # shape: batch_size x  class_num
        return y

    def get_logprobs_and_distances(self, X_dense):
        '''distances needed for the loss, but .forward used throughout this codebase. so this function appears in main.py'''
        X_sparse = self.dense2sparse(X_dense)
        hidden = self.feature_extractor(X_sparse)
        latents = self.dim_red(hidden) 
        proto_output_h, distances = self.proto_layer(latents) # proto_output shape: batch_size x num_features
        z_out = self.lin_layer(proto_output_h) # shape: batch_size x class_num 
        y = self.log_softmax_layer(z_out) # shape: batch_size x  class_num    
        return y, distances

    def get_logits(self, X_dense):
        X_sparse = self.dense2sparse(X_dense)
        hidden = self.feature_extractor(X_sparse)
        latents = self.dim_red(hidden) 
        proto_output_h, distances = self.proto_layer(latents) # proto_output shape: batch_size x num_features
        z_out = self.lin_layer(proto_output_h) # shape: batch_size x class_num 
        return z_out

    def push_forward(self, X_dense):
        '''used in push step'''
        X_sparse = self.dense2sparse(X_dense)
        hidden = self.feature_extractor(X_sparse)
        latents = self.dim_red(hidden) 
        proto_output_h, distances = self.proto_layer(latents) # proto_output shape: batch_size x num_features
        return latents, distances # latent shape: batch_size x proto_dim. distances shape: batch_size x num_prototypes

    def get_proto_output(self, X_dense):
        '''used when gathering similarity scores, e.g. in self.explain_instance'''
        X_sparse = self.dense2sparse(X_dense)
        hidden = self.feature_extractor(X_sparse)
        latents = self.dim_red(hidden) 
        proto_output_h, distances = self.proto_layer(latents) # proto_output shape: batch_size x num_features
        return proto_output_h, distances

    def initialize_from_pretrained(self, pretrained_path):
        print("Initializing model weights from model at %s" % pretrained_path)
        pretrained_model = torch.load(pretrained_path)
        state_dict = pretrained_model.state_dict()
        
        # delete classifier weights
        del state_dict['lin_layer.weight']
        del state_dict['lin_layer.bias']
        # delete dim reduction weights if they don't match pretrained model's dimensionality
        if self.dim_red_bool:
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

        if args.hadamard_importance:
            assert args.unfreeze_lin_layer > 999, "Using hadamard weighting in proto_layer, should keep tagger.lin_layer frozen as an identity matrix (i.e. set to at least 999)"
        
        # every layer in the network
        set_to = (epoch >= args.unfreeze_feature_extractor)
        for m in self.modules():
            if hasattr(m,'weight'):
                m.requires_grad = set_to
            if hasattr(m,'bias'):
                m.requires_grad = set_to

        # linear layer
        set_to = (epoch >= args.unfreeze_lin_layer)
        self.lin_layer.weight.requires_grad = set_to # there is no lin_layer.bias


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

        def _initalize_frozen_dim_red(m):
            if type(m) == nn.Linear: # i think needed because the dimension reduction has a bias?
                identity = torch.eye(self.proto_dim)
                m.weight.data.copy_(identity)

        # initialize
        _initialize_lin_layer(self)
        if self.dim_red_bool:
            self.dim_red.apply(_initialize_random_projection) 
        else:
            self.dim_red.apply(_initalize_frozen_dim_red) 


    def _set_grad_reqs(self):
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



    def push(self, args, dataset_obj, word_sequences, tag_sequences, batch_size = 10, save_prototype_files = False,
                saliency_type = 'counterfactual'):
        '''
        push algorithm for prototype-based classifiers
        saves prototypes in two files: .txt with all protos in it and an .hdf5 dict of Prototypes
        see src.classes.prototype for Prototype

        alg: for each batch/instance, check if that distance is < its proto's min

        conditions for prototypes:
            prototypes must correspond to instances of their pre-assigned class
            no two prototypes can point to the same instance (this is decided greedily)
            - keep heap of neighbors for each prototype, then within each class, greedily assign prototypes to neighbors

        '''

        
        self.eval()
        print('Starting push...')
        start = time.time()
        n_protos = self.num_prototypes
        n_protos_per_class = self.num_prototypes_per_class
        new_prototypes = np.zeros((n_protos,self.proto_dim))
        proto_idx_to_Prototype = dict() # maps from proto id : Prototype        
        heaps = [] # one heap per prototype
        for _ in range(n_protos):
            heaps.append([])

        proto_idx_to_Prototype = dict() # maps from proto id : Prototype
        lin_layer_weight = self.lin_layer.weight.data.detach().cpu() # for saving importance_score
        
        n_batches = math.floor(len(word_sequences) / batch_size)
        
        for n in range(n_batches):
            i = n*batch_size
            if n < n_batches - 1:
                j = (n + 1)*batch_size
            else:
                j = len(word_sequences)        
        
            batch = word_sequences[i:j]
            targets = self.tag_seq_indexer.items2idx(tag_sequences[i:j])      
            latents, distances = self.push_forward(batch) # latent shape: batch_size x proto_dim. distances shape: batch_size x n_protos

            latents = latents.detach().cpu().numpy()
            distances = distances.detach().cpu().numpy()

            for k in range(len(batch)):                       
                class_id = targets[k]      

                for m in range(n_protos):                                                
                    proto_class_id = m // self.num_prototypes_per_class                        
                    
                    # only consider if instance belongs to prototype's class            
                    if class_id == proto_class_id:                      
                        context = batch[k]
                        tag = self.tag_seq_indexer.idx2item_dict[class_id]       
                        distance = distances[k, m]
                        latent_vec = latents[k, :]
                        importance_score = lin_layer_weight[class_id, m].item() if not self.max_pool \
                                else lin_layer_weight[class_id, class_id].item() # lin_layer shape varies if max_pool or not
                        if self.proto_layer.hadamard_importance:
                            importance_score = self.proto_layer.importance_weights[m]
                        
                        latent = Prototype(dataset_obj = dataset_obj,
                                          prototype_id = m,
                                          context = context, 
                                          class_id = class_id,
                                          tag = tag,
                                          global_idx = (n,k),                                                                       
                                          batch_size = batch_size,
                                          importance_score = importance_score,
                                          distance=distance)
                        latent.vector = latent_vec                   

                        if len(heaps[m]) < n_protos_per_class: # simply add if there aren't n_protos_per_class nearest yet
                            heapq.heappush(heaps[m], latent)
                        else:
                            heapq.heappushpop(heaps[m], latent)

        # greedily assign prototypes to their nearest neighbors, removing assigned neighbors
        consumed_neighbor_idx = set()
        push_distances = np.zeros(n_protos)
        for m, heap in enumerate(heaps):            
            # sort heap to go closest to furthest
            heap.sort()
            heap.reverse()
            for neighbor in heap:
                unique_id = neighbor.global_idx[0] * neighbor.batch_size + neighbor.global_idx[1]
                if unique_id not in consumed_neighbor_idx:
                    proto_idx_to_Prototype[m] = neighbor
                    new_prototypes[m] = neighbor.vector
                    consumed_neighbor_idx.add(unique_id)
                    push_distances[m] = neighbor.distance
                    break
            # throw error if nothing was assigned
            assert not all(new_prototypes[m] == 0), "New prototype vector is still all zeros"


        # update prototype vectors
        new_prototypes = torch.Tensor(new_prototypes).view(n_protos, self.proto_dim, 1).cuda()
        self.prototypes.data.copy_(new_prototypes)

        # attach prototype objects to self
        self.prototype_dict = proto_idx_to_Prototype

        end = time.time()
        print('Push took %.1f seconds' % (end-start))
        quantiles = np.quantile(push_distances,(.1,.5,.9))
        print("\tPush distance quantiles: 10%%: %.2f    50%%: %.2f   90%%: %.2f" % (quantiles[0],quantiles[1],quantiles[2]))

        if save_prototype_files:
            # now write a .txt file with all the prototypes printed
            fname = os.path.join(args.save_dir, '%s-prototypes.txt' % args.save_name)
            with open(fname, 'w') as f:
                f.write('---------------------------------------------------------\n')
                f.write('\nFile contains prototypes for model at %s.hdf5 \n' % args.save_name)
                f.write('\nLevel of saliency/importance attributed to each word is denoted by the number of ` marks it is wrapped in\n')
                f.write('\n---------------------------------------------------------\n\n')

                for m in range(n_protos):
                    prototype = proto_idx_to_Prototype[m]
                    class_id = prototype.class_id
                    class_name = self.tag_seq_indexer.idx2item_dict[class_id]
                    importance_score = prototype.importance_score
                    global_idx = prototype.global_idx
                    f.write('Prototype %d, class: %d, tag: %s, importance score: %.2f, global id: (%d,%d) \n' % (
                                                                    m, 
                                                                    class_id, 
                                                                    class_name,                                                                
                                                                    importance_score,
                                                                    global_idx[0], global_idx[1])
                    )
                    self_activation_str = prototype.to_str()
                    f.write('%s \n\n\n' % self_activation_str)

            # and save the .hdf5 with the prototype dict
            save_path = os.path.join(args.save_dir, '%s-prototypes.hdf5' % args.save_name)
            torch.save(proto_idx_to_Prototype, save_path)

        return        




    def explain_instance(self, word_sequence, counterfactual_method = 'conditional_expected_value'):
        # word_sequence should be text with tokens separated by spaces or list of tokens
        self.eval()

        # import ipdb; ipdb.set_trace()
        word_sequence = force_array_2d(word_sequence)

        # local vars
        classes = self.tag_seq_indexer.idx2items([0,1])
        m = 10 # scaling factor for logits. easier to read values on the integer rather than decimal scale

        # original prediction and logits
        orig_logits = self.get_logits(word_sequence).squeeze().detach().cpu()
        orig_logits_str = "Evidence for classes: %s - %.2f | %.2f - %s \n\n" % (classes[0], m*orig_logits[0], m*orig_logits[1], classes[1])

        # get tags
        pred = np.argmax(orig_logits)
        predicted_tag = classes[pred]

        # proto output 
        proto_output, distances = self.get_proto_output(word_sequence)
        proto_output = proto_output.detach().cpu().squeeze()
        distances = distances.detach().cpu().squeeze()
        prototype_id = np.argmin(distances).item()
        activated_prototype = self.prototype_dict[prototype_id]
        similarity_score = self.proto_layer.similarity_score(torch.min(distances)).item()

        # get signed evidence measure
        signed_evidence_str = '+%.2f' % (m*similarity_score) if activated_prototype.tag == 'above $50K' else '-%.2f' % (m*similarity_score)

        orig_explanation = self.saliency_map(word_sequence, 
                                            counterfactual_method = counterfactual_method,
                                            scaling_factor = m)

        # make explanation
        explanation_line_1 = "Most activated prototype (label: %s) | evidence: %s \n" % (predicted_tag, signed_evidence_str)
        explanation_prototype_and_input = two_data_points_human_readable(self.data_encoder, activated_prototype.context, word_sequence, header = "")
        explanation_input_importance = orig_explanation

        explanation = explanation_line_1 + \
            'Variable | Prototype  --- Original Input\n' + \
            '----\n' + \
            explanation_prototype_and_input + \
            '----\n' + \
            "Informative values in input:\n" + \
            explanation_input_importance


        # write explanation
        # explanation = "Most activated prototype (%s) | similarity_score : %.2f \n" % (predicted_tag, m*similarity_score) + \
        #         activated_prototype.to_str() + \
        #         '\n----\n' + \
        #         orig_proto_salience + '\n' + \
        #         "Important values in input:\n" + \
        #         data_row_to_str(self.data_encoder, word_sequence) + \
        #         '\n----\n' + \
        #         orig_explanation

        return explanation


    def give_similar_case(self, word_sequence):
        # word_sequence should be text with tokens separated by spaces or list of tokens
        self.eval()

        word_sequence = force_array_2d(word_sequence)

        # local vars
        classes = self.tag_seq_indexer.idx2items([0,1])

        # original prediction and logits
        orig_logits = self.get_logits(word_sequence).squeeze().detach().cpu()

        # get tags
        pred = np.argmax(orig_logits)
        predicted_tag = classes[pred]

        # proto output 
        proto_output, distances = self.get_proto_output(word_sequence)
        proto_output = proto_output.detach().cpu().squeeze()
        distances = distances.detach().cpu().squeeze()
        prototype_id = np.argmin(distances).item()
        activated_prototype = self.prototype_dict[prototype_id]
        similarity_score = self.proto_layer.similarity_score(torch.min(distances)).item()

        # make explanation
        explanation_prototype_and_input = two_data_points_human_readable(self.data_encoder, activated_prototype.context, word_sequence, header = "")

        explanation = '\n' + explanation_prototype_and_input

        return explanation

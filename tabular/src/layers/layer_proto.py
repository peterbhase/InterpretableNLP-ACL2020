"""class implements prototype layer"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.layers.layer_base import LayerBase

class LayerProto(LayerBase):
    """implements prototype layer"""
    def __init__(self, input_dim, prototypes, num_classes, num_prototypes_per_class, gpu, max_pool, similarity_epsilon,
                        hadamard_importance, similarity_function_name):
        '''
        if max_pool, output is within-class max-pool over prototype activations (or prototype_activations if hadamard_importance is true)
        if hadamard_importance, the prototype activations are hadamard-product-ed importance scores
        '''
        super(LayerProto, self).__init__(gpu)
        self.input_dim = input_dim
        num_prototypes = prototypes.shape[0]
        self.output_dim = num_classes if max_pool else num_prototypes
        self.prototype_vectors = prototypes 
        self.ones = nn.Parameter(torch.ones(prototypes.shape), requires_grad=False)   
        self.num_prototypes_per_class = num_prototypes_per_class
        self.max_pool = max_pool
        self.max_pool_f = nn.MaxPool1d(kernel_size = num_prototypes_per_class, 
                                       stride = num_prototypes_per_class)
        self.epsilon = similarity_epsilon
        self.hadamard_importance = hadamard_importance
        self.similarity_function_name = similarity_function_name
        if hadamard_importance:
            self.importance_weights = nn.Parameter(torch.ones(num_prototypes), requires_grad = True)
        # if similarity_function_name == 'gaussian':
        #     self.scale = 


    def forward(self, x):
        distances = self.prototype_distances(x)
        distances = distances.view(-1, self.prototype_vectors.shape[0])
        similarity_scores = self.similarity_score(distances)         
        # if max_pooling: reduce shape to batch_size x num_classes. but don't touch distances (remain batch_size x num_prototypes)
        if self.max_pool: 
            if self.hadamard_importance:
                similarity_scores = similarity_scores * self.importance_weights.expand_as(similarity_scores)
            n_features = similarity_scores.shape[-1]
            similarity_scores = similarity_scores.view(-1,1,n_features) # add channel dim for the conv
            similarity_scores = self.max_pool_f(similarity_scores)
            similarity_scores = similarity_scores.view(-1,self.output_dim)

        return similarity_scores, distances

    def similarity_score(self, distances):
        
        if self.similarity_function_name == "log_inv_distance":
            return torch.log(1 + (1 / (distances + self.epsilon)))
        
        elif self.similarity_function_name == "gaussian":
            return torch.exp(-distances)


    def prototype_distances(self, x):
        '''x is the blackbox output. squared l2 distances computed by convolution'''
        '''x should be shape: batch_size x proto_dim'''
        x = x.view(-1,x.shape[-1],1) # reshape to batch_size x proto_dim x 1
        x2 = x ** 2
        x2_patch_sum = nn.functional.conv1d(input=x2,weight=self.ones)
        p2 = self.prototype_vectors ** 2
        p2 = torch.sum(p2, dim = (1,2))
        p2_reshape = p2.view(-1,1)
        xp = nn.functional.conv1d(input=x, weight=self.prototype_vectors)        
        distances = nn.functional.relu(x2_patch_sum - 2*xp + p2_reshape)
        distances = distances.view(-1,self.output_dim)
        return distances # shape: batch_size x num_protos



        
        

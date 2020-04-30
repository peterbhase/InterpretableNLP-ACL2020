"""abstract base class for all types of taggers"""
import math
import torch
import torch.nn as nn
import os
import copy
import time
from src.classes.prototype import Prototype
import numpy as np
import heapq


class TaggerBase(nn.Module):
    """TaggerBase is an abstract class for tagger models. It implements the tagging functionality for
    different types of inputs (sequences of tokens, sequences of integer indices, tensors). Auxiliary class
    SequencesIndexer is used for input and output data formats conversions. Abstract method `forward` is used in order
    to make these predictions, it have to be implemented in ancestors."""
    def __init__(self,  word_seq_indexer, tag_seq_indexer, gpu, batch_size):
        super(TaggerBase, self).__init__()
        self.word_seq_indexer = word_seq_indexer
        self.tag_seq_indexer = tag_seq_indexer
        self.gpu = gpu
        self.batch_size = batch_size

    def tensor_ensure_gpu(self, tensor):
        if self.gpu >= 0:
            return tensor.cuda(device=self.gpu)
        else:
            return tensor

    def self_ensure_gpu(self):
        if self.gpu >= 0:
            self.cuda(device=self.gpu)
        else:
            self.cpu()

    def save_tagger(self, checkpoint_fn):
        self.cpu()
        torch.save(self, checkpoint_fn)
        self.self_ensure_gpu()

    def forward(self, *input):
        pass

    def predict_idx_from_words(self, word_sequences, constrain_to_classes = None):
        self.eval()
        # constrain predictions to just these classes, if not None
        if constrain_to_classes is not None:
            valid_classes = self.tag_seq_indexer.items2idx(constrain_to_classes)
        # if feeding in just a str, put it in a list
        if type(word_sequences) is str:
            word_sequences = [word_sequences]
        # split strings in word_sequences if there are strings in word_sequences
        if any([type(words) is str for words in word_sequences]):
            word_sequences = [words.split() for words in word_sequences]

        outputs_tensor = self.forward(word_sequences) # batch_size x num_class
        predicted_idx = []
        
        for k in range(len(word_sequences)):
            if constrain_to_classes is None:
                curr_output = outputs_tensor[k, :]
                prediction = curr_output.argmax(dim=0).item()                
            else:
                curr_output = outputs_tensor[k, valid_classes]
                max_no = curr_output.argmax(dim=0).item()
                prediction = valid_classes[max_no]
            predicted_idx.append(prediction)        
        return predicted_idx

    
    def predict_idx_from_words_np(self, word_sequences, constrain_to_classes = None):
        self.eval()
        # constrain predictions to just these classes, if not None
        if constrain_to_classes is not None:
            valid_classes = self.tag_seq_indexer.items2idx(constrain_to_classes)
        # if feeding in just a str, put it in a list
        if type(word_sequences) is str:
            word_sequences = [word_sequences]
        # split strings in word_sequences if there are strings in word_sequences
        if any([type(words) is str for words in word_sequences]):
            word_sequences = [words.split() for words in word_sequences]

        outputs_tensor = self.forward(word_sequences) # batch_size x num_class
        predicted_idx = []
        
        for k in range(len(word_sequences)):
            if constrain_to_classes is None:
                curr_output = outputs_tensor[k, :]
                prediction = curr_output.argmax(dim=0).item()                
            else:
                curr_output = outputs_tensor[k, valid_classes]
                max_no = curr_output.argmax(dim=0).item()
                prediction = valid_classes[max_no]
            predicted_idx.append(prediction)        
        return np.array(predicted_idx)


    def predict_probs_from_words(self, word_sequences, constrain_to_classes = None):
        # returns numpy array of predicted probs, for use in LIME
        self.eval()
        # constrain predictions to just these classes, if not None
        if constrain_to_classes is not None:
            valid_classes = self.tag_seq_indexer.items2idx(constrain_to_classes)
        # if feeding in just a str, put it in a list
        if type(word_sequences) is str:
            word_sequences = [word_sequences]
        # split strings in word_sequences if there are strings in word_sequences
        if any([type(words) is str for words in word_sequences]):
            word_sequences = [words.split() for words in word_sequences]

        outputs_tensor = self.forward(word_sequences) # batch_size x num_class
        predicted_probs = []

        for k in range(len(word_sequences)):
            if constrain_to_classes is None:
                curr_output = outputs_tensor[k, :]
                prediction = curr_output.argmax(dim=0).item()                
            else:
                curr_output = outputs_tensor[k, valid_classes]
                max_no = curr_output.argmax(dim=0).item()
                prediction = valid_classes[max_no]
            
            probs = torch.nn.functional.softmax(curr_output, dim=0).detach().cpu().numpy()
            predicted_probs.append(probs)
        return np.array(predicted_probs)            


    def predict_tags_from_words(self, word_sequences, batch_size=-1, quiet = False,
                                constrain_to_classes = None):
        '''
        they use .extend because curr_output_tag_sequences is a list of lists, so what gets appended is just the list of tags
        word_sequences must be list of lists of tokens
        '''
        if batch_size == -1:
            batch_size = self.batch_size
        if not quiet:
            print('\n')
        batch_num = math.floor(len(word_sequences) / batch_size)
        if len(word_sequences) > 0 and len(word_sequences) < batch_size:
            batch_num = 1
        output_tag_sequences = list()
        for n in range(batch_num):
            i = n*batch_size
            if n < batch_num - 1:
                j = (n + 1)*batch_size
            else:
                j = len(word_sequences)
            curr_output_idx = self.predict_idx_from_words(word_sequences[i:j], constrain_to_classes = constrain_to_classes)
            curr_output_tag_sequences = self.tag_seq_indexer.idx2items(curr_output_idx)
            output_tag_sequences.extend(curr_output_tag_sequences)
            if math.ceil(n * 100.0 / batch_num) % 25 == 0 and not quiet:
                print('\r++ predicting, batch %d/%d (%1.2f%%).' % (n + 1, batch_num, math.ceil(n * 100.0 / batch_num)),
                      end='', flush=True)
            # not_O = sum([sum([x != 2 for x in sent]) for sent in curr_output_idx2])
            # print(not_O)
        return output_tag_sequences

    def get_mask_from_word_sequences(self, word_sequences):
        batch_num = len(word_sequences)
        max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        mask_tensor = self.tensor_ensure_gpu(torch.zeros(batch_num, max_seq_len, dtype=torch.float))
        for k, word_seq in enumerate(word_sequences):
            mask_tensor[k, :len(word_seq)] = 1
        return mask_tensor # batch_size x max_seq_len

    def apply_mask(self, input_tensor, mask_tensor):
        input_tensor = self.tensor_ensure_gpu(input_tensor)
        mask_tensor = self.tensor_ensure_gpu(mask_tensor)
        return input_tensor*mask_tensor.unsqueeze(-1).expand_as(input_tensor)


    def push(self, args, word_sequences, tag_sequences, batch_size = 10, save_prototype_files = False,
                saliency_type = 'counterfactual'):
        '''
        push algorithm for prototype-based classifiers
        saves prototypes in two files: .txt with all protos in it and an .hdf5 dict of Prototypes
        see src.classes.prototype for Prototype

        alg 1: for each batch/sequence/distance, check if that distance is < its proto's min
        alg 2: for each prototype, pull out distances corresponding to instances of its class, check min                
        this implementation is alg 1 but should probably be alg 2

        conditions for prototypes:
            prototypes must correspond to instances of their pre-assigned class
            no two prototypes can point to the same instance (this is decided greedily)
            - keep heap of neighbors for each prototype, then within each class, greedily assign prototypes to neighbors

        saliency_type is used for self activation maps
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
                        
                        latent = Prototype(prototype_id = m,
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
                    self_activation_str = self.prototype_saliency_map(prototype.context, saliency_type = saliency_type, prototype_id = m)
                    f.write('%s \n\n\n' % self_activation_str)

            # and save the .hdf5 with the prototype dict
            save_path = os.path.join(args.save_dir, '%s-prototypes.hdf5' % args.save_name)
            torch.save(proto_idx_to_Prototype, save_path)

        return        
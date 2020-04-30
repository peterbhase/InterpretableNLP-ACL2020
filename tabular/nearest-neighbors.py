'''script which finds and saves k nearest neighbors to each prototype'''

import torch
#import numpy as np
import heapq
import os
import time
import argparse
import math
from src.factories.factory_data_io import DataIOFactory
from src.factories.factory_tagger import TaggerFactory
from src.classes.utils import *

class Latent():
    '''class for saving the human readable information associated with latent vectors'''

    def __init__(self, context, global_idx, distance, class_id, tag, batch_size = 10):
        # distance is distance to nearest prototype
        self.context = context
        self.global_idx = global_idx
        self.negative_distance = -distance # since popping from a heap removes the smallest element
        self.class_id = class_id
        self.tag = tag
        self.batch_size = batch_size

    def __str__(self):
        '''print with tokens separated by spaces and prototype word singled out with asterisks: *word*'''
        return ' '.join(self.context)

    def to_str(dataset_obj, self):
        return ' '.join(self.context)  

    def __lt__(self, other):
        # since popping from a heap removes the smallest element
        return self.negative_distance < other.negative_distance


def find_k_nearest_latents_to_prototypes(args, tagger, word_sequences, tag_sequences, data_name, k = 5):
    '''find k nearest latent vectors to each prototype'''
    '''save them as Prototype classes (slight abuse of the class name)'''
    '''two save formats: .txt with everything laid out, and .hdf5 dict of proto_id : list of Prototypes'''
    print("Finding nearest latents in %s dataset..." % data_name)
    tagger.eval()
    start = time.time()
    n_protos = tagger.num_prototypes
    batch_size = args.batch_size
    n_batches = math.floor(len(word_sequences) / batch_size)
    
    heaps = [] # one heap per prototype
    for _ in range(n_protos):
        heaps.append([])

    protos2times_used = {idx : 0 for idx in range(n_protos)} # record num times each prototype is the closest to an instance
    pred_tag_counts = {tag : 0 for tag in tagger.tag_seq_indexer.item2idx_dict.keys()}
    prototype_dict_path = os.path.join('saved_models', '%s-prototypes.hdf5' % args.load_name)
    prototype_dict = torch.load(prototype_dict_path)    

        
    # find k nearest for each prototype
    for n in range(n_batches):
        # import ipdb; ipdb.set_trace()
        i = n*batch_size
        if n < n_batches - 1:
            j = (n + 1)*batch_size
        else:
            j = len(word_sequences)        
    
        batch = word_sequences[i:j]
        targets = tagger.tag_seq_indexer.items2idx(tag_sequences[i:j]) # list of lists           
        latents, distances = tagger.push_forward(batch) # latent shape: batch_size x proto_dim. distances shape: batch_size x num_prototypes

        distances = distances.detach().cpu().numpy()

        for r in range(len(batch)): # reserving the k index, of course               
            class_id = targets[r]
            context = batch[r]
            tag = tagger.tag_seq_indexer.idx2item_dict[class_id]

            for m in range(n_protos):
                prototype = prototype_dict[m]
                distance = distances[r, m]
                latent = Latent(context = context, # have to make a latent for every prototype, since distance changes
                                class_id = class_id,
                                tag = tag,
                                distance = distance,
                                global_idx = (n,r),                                                                       
                                batch_size = batch_size)
                latent.vector = latents[r]

                # record if this prototpype is the closest
                if distance == min(distances[r,:]):
                    protos2times_used[m] += 1
                    pred_tag = tagger.tag_seq_indexer.idx2item_dict[m // tagger.num_prototypes_per_class]
                    pred_tag_counts[pred_tag] += 1

                if len(heaps[m]) < k: # simply add if there aren't k nearest yet
                    heapq.heappush(heaps[m], latent)
                else:
                    heapq.heappushpop(heaps[m], latent)
          

    end = time.time()
    print("Took %.1f seconds" % (end-start))

    # gather use_freq statistics
    use_freqs = np.zeros(n_protos)

    # saving time
    fname = os.path.join('saved_models', '%s-nearest-neighbors-%s.txt' % (args.load_name, data_name))
    with open(fname, 'w') as f:
            f.write('---------------------------------------------------------\n')
            f.write('\nFile contains prototype nearest neighbors for model at %s.hdf5 \n' % args.save_name)
            f.write('\nLevel of saliency/importance attributed to each word is denoted by the number of ` marks it is wrapped in\n')   
            f.write('\n---------------------------------------------------------\n\n')

            prototype_dict_path = os.path.join('saved_models', '%s-prototypes.hdf5' % args.load_name)
            prototype_dict = torch.load(prototype_dict_path)

            for m in range(n_protos):
                heaps[m].sort()
                heaps[m].reverse() # now closest to farthest    
                
                prototype = prototype_dict[m]
                proto_class_id = m // tagger.num_prototypes_per_class
                proto_class_name = tagger.tag_seq_indexer.idx2item_dict[proto_class_id]
                token_class_name = prototype.tag
                global_idx = prototype.global_idx

                # save use_freq
                use_freqs[m] = protos2times_used[m] / pred_tag_counts[proto_class_name]            

                f.write('\n---------------------------------------------------------\n')

                f.write('Prototype %d, tag: %s, token_tag: %s, global_idx: (%d,%d), within_class_use_freq: %2.2f \n' % (m, 
                                                                proto_class_name,
                                                                token_class_name,
                                                                global_idx[0], global_idx[1],
                                                                use_freqs[m]
                                                                )
                )
                prototype_str = data_row_to_str(tagger.data_encoder, prototype.context)
                f.write('%s \n\n' % (prototype_str))
                for i in range(k):
                    latent = heaps[m][i]
                    global_idx = latent.global_idx
                    f.write('Nearest neighbor %d, class: %d, tag: %s, global_idx: (%d,%d) \n' % (i, latent.class_id, latent.tag,
                                                                                                global_idx[0], global_idx[1]))
                    latent_str = data_row_to_str(tagger.data_encoder, prototype.context)
                    f.write('%s \n\n' % (latent_str))

                f.write('\n---------------------------------------------------------\n\n')                            

    # save the heaps var, now that they're all sorted
    save_path = os.path.join('saved_models', '%s-nearest-neighbors-%s.hdf5' % (args.load_name, data_name))
    torch.save(heaps, save_path)

    # show use_freq stats
    print("prototype use frequencies:\n", np.round(use_freqs,2))
    print("frequency entropy: ", sum(-use_freqs * np.log(use_freqs)))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find k nearest neighbors to prototypes for a tagger')
    parser.add_argument('--load-name', help='Path to load from the trained model.')
    parser.add_argument('--data-name', default='adult',
                        help='Train data in format defined by --data-io param.')
    parser.add_argument('--output', '-o', help='Output JSON filename.',
                        default='out.json')
    parser.add_argument('--data-io', '-d', 
                        default='adult', help='Data read file format.')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU device number, 0 by default, -1 means CPU.')
    parser.add_argument('--save-name', default='push',
                        help='for debugging push')    
    parser.add_argument('--save-dir', '-s', default='saved_models',
                    help='Path to dir to save the trained model.')
    parser.add_argument('--verbose', type=str2bool, default=False, help='Show additional information.', nargs='?',
                        choices=['yes (default)', True, 'no', False])    
    parser.add_argument('--batch-size', '-b', type=int, default=10, help='Batch size, samples.')
    parser.add_argument('--save-data', type=str2bool, default=False, help='Save a new dataset split.')
    
    parser.add_argument('--saliency-type', default='counterfactual', help='Data read file format.',
                                choices = ['attention','serrano','li','simonyan','directional','counterfactual'])     
    args = parser.parse_args()

    # Load tagger model
    load_path = os.path.join('saved_models','%s.hdf5' % args.load_name)
    print("Loading model from %s" % load_path)
    tagger = TaggerFactory.load(load_path, args.gpu)    
    tagger.eval()
    
    # Load text data as lists of lists of words (sequences) and corresponding list of lists of tags
    data_io = DataIOFactory.create(args)
    dataset, X_train, Y_train, X_dev, Y_dev, X_test, Y_test = data_io.read_train_dev_test(args)

    # and run
    find_k_nearest_latents_to_prototypes(args=args,
                                         tagger=tagger,
                                         word_sequences=X_train,
                                         tag_sequences=Y_train,
                                         data_name="train",
                                         k=5)

    find_k_nearest_latents_to_prototypes(args=args,
                                         tagger=tagger,
                                         word_sequences=X_dev,
                                         tag_sequences=Y_dev,
                                         data_name="dev",
                                         k=4)    

    # find_k_nearest_latents_to_prototypes(args=args,
    #                                      tagger=tagger,
    #                                      word_sequences=word_sequences_test,
    #                                      tag_sequences=tag_sequences_test,
    #                                      data_name="test",
    #                                      k=5)

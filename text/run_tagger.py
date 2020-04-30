'''
get model accuracy on data
'''

from __future__ import print_function
import argparse
import json
import os
from src.factories.factory_evaluator import EvaluatorFactory
from src.factories.factory_data_io import DataIOFactory
from src.factories.factory_tagger import TaggerFactory
from src.factories.factory_datasets_bank import DatasetsBankFactory
from src.classes.utils import *
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run trained model')
    parser.add_argument('--load-name', help='Path to load from the trained model.',
                        default='birnn')
    parser.add_argument('--data-dir', default='data/rt-polarity',
                        help='Train data in format defined by --data-io param.')   
    parser.add_argument('--input', help='Input CoNNL filename.',
                        default='data/NER/CoNNL_2003_shared_task/test.txt')
    parser.add_argument('--output', '-o', help='Output JSON filename.',
                        default='out.json')
    parser.add_argument('--data-io', '-d', default='reviews', help='Data read file format.')
    parser.add_argument('--evaluator', '-v', default='token-acc',
                        help='Evaluation method.',
                        choices=['f1-connl', 'f1-alpha-match-10',
                                 'f1-alpha-match-05', 'f1-macro', 'f05-macro',
                                 'token-acc'])
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU device number, 0 by default, -1 means CPU.')
    parser.add_argument('--break_early', '-b', type=bool, default=False)
    parser.add_argument('--save-name', default='push',
                        help='name for saving the pushed model')    
    parser.add_argument('--save-dir', '-s', default='saved_models',
                        help='Path to dir to save the trained model.')  
    parser.add_argument('--verbose', type=str2bool, default=True, help='Show additional information.', nargs='?',
                        choices=['yes (default)', True, 'no', False])      
    parser.add_argument('--dataset-sort', type=str2bool, default=False, help='Sort sequences by length for training.',
                        nargs='?', choices=['yes', True, 'no (default)', False])
    parser.add_argument('--save-data', type=str2bool, default=False, help='Save a new dataset split.')
    args = parser.parse_args()

    # Load tagger model
    load_path = os.path.join('saved_models','%s.hdf5' % args.load_name)
    print("Loading model from %s" % load_path)
    tagger = TaggerFactory.load(load_path, args.gpu)   

    # random question -- how far is the unknown vector from the other embeddings?
    # (Answer: very far. 99% quanile of cosine distances is .02)
    # def dist(v,u):
    #     v = np.array(v)
    #     u = np.array(u)
    #     v_norm =  np.dot(v,v)
    #     u_norm =  np.dot(u,u)
    #     if v_norm == 0 or u_norm == 0:
    #         return -1
    #     return np.dot(v,u) / v_norm / u_norm

    # unk_vec = tagger.word_seq_indexer.embedding_vectors_list[1]
    # dists2unk = [dist(unk_vec, vec) for vec in tagger.word_seq_indexer.embedding_vectors_list] 
    # quantiles = np.arange(.9,1.01,.01)
    # qs = np.quantile(dists2unk,quantiles)
    # print([(x,y) for (x,y) in zip(quantiles,qs)])


    # Create DataIO object
    data_io = DataIOFactory.create(args)
    
    # Load text data as lists of lists of words (sequences) and corresponding list of lists of tags
    data_io = DataIOFactory.create(args)
    word_sequences_train, tag_sequences_train, word_sequences_dev, tag_sequences_dev, word_sequences_test, tag_sequences_test = data_io.read_train_dev_test(args)  
    
    # DatasetsBank provides storing the different dataset subsets (train/dev/test) and sampling batches
    loading_word_seq_indexer = False
    datasets_bank = DatasetsBankFactory.create(args)
    datasets_bank.add_train_sequences(word_sequences_train, tag_sequences_train, loading_word_seq_indexer)
    datasets_bank.add_dev_sequences(word_sequences_dev, tag_sequences_dev, loading_word_seq_indexer)
    datasets_bank.add_test_sequences(word_sequences_test, tag_sequences_test, loading_word_seq_indexer)

    # Create evaluator
    evaluator = EvaluatorFactory.create(args)

    train_score, dev_score, test_score, test_msg = evaluator.get_evaluation_score_train_dev_test(tagger,
                                                                                                 datasets_bank,
                                                                                                 batch_size=1)
    print('\n train / dev / test | %1.2f / %1.2f / %1.2f.' % (train_score, dev_score, test_score))  
    print(test_msg)

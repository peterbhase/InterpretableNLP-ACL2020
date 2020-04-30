'''
run to generate examples of each explanation method on test movie reviews
'''


from __future__ import print_function
import argparse
import json
from src.factories.factory_evaluator import EvaluatorFactory
from src.factories.factory_data_io import DataIOFactory
from src.factories.factory_datasets_bank import DatasetsBankFactory
from src.factories.factory_tagger import TaggerFactory
import numpy as np
import pandas as pd
import os
import torch
from src.classes.utils import *
from src.classes.latent import Latent
from lime.lime import lime_text
from lime.lime.lime_text import LimeTextExplainer
from anchor.anchor.anchor_text import AnchorText
from anchor.anchor.utils import *
import spacy, en_core_web_lg
import time
import warnings
from transformers import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to get HIT data')
    parser.add_argument('--data-dir', default='data/rt-polarity',
                        help='Train data in format defined by --data-io param.')
    parser.add_argument('--data-io', '-d', default='reviews', help='Data read file format.')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU device number, 0 by default, -1 means CPU.')
    parser.add_argument('--seed-num', type=int, default=1, help='Random seed number, only used when > 0.')
    parser.add_argument('--save-data', type=str2bool, default=False, help='Save a new dataset split.')
    parser.add_argument('--verbose', type=str2bool, default=True, help='Show additional information.', nargs='?',
                    choices=['yes (default)', True, 'no', False])

    parser.add_argument('--threshold', type=float, default=.75, help='Anchor threshold')        
    parser.add_argument('--saliency-type', default='directional', help = "Saliency map methods. See tagger.saliency_map methods")
    parser.add_argument('--batch-size', '-b', type=int, default= 40 , help='Batch size, samples.')
    parser.add_argument('--save-dir', '-s', default='data/HIT-data-test', help='Path to dir to save the data.')

    parser.add_argument('--save-all', type=str2bool, default=False, help='Save entire new batch of data')

    args = parser.parse_args()
    start_time = time.time()

    warnings.filterwarnings('ignore')

    if args.seed_num > 0:
        np.random.seed(args.seed_num)
        torch.manual_seed(args.seed_num)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed_num)

    print("Working on gpu: %d" % torch.cuda.current_device())

    # save dirs
    makedir(args.save_dir)
    makedir(os.path.join(args.save_dir,'Tests'))
    makedir(os.path.join(args.save_dir,'Tests','Forward'))
    makedir(os.path.join(args.save_dir,'Tests','Forward','Learning-Data'))
    makedir(os.path.join(args.save_dir,'Tests','Counterfactual'))
    makedir(os.path.join(args.save_dir,'Tests','Validity'))
    makedir(os.path.join(args.save_dir,'Master-Data-Files'))

    # Load text data as lists of lists of words (sequences) and corresponding list of lists of tags
    data_io = DataIOFactory.create(args)
    word_sequences_train, tag_sequences_train, word_sequences_dev, tag_sequences_dev, word_sequences_test, tag_sequences_test = data_io.read_train_dev_test(args)

    # Load taggers
    print("Loading models...")
    blackbox_load_name = 'attention'
    prototype_load_name_1 = 'proto-p40-kmeans-push20'
    prototype_load_name_2 = 'proto-fixed-p40'
    blackbox_path = os.path.join('saved_models','%s.hdf5' % blackbox_load_name)    
    prototype_path_1 = os.path.join('saved_models','%s.hdf5' % prototype_load_name_1)
    prototype_path_2 = os.path.join('saved_models','%s.hdf5' % prototype_load_name_2)
    blackbox_tagger = TaggerFactory.load(blackbox_path, args.gpu)    
    prototype_tagger = TaggerFactory.load(prototype_path_1, args.gpu)    
    fixed_prototype_tagger = TaggerFactory.load(prototype_path_2, args.gpu)       

    # put taggers in dict
    name2tagger_dict = {
        'blackbox' : blackbox_tagger,
        'prototype' : prototype_tagger,
    }

    # all tags and predicted tags must be one of these
    class_names = ['pos','neg']    

    # tokenizer + word embedding object (Spacy object) for perturb_sentence() from anchors.anchors.utils
    print("Loading spacy object...")
    spacy_obj = en_core_web_lg.load()
    neighbors_obj = Neighbors(spacy_obj) # class from anchor.anchor.util

    # create LIME and ANCHOR objects
    LIME = LimeTextExplainer(verbose=False,
                            class_names=class_names)
    Anchor = AnchorText(nlp = spacy_obj,
                        class_names=class_names,
                        use_unk_distribution = False)    
    # Anchor = AnchorText(nlp = spacy_obj,
    #                     class_names=class_names,
    #                     use_unk_distribution = True,
    #                     mask_string = '<unk>')
    similar_case_fn = fixed_prototype_tagger.explain_instance
    composite_explain_fn = get_composite_explain_fn(args, blackbox_tagger, LIME, Anchor, similar_case_fn, neighbors_obj)

    tokenizer = None
    language_model = None


     # save an entirely new set of explanations from old data files
    def get_explain_fn(name):
        # get explain_fn
        if name == 'prototype':
            explain_fn = prototype_tagger.explain_instance
        elif name == 'LIME':
            explain_fn = get_LIME_fn(LIME, blackbox_tagger)
        elif name == 'Anchor':
            explain_fn = get_Anchor_fn(Anchor, blackbox_tagger, args, short_exp = False)
        elif name == 'omission':
            explain_fn = get_omission_fn(blackbox_tagger, neighbors_obj)
        elif name == 'decision_boundary':
            explain_fn = get_decision_boundary_fn(blackbox_tagger, neighbors_obj)
        elif name == 'composite':
            explain_fn = composite_explain_fn
        return explain_fn


    # idx doesn't match data ordering because of ignore_idx in v3 and how idx get assigned based on enumerate() after removing ignore_idx
    # example_idx = [1939, 1450, 1047, 558, 109]


    # these examples from test data    
    examples = []
    examples.extend(
        [
        'this is not a good movie .', 
        'this is not a bad movie .',
        'the entire movie is in need of a scented bath .',
        'more trifle than triumph .',
        'despite modest aspirations its occasional charms are not to be dismissed .',
        'one ca n\'t deny its seriousness and quality .',
        'a bittersweet film , simple in form but rich with human events .'
        ])

    example_predictions = blackbox_tagger.predict_tags_from_words(examples)

    zipped = [(example, pred) for (example,pred) in zip(examples, example_predictions)]
    print(zipped)

    conditions = ['LIME','Anchor','prototype','decision_boundary']

    file = open('examples.txt','w')

    for example in examples:

        file.write('EXAMPLE: %s \n' % example)

        print('on example: %s' % example)

        for name in conditions:
            print('getting %s explanation' % name)
            explain_fn = get_explain_fn(name)

            explanation = explain_fn(example)
            file.write('-'*10 + '\n' + explanation + '\n' + '-'*10 + '\n')




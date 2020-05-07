'''
saves HIT data 

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
# from transformers import *

def get_balanced_batches(word_sequences, tag_sequences, name2tagger_dict, constrain_to_classes, num_batches = 3, batch_size = 100, 
                        ignore_idx = None,
                        load_path_1 = None, load_path_2 = None, load_path_3 = None):
    '''
    return num_batches batches of batch_size balanced by predictions, i.e. equal num TP/FP/TN/FN (enforcing agreement among models in name2tagger_dict)
    format is list of batches. batch is list of tuples. tuple is (idx, word_sequence, tag)
    '''
    # learning data. validity data. counterfactual data

    # load data if path provided
    batches = []
    load_paths = [load_path_1, load_path_2, load_path_3]
    for load_path in load_paths:
        if load_path is not None:
            batch = []
            data = pd.read_excel(load_path)
            idx = data.id
            word_sequences = data.context
            tag_sequences = data.tag
            for i in range(len(idx)):
                batch.append((idx, word_sequences[i], tag_sequences[i]))
                batches.append(batch)


    # otherwise, gather data
    else:
        assert batch_size % 4 == 0, "Batch size should be divisable by 4"
        if constrain_to_classes is None:
            constrain_to_classes = tagger.tag_seq_indexer.item2idx_dict.keys()
        TP_idx = []
        FP_idx = []
        TN_idx = []
        FN_idx = []
        batches = []

        blackbox_tagger = name2tagger_dict['blackbox']
        prototype_tagger = name2tagger_dict['prototype']
        pos_neg_dict = blackbox_tagger.tag_seq_indexer.item2idx_dict

        # remove certain idx
        if ignore_idx is not None:
            word_sequences = [words for i, words in enumerate(word_sequences) if i not in ignore_idx]
            tag_sequences = [tags for i, tags in enumerate(tag_sequences) if i not in ignore_idx]

        # first, check agreement of predictions
        blackbox_tags = blackbox_tagger.predict_tags_from_words(word_sequences, constrain_to_classes = constrain_to_classes, quiet = True)
        prototype_tags = prototype_tagger.predict_tags_from_words(word_sequences, constrain_to_classes = constrain_to_classes, quiet = True)
        blackbox_pred_type = np.array([prediction_type(pos_neg_dict, tag, pred_tag) for (tag,pred_tag) in zip(tag_sequences, blackbox_tags)])
        prototype_pred_type = np.array([prediction_type(pos_neg_dict, tag, pred_tag) for (tag,pred_tag) in zip(tag_sequences, prototype_tags)])
        agree_bool = (blackbox_pred_type == prototype_pred_type)
        agree_idx = np.argwhere(agree_bool).reshape(-1)
        
        agree_types = blackbox_pred_type[agree_idx]
        # print("Number of blackbox prediction types")
        # print([(pred_type, sum(blackbox_pred_type == pred_type)) for pred_type in set(blackbox_pred_type)]) # so this is table(df$var) in R
        print("Number of agreed upon prediction types between models for the provided dataset:")
        print([(pred_type, sum(agree_types == pred_type)) for pred_type in set(agree_types)]) # so this is table(df$var) in R

        # pick out pred_types from those that the models agree upon
        TP_idx = np.argwhere((blackbox_pred_type == 'TP') * agree_bool).reshape(-1)
        FP_idx = np.argwhere((blackbox_pred_type == 'FP') * agree_bool).reshape(-1)
        TN_idx = np.argwhere((blackbox_pred_type == 'TN') * agree_bool).reshape(-1)
        FN_idx = np.argwhere((blackbox_pred_type == 'FN') * agree_bool).reshape(-1)

        num_each_per_batch = batch_size / 4
        total_to_sample = int(num_each_per_batch * num_batches)

        # sample even number of each kind of instance
        try:
            TP_sample_idx = np.random.choice(TP_idx, total_to_sample, replace=False)
            FP_sample_idx = np.random.choice(FP_idx, total_to_sample, replace=False)
            TN_sample_idx = np.random.choice(TN_idx, total_to_sample, replace=False)
            FN_sample_idx = np.random.choice(FN_idx, total_to_sample, replace=False)
        except:
            raise ValueError("Requesting too many of some kind(s) of predictions, should have at least %d of each type" % total_to_sample)

        # split idx to be used in each batch
        TP_sample_idx_split = np.split(TP_sample_idx, num_batches)
        FP_sample_idx_split = np.split(FP_sample_idx, num_batches)
        TN_sample_idx_split = np.split(TN_sample_idx, num_batches)
        FN_sample_idx_split = np.split(FN_sample_idx, num_batches)

        for i in range(num_batches):
            batch = []
            for idx in np.concatenate((
                    TP_sample_idx_split[i],
                    FP_sample_idx_split[i],
                    TN_sample_idx_split[i],
                    FN_sample_idx_split[i]
                    )):
                batch.append((idx, word_sequences[idx], tag_sequences[idx]))
            batches.append(batch)

        return batches


def pick_perturbation(text, name2tagger_dict, pick_type, neighbors_obj, constrain_to_classes = None, num_attempts = 10):
    '''
    sample from a perturbation distrbution around text
    need to have 'same' or 'diff' consensus among models in name2tagger_dict. i.e. a 'same' instance must be 'same' according to blackbox and proto
    by construction, blackbox and prototype agree on the original text, so here we just need to enforcement agreement on perturbations
    if pick_type is same, else sample one of a different predicted class
    return that perturbation as a str, along with a str version with the perturbed words wrapped with double asterisks
    '''

    blackbox_tagger = name2tagger_dict['blackbox']
    prototype_tagger = name2tagger_dict['prototype']
    original_prediction = np.array(blackbox_tagger.predict_tags_from_words([text], constrain_to_classes=constrain_to_classes, quiet = True)[0])
    perturbation_found = False
    start = time.time()
    word_sequence = text.split()
    
    # sampling parameters -- these are incremented if no perturbation of the needed pick_type is found
    top_n = 100
    proba_change = .25 - min(.005 * len(text.split()), .15) # lower bound at .1, and decrease prob from .25 according to length until then
    temperature = .4
    for i in range(num_attempts):        
        try:
            perturbations, _, _ = perturb_sentence(
                                        text = text, 
                                        present = [], 
                                        n = 10000, 
                                        neighbors = neighbors_obj, 
                                        top_n=top_n,
                                        proba_change = proba_change,
                                        temperature = temperature, 
                                        use_proba=False)           
            perturbations = set(perturbations) # remove duplicates
            perturbations = [pert for pert in perturbations if pert != text] # remove unperturbed examples
            perturbations = [combine_contractions(pert) for pert in perturbations] # combine contractions

            blackbox_predictions = np.array(blackbox_tagger.predict_tags_from_words(perturbations, constrain_to_classes = constrain_to_classes, quiet = True))
            prototype_predictions = np.array(prototype_tagger.predict_tags_from_words(perturbations, constrain_to_classes = constrain_to_classes, quiet = True))

            # enforce agreement on perturbations
            eligible_same_prediction = (original_prediction == blackbox_predictions) * (original_prediction == prototype_predictions)
            eligible_diff_prediction = (original_prediction != blackbox_predictions) * (original_prediction != prototype_predictions)

            models_disagree = (blackbox_predictions != prototype_predictions)            
            
            same_idx = np.argwhere(eligible_same_prediction).reshape(-1)
            diff_idx = np.argwhere(eligible_diff_prediction).reshape(-1)
            
            # pick randomly if pick_type is same, else pick lowest edit distance counterfactual
            pick_idx = same_idx if pick_type == 'same' else diff_idx
            pick_id = sample_capped_edit_distance_id(text, perturbations, pick_idx)
            chosen_perturbation = perturbations[pick_id]
            chosen_perturbation_highlighted = highlight_differences(text.split(),chosen_perturbation.split())
            perturbation_found = True
            # print("Perturbation numbers:")
            # print("Num eligible same: %d | Num eligible diff: %d | Total: %d" % (sum(eligible_same_prediction),sum(eligible_diff_prediction),len(perturbations)))
            if perturbation_found:
                # print("Got perturbation on attempt %d" % i)
                # print("Getting perturbation took %.2f seconds" % (time.time() - start))
                break            
            else:
                print("Could not find perturbation of needed type on sample %d" % i)
                top_n += 10
                proba_change += .05
                temperature += .4
        except:
            print("Code failure while getting perturbation of needed type on sample %d: \n%s" % (i, text))
            # print("Num eligible same: %d | Num eligible diff: %d | Total: %d" % (sum(eligible_same_prediction),sum(eligible_diff_prediction),len(perturbations)))            
            top_n += 10
            proba_change += .05
            temperature += .4
            continue
    if not perturbation_found:
        # assuming it was a failure to find a diff predicted class perturbation
        print("\n Failed to get diff perturbation for instance (will pick a perturbation of the same prediction type): \n %s" % text)        
        pick_id = sample_capped_edit_distance_id(text, perturbations, same_idx)
        chosen_perturbation = perturbations[pick_id]
        chosen_perturbation_highlighted = highlight_differences(text.split(),chosen_perturbation.split())
        print("\n Picked this perturbation: \n %s" % chosen_perturbation_highlighted)

    return chosen_perturbation, chosen_perturbation_highlighted


def save_forward_data(args, batch, name2tagger_dict, constrain_to_classes, save_dir, condition_names):
    '''
    save a file for each condition, as well as a master file with all data
    master file: id, context, true_tag, tags-for-each-tagger
    for each condition: id, context, user
    '''

    # unpack the batch
    instance_idx = []
    word_sequences = []    
    tags = []
    for (idx, words, tag) in batch:
        instance_idx.append(idx)
        word_sequences.append(words)
        tags.append(tag)
    text_sequences = [' '.join(words) for words in word_sequences]    

    # get batch predicted tags
    blackbox_tagger = name2tagger_dict['blackbox']
    predicted_tags = blackbox_tagger.predict_tags_from_words(word_sequences, constrain_to_classes = constrain_to_classes, quiet = True)

    # set up data frames
    test_data = pd.DataFrame(
        {'id' : instance_idx,
        'context' : text_sequences,
        'user' : ""
        }
    )        
    test_data = test_data[['id','context','user']] #re-order data
    
    master_data = pd.DataFrame(
        {'id' : instance_idx,
        'context' : text_sequences,
        'label' : tags
        }
    )
    master_data['model'] = predicted_tags

    # save an excel file for each condition
    save_names = [name for name in condition_names if name != 'blackbox']
    for save_name in save_names:
        pre_save_path = os.path.join(save_dir,'Tests','Forward','forward-pre-%s.xlsx' % save_name)
        post_save_path = os.path.join(save_dir,'Tests','Forward','forward-post-%s.xlsx' % save_name)

        test_data.sample(frac=1).to_excel(pre_save_path, index = False)
        test_data.sample(frac=1).to_excel(post_save_path, index = False)

    # save master data
    master_save_path = os.path.join(save_dir,'Master-Data-Files','master-forward.xlsx')
    master_data = master_data[['id','context','label','model']]
    master_data.to_excel(master_save_path, index = False)

    return

def save_validity_data(args, batch, name2tagger_dict, constrain_to_classes, save_dir, condition_names,
                        LIME, Anchor, neighbors_obj, language_model, tokenizer, composite_explain_fn):
    '''
    save a master file for each condition with explanations in randomized order
    master file: id, context, true_tag, pred_tag, and explanations for each tagger
    - note pred_tag consistent between blackbox_tagger and prototype_tagger by construction of batch
    '''

    # unpack the batch
    instance_idx = []
    word_sequences = []    
    tags = []
    for i, (idx, words, tag) in enumerate(batch):
        instance_idx.append(idx)
        word_sequences.append(words)
        tags.append(tag)
    text_sequences = [' '.join(words) for words in word_sequences]

    # get predictions
    pos_neg_dict = {'neg': 0, 'pos': 1}
    blackbox_tagger = name2tagger_dict['blackbox']
    blackbox_tags = blackbox_tagger.predict_tags_from_words(word_sequences, constrain_to_classes = constrain_to_classes, quiet = True)

    # classifier_fns for lime and anchor
    lime_classifier_fn = blackbox_tagger.predict_probs_from_words
    anchor_classifier_fn = blackbox_tagger.predict_idx_from_words_np

    # set up master data
    master_data = pd.DataFrame(
        {'id' : instance_idx,
        'context' : text_sequences,
        'label' : tags,
        'model' : blackbox_tags
        }
    )

    # for each explanation condition, attach explanations to master
    explanation_conditions = [name for name in condition_names if name != 'blackbox']    
    for condition_name in explanation_conditions:
        explanations = []
        cf_data = pd.read_excel('data/HIT-data-test/Tests/Counterfactual/counterfactual-post-%s.xlsx' % condition_name)
        
        # condition-specific variables
        if condition_name == 'prototype':
            explanations = cf_data['explanation']
            prototype_explanations = explanations

        elif condition_name == 'LIME':        
            explanations = cf_data['explanation']
            LIME_explanations = explanations

        elif condition_name == 'Anchor':
            explanations = cf_data['explanation']
            Anchor_explanations = explanations

        elif condition_name == 'omission':
            explanations = cf_data['explanation']

        elif condition_name == 'decision_boundary':
            explanations = cf_data['explanation']
            decision_boundary_explanations = explanations

        elif condition_name == 'composite':
            for i in range(len(word_sequences)):
                counterfactual_explanation = decision_boundary_last_step(decision_boundary_explanations[i])
                explanation = make_composite_explanation(LIME_explanations[i], 
                                                    Anchor_explanations[i], 
                                                    decision_boundary_explanations[i], 
                                                    prototype_explanations[i])
                explanations.append(explanation)
    
        # condition-general variables
        master_data['%s_explanation' % condition_name] = explanations
        master_data['user_%s' % condition_name] = ""

    # save a file for each condition with explanations in mostly-random order      
    for i in range(len(explanation_conditions)):

        # current condition and save path
        current_condition = explanation_conditions[i]
        master_save_path = os.path.join(save_dir, 'Tests', 'Validity', 'validity-%s.xlsx' % current_condition)
        
        # always begin with that condition's explanations and end with composite (if that condition isn't itself composite)        
        save_conditions = [name for name in condition_names if name != 'blackbox' and name != 'composite' and name != current_condition]
        explanation_order = ['id']
        np.random.shuffle(save_conditions)
        ordering = [current_condition] + save_conditions + (['composite'] if current_condition != 'composite' else [])
        for name in ordering: 
            explanation_order += ['context', 'label', 'model', '%s_explanation' % name, 'user_%s' % name]
        save_data = master_data.copy()[explanation_order]

        # save data, this time sorted by id
        save_data = save_data.sort_values('id')
        save_data.to_excel(master_save_path, index = False)

    return


def save_counterfactual_data(args, batch, name2tagger_dict, constrain_to_classes, save_dir, condition_names,
                            LIME, Anchor, neighbors_obj, language_model, tokenizer, composite_explain_fn):
    assert len(batch) >= 8, "Batch should be at least 8 instances to cover 4 prediction types x 2 perturbation type conditions"
    '''
    NOTE that blackbox and prototype model will agree on all batch and perturbation predictions, by construction
    '''

    # unpack the batch
    instance_idx = []
    word_sequences = []    
    tags = []
    for (idx, words, tag) in batch:
        instance_idx.append(idx)
        word_sequences.append(words)
        tags.append(tag)
    text_sequences = [' '.join(words) for words in word_sequences]

    # get predictions
    blackbox_tagger = name2tagger_dict['blackbox']
    predicted_tags = blackbox_tagger.predict_tags_from_words(word_sequences, constrain_to_classes = constrain_to_classes, quiet = True)

    # classifier_fns for lime and anchor
    lime_classifier_fn = blackbox_tagger.predict_probs_from_words
    anchor_classifier_fn = blackbox_tagger.predict_idx_from_words_np

    # set up pre-condition-data, post-condition-data, and master_data
    pre_condition_data = pd.DataFrame(
        {'id' : instance_idx,
        'context' : text_sequences,
        'label' : tags
        }
    )    
    
    # get list of pick_types corresponding to the perturbations of word_sequences
    # within each prediction type, split ~50/50 same/diff -- within prediction types
    pos_neg_dict = blackbox_tagger.tag_seq_indexer.item2idx_dict    
    prediction_types = np.array([prediction_type(pos_neg_dict, tag, pred) for tag, pred in zip(tags, predicted_tags)])
    num_each_type = sum([pred_type == 'TP' for pred_type in prediction_types]) # assuming balanced types
    num_diff = int(np.ceil(num_each_type / 2))
    num_same = int(np.floor(num_each_type / 2))
    diff_same_set = np.concatenate(
                                (np.array(['same']*num_same),
                                np.array(['diff']*num_diff))
                                ) # will shuffle this set and assign to locations of each prediction_type
    perturbation_types = np.array(['placeholder']*(num_each_type*4))
    for pred_type in set(prediction_types):
        where_idx = np.argwhere(prediction_types == pred_type).reshape(-1)
        perturbation_types[where_idx] = np.random.permutation(diff_same_set)    

    # get perturbations    
    print("\tGetting perturbations...")
    perturbation_texts = []
    perturbation_texts_highlighted = [] # differences from original are highlighted
    perturbation_sequences = []
    for i, (text, pick_type) in enumerate(zip(text_sequences, perturbation_types)):
        # print("Perturbing instance %d" % i)
        perturbation, perturbation_highlighted = pick_perturbation(text, name2tagger_dict, pick_type, neighbors_obj)        
        perturbation_texts.append(perturbation)
        perturbation_texts_highlighted.append(perturbation_highlighted)
        perturbation_sequences.append(perturbation.split())

    # add perturbations to pre_condition data
    pre_condition_data['perturbation'] = perturbation_texts_highlighted

    # now copy over for bases of post_condition and master too
    post_condition_data = pd.DataFrame.copy(pre_condition_data)
    master_data = pd.DataFrame.copy(post_condition_data)    
    
    # add predicted_tags and perturbation_predicted_tags to master_data
    master_data['model'] = predicted_tags
    perturbation_predicted_tags = blackbox_tagger.predict_tags_from_words(perturbation_sequences, constrain_to_classes = constrain_to_classes, quiet = True)
    master_data['perturbation_model'] = perturbation_predicted_tags

    print("\tNum same perturbations: %d" % sum(np.array(perturbation_predicted_tags) == np.array(predicted_tags)))
    print("\tNum diff perturbations: %d" % sum(np.array(perturbation_predicted_tags) != np.array(predicted_tags)))

    # explain original word_sequences for each condition. save pre- and post-condition data
    print("\tGetting explanations...")
    explanation_conditions = [name for name in condition_names if name != 'blackbox']
    for condition_name in explanation_conditions:

        explanations = []

        # condition-specific variables
        if 'prototype' in condition_name:
            tagger = name2tagger_dict[condition_name]
            for word_sequence in word_sequences:
                explanation = tagger.explain_instance(word_sequence, saliency_type = "counterfactual", neighbors_obj = neighbors_obj,
                                                                counterfactual_method = 'unk')
                explanations.append(explanation)
            prototype_explanations = explanations

        elif condition_name == 'LIME':
            for text in text_sequences:
                explanation = LIME.safe_explain_instance(text, lime_classifier_fn)
                explanations.append(explanation)
            LIME_explanations = explanations

        elif condition_name == 'Anchor':
            for i, text in enumerate(text_sequences):                
                anchor_time = time.time()
                explanation = Anchor.safe_explain_instance(text, anchor_classifier_fn, threshold = args.threshold, verbose = False, short_exp = False)
                explanations.append(explanation)
                # print("Time to fit cf Anchor %d: %.2f seconds" % (i, time.time() - anchor_time))
            Anchor_explanations = explanations

        elif condition_name == 'omission':
            for word_sequence in word_sequences:
                explanation = blackbox_tagger.explain_instance(word_sequence, saliency_type = "counterfactual", neighbors_obj = neighbors_obj,
                                                                counterfactual_method = 'unk')
                explanations.append(explanation)    

        elif condition_name == 'decision_boundary':
            for i, word_sequence in enumerate(word_sequences):
                decision_time = time.time()
                explanation = blackbox_tagger.decision_boundary_explanation(word_sequence, neighbors_obj = neighbors_obj, sufficient_conditions_print = False)
                explanations.append(explanation)  
                # print("Time to fit cf decision_boundary %d: %.2f seconds" % (i, time.time() - decision_time))
            decision_boundary_explanations = explanations

        elif condition_name == 'composite':
            for i in range(len(word_sequences)):
                # counterfactual_explanation = decision_boundary_last_step(decision_boundary_explanations[i])
                explanation = make_composite_explanation(LIME_explanations[i], 
                                                    Anchor_explanations[i], 
                                                    decision_boundary_explanations[i], 
                                                    prototype_explanations[i])
                explanations.append(explanation)                      

        # save pre-condition data - NOTE saving shuffled data
        pre_condition_data_copy = pd.DataFrame.copy(pre_condition_data)        
        pre_condition_data_copy['model'] = predicted_tags
        pre_condition_data_copy['user_rating'] = ""
        pre_condition_data_copy['user_prediction'] = ""
        pre_condition_data_copy = pre_condition_data_copy[['id','context','label','model','perturbation','user_prediction']]
        pre_condition_save_path = os.path.join(args.save_dir, 'Tests', 'Counterfactual','counterfactual-pre-%s.xlsx' % condition_name)
        pre_condition_data_copy.sample(frac=1).to_excel(pre_condition_save_path, index = False)

        # save post-condition data - NOTE saving shuffled data
        post_condition_data_copy = pd.DataFrame.copy(post_condition_data)   
        post_condition_data_copy['model'] = predicted_tags
        post_condition_data_copy['explanation'] = explanations
        post_condition_data_copy['user_rating'] = ""
        post_condition_data_copy['user_prediction'] = ""
        post_condition_data_copy = post_condition_data_copy[['id','context','label','model','explanation','perturbation','user_rating', 'user_prediction']]
        post_condition_save_path = os.path.join(args.save_dir, 'Tests', 'Counterfactual','counterfactual-post-%s.xlsx' % condition_name)
        post_condition_data_copy.sample(frac=1).to_excel(post_condition_save_path, index = False)

    # reorder columns and save - NOTE saving shuffled data
    master_save_path = os.path.join(save_dir, 'Master-Data-Files','master-counterfactual.xlsx')    
    master_data = master_data[['id','context','perturbation', 'label', 'model', 'perturbation_model']]
    master_data.to_excel(master_save_path, index = False)

    return



def save_learning_data(args, word_sequences, tag_sequences, name2tagger_dict, batch_size, 
                    LIME, Anchor, 
                    condition_names, constrain_to_classes,
                    neighbors_obj = None, language_model = None, tokenizer = None, composite_explain_fn = None):
    '''
    saves learning data for the forward test learning period(s)
    for each condition in condition_names, save explanations from dev data
    we try to control for the quantity of explanation (as opposed to quality) by balancing counts of instances across conditions
    NOTE by construction, blackbox and prototype model will agree on these examples. see get_balanced_data()

    '''
    print("Getting learning data for forward test...")
    blackbox_tagger = name2tagger_dict['blackbox']
    prototype_tagger = name2tagger_dict['prototype']

    batches = get_balanced_batches(word_sequences, tag_sequences, name2tagger_dict, constrain_to_classes,
        num_batches = 1, batch_size = batch_size, ignore_idx = [156])    

    # unpack the batch
    instance_idx = []
    word_sequences = []    
    tags = []
    for (idx, words, tag) in batches[0]:
        instance_idx.append(idx)
        word_sequences.append(words)
        tags.append(tag)
    text_sequences = [' '.join(words) for words in word_sequences]

    # get predictions. NOTE by construction, blackbox and prototype model will agree on these examples. see get_balanced_data()
    blackbox_tagger = name2tagger_dict['blackbox']
    predicted_tags = blackbox_tagger.predict_tags_from_words(word_sequences, constrain_to_classes = constrain_to_classes, quiet = True)

    # classifier_fns for lime and anchor
    lime_classifier_fn = blackbox_tagger.predict_probs_from_words
    anchor_classifier_fn = blackbox_tagger.predict_idx_from_words_np

    # set up condition data base and master data
    condition_data = pd.DataFrame(
        {'id' : instance_idx,
        'context' : text_sequences,
        'label' : tags,
        }
    )
    condition_data = condition_data[['id','context','label']] # re-order

    # include blackbox here as a condition
    for condition_name in condition_names:
        explanations = []

        print('learning ', condition_name)

        # condition-specific variables
        if 'prototype' in condition_name:
            tagger = name2tagger_dict[condition_name]
            for word_sequence in word_sequences:
                explanation = tagger.explain_instance(word_sequence, saliency_type = "counterfactual", neighbors_obj = neighbors_obj,
                                                                counterfactual_method = 'unk')
                explanations.append(explanation)
            prototype_explanations = explanations

        elif condition_name == 'LIME':
            for text in text_sequences:
                explanation = LIME.safe_explain_instance(text, lime_classifier_fn)
                explanations.append(explanation)
            LIME_explanations = explanations

        elif condition_name == 'Anchor':
            for i, text in enumerate(text_sequences):                
                anchor_time = time.time()
                explanation = Anchor.safe_explain_instance(text, anchor_classifier_fn, threshold = args.threshold, verbose = False, short_exp = False)
                explanations.append(explanation)
                print("Time to fit learning Anchor %d: %.2f seconds" % (i, time.time() - anchor_time))
                print(explanation)
            Anchor_explanations = explanations

        elif condition_name == 'omission':
            for word_sequence in word_sequences:
                explanation = blackbox_tagger.explain_instance(word_sequence, saliency_type = "counterfactual", neighbors_obj = neighbors_obj,
                                                                counterfactual_method = 'unk')
                explanations.append(explanation)    

        elif condition_name == 'decision_boundary':
            for i, word_sequence in enumerate(word_sequences):
                decision_time = time.time()
                explanation = blackbox_tagger.decision_boundary_explanation(word_sequence, neighbors_obj = neighbors_obj, sufficient_conditions_print = False)
                explanations.append(explanation)  
                print("Time to fit learning decision_boundary %d: %.2f seconds" % (i, time.time() - decision_time))
            decision_boundary_explanations = explanations

        elif condition_name == 'composite':
            for i in range(len(word_sequences)):
                counterfactual_explanation = decision_boundary_last_step(decision_boundary_explanations[i])
                explanation = make_composite_explanation(LIME_explanations[i], 
                                                    Anchor_explanations[i], 
                                                    decision_boundary_explanations[i], 
                                                    prototype_explanations[i])
                explanations.append(explanation)

        # save data
        condition_data_copy = pd.DataFrame.copy(condition_data)
        condition_data_copy['model'] = predicted_tags         
        if condition_name == 'blackbox':
            condition_data_copy = condition_data_copy[['id','context','label','model']]            
        else: 
            condition_data_copy['explanation'] = explanations
            condition_data_copy = condition_data_copy[['id','context','label','model','explanation']] 
            condition_data_copy['user_rating'] = ''           
        condition_data_save_path = os.path.join(args.save_dir, 'Tests', 'Forward', 'learning-%s.xlsx' % condition_name)
        condition_data_copy.to_excel(condition_data_save_path, index = False)    
    
    return

    
def save_testing_data(args, word_sequences, tag_sequences, name2tagger_dict, batch_size, neighbors_obj,
                    LIME, Anchor, condition_names,
                    constrain_to_classes, language_model, tokenizer, composite_explain_fn):


    print("Getting balanced data...")
    batches = get_balanced_batches(word_sequences, tag_sequences, name2tagger_dict, constrain_to_classes,
        num_batches = 2, batch_size = batch_size, ignore_idx = [900, 313, 1791, 335, 695])

    forward_batch = batches[0]
    counterfactual_batch = batches[1]
    #validity_batch = batches[2]

    # print('Getting forward data...')
    save_forward_data(args = args,
        batch = forward_batch, 
        name2tagger_dict = name2tagger_dict, 
        constrain_to_classes = constrain_to_classes, 
        save_dir = args.save_dir, 
        condition_names = condition_names)

    print('Getting counterfactual data....')
    save_counterfactual_data(args = args,
        batch = counterfactual_batch, 
        name2tagger_dict = name2tagger_dict, 
        constrain_to_classes = constrain_to_classes, 
        save_dir = args.save_dir, 
        condition_names = condition_names,
        LIME = LIME,
        Anchor = Anchor,
        neighbors_obj = neighbors_obj,
        language_model = language_model,
        tokenizer = tokenizer,
        composite_explain_fn = composite_explain_fn)

    print('Getting validity data....')
    save_validity_data(args = args,
        batch = counterfactual_batch, 
        name2tagger_dict = name2tagger_dict, 
        constrain_to_classes = constrain_to_classes, 
        save_dir = args.save_dir, 
        condition_names = condition_names,
        LIME = LIME,
        Anchor = Anchor,
        neighbors_obj = neighbors_obj,
        language_model = language_model,
        tokenizer = tokenizer,
        composite_explain_fn = composite_explain_fn)    

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to get HIT data')
    parser.add_argument('--data-dir', default='data/rt-polarity',
                        help='Train data in format defined by --data-io param.')
    parser.add_argument('--data-io', '-d', default='reviews', help='Data read file format.')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU device number, 0 by default, -1 means CPU.')
    parser.add_argument('--seed-num', type=int, default=10, help='Random seed number, only used when > 0.')
    parser.add_argument('--save-data', type=str2bool, default=False, help='Save a new dataset split.')
    parser.add_argument('--verbose', type=str2bool, default=True, help='Show additional information.', nargs='?',
                    choices=['yes (default)', True, 'no', False])

    parser.add_argument('--threshold', type=float, default=.95, help='Anchor threshold')        
    parser.add_argument('--saliency-type', default='directional', help = "Saliency map methods. See tagger.saliency_map methods")
    parser.add_argument('--batch-size', '-b', type=int, default= 32, help='Batch size, samples.')
    parser.add_argument('--save-dir', '-s', default='data/HIT-data-test', help='Path to dir to save the data.')

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
    constrain_to_classes = ['pos','neg']    

    # tokenizer + word embedding object (Spacy object) for perturb_sentence() from anchors.anchors.utils
    print("Loading spacy object...")
    spacy_obj = en_core_web_lg.load()
    neighbors_obj = Neighbors(spacy_obj) # class from anchor.anchor.util

    # create LIME and ANCHOR objects
    LIME = LimeTextExplainer(verbose=False,
                            class_names=constrain_to_classes)
    Anchor = AnchorText(nlp = spacy_obj,
                        class_names=constrain_to_classes,
                        use_unk_distribution = False)
    similar_case_fn = fixed_prototype_tagger.give_similar_case
    composite_explain_fn = get_composite_explain_fn(args, blackbox_tagger, LIME, Anchor, similar_case_fn, neighbors_obj)


    # condition names
    condition_names = ['blackbox', 'prototype' , 'LIME', 'omission', 'decision_boundary', 'Anchor', 'composite']

    # language model and corresponding tokenizer for expected_score method of assigning word_importance / saliency
    tokenizer = None
    language_model = None

    # save learning data for forward test
    save_learning_data(args = args, 
                word_sequences = word_sequences_dev, 
                tag_sequences = tag_sequences_dev,
                name2tagger_dict = name2tagger_dict, 
                batch_size= 16, 
                neighbors_obj=neighbors_obj,
                LIME = LIME,
                Anchor = Anchor,
                condition_names = condition_names,
                constrain_to_classes=constrain_to_classes,
                language_model = language_model,
                tokenizer = tokenizer,
                composite_explain_fn = composite_explain_fn)

    # save test data for all tests
    save_testing_data(args = args, 
                word_sequences = word_sequences_test, 
                tag_sequences = tag_sequences_test,
                name2tagger_dict = name2tagger_dict, 
                batch_size = args.batch_size, 
                neighbors_obj = neighbors_obj,
                LIME = LIME,
                Anchor = Anchor,
                constrain_to_classes = constrain_to_classes,
                condition_names = condition_names,
                language_model = language_model,
                tokenizer = tokenizer,
                composite_explain_fn = composite_explain_fn)


    print('done!')
    print('total runtime: %.2f hours' % ((time.time() - start_time)/3600))

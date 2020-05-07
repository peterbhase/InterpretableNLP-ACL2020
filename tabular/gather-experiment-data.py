'''
saves HIT data 
'''

import argparse
from src.factories.factory_evaluator import EvaluatorFactory
from src.factories.factory_data_io import DataIOFactory
from src.factories.factory_datasets_bank import DatasetsBankFactory
from src.factories.factory_tagger import TaggerFactory
import numpy as np
import pandas as pd
import os
import torch
from src.classes.utils import *
from anchor import anchor_tabular
import time
from anchor import utils


def get_balanced_batches(word_sequences, tag_sequences, name2tagger_dict, batch_size, constrain_to_classes = None, num_batches = 3):
    '''
    return num_batches batches of batch_size balanced by predictions, i.e. equal num TP/FP/TN/FN (enforcing agreement among models in name2tagger_dict)
    format is list of batches. batch is list of tuples. tuple is (idx, word_sequence, tag)
    '''
    # learning data. validity data. counterfactual data

    assert batch_size % 4 == 0, "Batch size should be divisable by 4"

    TP_idx = []
    FP_idx = []
    TN_idx = []
    FN_idx = []
    batches = []

    blackbox_tagger = name2tagger_dict['blackbox']
    prototype_tagger = name2tagger_dict['prototype']
    pos_neg_dict = blackbox_tagger.tag_seq_indexer.item2idx_dict

    # first, check agreement of predictions
    blackbox_tags = name2tagger_dict['blackbox'].predict_tags_from_words(word_sequences, constrain_to_classes = constrain_to_classes, quiet = True)
    prototype_tags = name2tagger_dict['prototype'].predict_tags_from_words(word_sequences, constrain_to_classes = constrain_to_classes, quiet = True)
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


def pick_perturbation(data_row, name2tagger_dict, pick_type, explainer_obj, constrain_to_classes = None, num_attempts = 10):
    '''
    sample from a perturbation distrbution around text
    need to have 'same' or 'diff' consensus among models in name2tagger_dict. i.e. a 'same' instance must be 'same' according to blackbox and proto
    by construction, blackbox and prototype agree on the original text, so here we just need to enforcement agreement on perturbations
    if pick_type is same, else sample one of a different predicted class
    return that perturbation as a str, along with a str version with the perturbed words wrapped with double asterisks
    '''

    blackbox_tagger = name2tagger_dict['blackbox']
    prototype_tagger = name2tagger_dict['prototype']
    data_encoder_obj = prototype_tagger.data_encoder
    original_prediction = np.array(blackbox_tagger.predict_tags_from_words(data_row, constrain_to_classes=constrain_to_classes, quiet = True)[0])
    perturbation_found = False
    start = time.time()
    
    for i in range(num_attempts):        
        try:
            # sample data points -- treating this as blackbox sampler from Ribeiro
            # predict_proba_fn = self.predict_probs_from_words
            # sample_fn, mapping = explainer_obj.get_sample_fn(data_row, predict_proba_fn,
            #                                        sample_whole_instances=True,
            #                                        encode_before_forward=False)
            # X_dense_perturbations, data, _ = sample_fn([], 5000, False)

            # sample perturbations using our distribution (which is narrower than the default Anchor distribution)
            perturbations = sample_perturbations(data_encoder_obj, data_row)

            # get perturbation predictions from each model
            blackbox_predictions = np.array(blackbox_tagger.predict_tags_from_words(perturbations, constrain_to_classes = constrain_to_classes, quiet = True))
            prototype_predictions = np.array(prototype_tagger.predict_tags_from_words(perturbations, constrain_to_classes = constrain_to_classes, quiet = True))

            # enforce agreement on perturbations
            eligible_same_prediction = (original_prediction == blackbox_predictions) * (original_prediction == prototype_predictions)
            eligible_diff_prediction = (original_prediction != blackbox_predictions) * (original_prediction != prototype_predictions)
            
            same_idx = np.argwhere(eligible_same_prediction).reshape(-1)
            diff_idx = np.argwhere(eligible_diff_prediction).reshape(-1)
            
            pick_idx = np.random.choice(same_idx) if pick_type == 'same' else np.random.choice(diff_idx)
            chosen_perturbation = perturbations[pick_idx]
            chosen_perturbation_highlighted = highlight_differences(data_encoder_obj, data_row, chosen_perturbation)
            perturbation_found = True
            # print("Num eligible same: %d | Num eligible diff: %d | Total: %d" % (sum(eligible_same_prediction),sum(eligible_diff_prediction),len(perturbations)))
            if perturbation_found:
                # print("Got perturbation on attempt %d" % i)
                # print("Getting perturbation took %.2f seconds" % (time.time() - start))
                break            
        except:
            print("Code failure while getting perturbation of needed type on sample %d: \n%s" % (i, data_row_to_str(data_encoder_obj, data_row)))
            continue
    if not perturbation_found:
        # assuming it was a failure to find a diff predicted class perturbation
        print("\n Failed to get diff perturbation for instance (will pick a perturbation of the same prediction type)")
        pick_idx = np.random.choice(same_idx)
        chosen_perturbation = perturbations[pick_idx]
        chosen_perturbation_highlighted = highlight_differences(data_encoder_obj, data_row, chosen_perturbation)
        print("\n Picked this perturbation: \n %s" % chosen_perturbation_highlighted)

    return chosen_perturbation, chosen_perturbation_highlighted


def save_forward_data(args, batch, name2tagger_dict, save_dir, condition_names, constrain_to_classes = None):
    '''
    save a file for each condition, as well as a master file with all data
    master file: id, context, true_tag, tags-for-each-tagger
    for each condition: id, context, user
    '''

    # unpack tagger
    blackbox_tagger = name2tagger_dict['blackbox']
    data_encoder_obj = blackbox_tagger.data_encoder

    # unpack the batch
    instance_idx = []
    word_sequences = []    
    tags = []
    for (idx, words, tag) in batch:
        instance_idx.append(idx)
        word_sequences.append(words)
        tags.append(tag)
    text_sequences = [data_row_to_str(data_encoder_obj, data_row) for data_row in word_sequences]  
    word_sequences = np.array(word_sequences)

    # get batch predicted tags
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
        'label' : tags,
        'model' : predicted_tags
        }
    )

    # save pre- and post- versions of each file
    save_names = [name for name in condition_names if name != 'blackbox']
    for condition_name in save_names:
        pre_save_path = os.path.join(save_dir,'Tests','Forward','forward-pre-%s.xlsx' % condition_name)
        post_save_path = os.path.join(save_dir,'Tests','Forward','forward-post-%s.xlsx' % condition_name)

        test_data.sample(frac=1).to_excel(pre_save_path, index = False)
        test_data.sample(frac=1).to_excel(post_save_path, index = False)


    # save master data
    master_save_path = os.path.join(save_dir,'Master-Data-Files','master-forward.xlsx')
    master_data = master_data[['id','context','label','model']]
    master_data.to_excel(master_save_path, index = False)

    return

def save_validity_data(args, batch, name2tagger_dict, constrain_to_classes, save_dir, condition_names,
                        LIME_explain_fn, Anchor_explain_fn, composite_explain_fn):
    '''
    save a master file for each condition with explanations in randomized order
    master file: id, context, true_tag, pred_tag, and explanations for each tagger
    - note pred_tag consistent between blackbox_tagger and prototype_tagger by construction of batch
    '''

    # unpack taggers
    blackbox_tagger = name2tagger_dict['blackbox']
    prototype_tagger = name2tagger_dict['prototype']
    data_encoder_obj = blackbox_tagger.data_encoder

    # unpack the batch
    instance_idx = []
    word_sequences = []    
    tags = []
    for (idx, words, tag) in batch:
        instance_idx.append(idx)
        word_sequences.append(words)
        tags.append(tag)
    text_sequences = [data_row_to_str(data_encoder_obj, data_row) for data_row in word_sequences]  
    word_sequences = np.array(word_sequences)

    # get predictions
    blackbox_tags = blackbox_tagger.predict_tags_from_words(word_sequences, constrain_to_classes = constrain_to_classes, quiet = True)

    # classifier_fns for lime and anchor
    lime_classifier_fn = blackbox_tagger.predict_probs_from_words
    anchor_classifier_fn = blackbox_tagger.predict_idx_from_words

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
                                                    counterfactual_explanation, 
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
                            LIME_explain_fn, Anchor_explain_fn, composite_explain_fn, explainer_obj):
    assert len(batch) >= 8, "Batch should be at least 8 instances to cover 4 prediction types x 2 perturbation type conditions"
    '''
    NOTE that blackbox and prototype model will agree on all batch and perturbation predictions, by construction
    '''

    # get models
    blackbox_tagger = name2tagger_dict['blackbox']
    prototype_tagger = name2tagger_dict['prototype']
    data_encoder_obj = blackbox_tagger.data_encoder

    # unpack the batch
    instance_idx = []
    word_sequences = []
    tags = []
    for (idx, words, tag) in batch:
        instance_idx.append(idx)
        word_sequences.append(words)
        tags.append(tag)
    text_sequences = [data_row_to_str(data_encoder_obj, data_row) for data_row in word_sequences]  
    word_sequences = np.array(word_sequences)

    # get predictions (models agree on these)
    predicted_tags = blackbox_tagger.predict_tags_from_words(word_sequences, constrain_to_classes = constrain_to_classes, quiet = True)

    # classifier_fns for lime and anchor
    lime_classifier_fn = blackbox_tagger.predict_probs_from_words
    anchor_classifier_fn = blackbox_tagger.predict_idx_from_words

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
    for i, (data_row, pick_type) in enumerate(zip(word_sequences, perturbation_types)):
        perturbation, perturbation_highlighted = pick_perturbation(data_row, name2tagger_dict, pick_type, explainer_obj = explainer_obj)        
        perturbation_texts.append(perturbation)
        perturbation_texts_highlighted.append(perturbation_highlighted)
        perturbation_sequences.append(perturbation)

    # add perturbations to pre_condition data
    pre_condition_data['perturbation'] = perturbation_texts_highlighted

    # init post_condition and master data
    post_condition_data = pd.DataFrame.copy(pre_condition_data)
    master_data = pd.DataFrame.copy(post_condition_data)    
    
    # add predicted_tags and perturbation_predicted_tags to master_data
    master_data['model'] = predicted_tags
    perturbation_predicted_tags = blackbox_tagger.predict_tags_from_words(np.array(perturbation_sequences), constrain_to_classes = constrain_to_classes, quiet = True)
    master_data['perturbation_model'] = perturbation_predicted_tags

    print("\tNum same perturbations: %d" % sum(np.array(perturbation_predicted_tags) == np.array(predicted_tags)))
    print("\tNum diff perturbations: %d" % sum(np.array(perturbation_predicted_tags) != np.array(predicted_tags)))

    # explain original word_sequences for each condition. save pre- and post-condition data
    print("\tGetting explanations...")
    explanation_conditions = [name for name in condition_names if name != 'blackbox']
    for condition_name in explanation_conditions:

        explanations = []

        # condition-specific variables
        if condition_name == 'prototype':
            for word_sequence in word_sequences:
                explanation = prototype_tagger.explain_instance(word_sequence, counterfactual_method = args.counterfactual_method)
                explanations.append(explanation)
            prototype_explanations = explanations

        elif condition_name == 'LIME':
            for word_sequence in word_sequences:
                explanation = LIME_explain_fn(word_sequence)
                explanations.append(explanation)
            LIME_explanations = explanations

        elif condition_name == 'Anchor':
            for i, word_sequence in enumerate(word_sequences):
                explanation = Anchor_explain_fn(word_sequence)
                explanations.append(explanation)
            Anchor_explanations = explanations

        elif condition_name == 'omission':
            for word_sequence in word_sequences:
                explanation = blackbox_tagger.saliency_explain_instance(word_sequence, counterfactual_method = args.counterfactual_method)
                explanations.append(explanation)    

        elif condition_name == 'decision_boundary':
            for word_sequence in word_sequences:
                explanation = blackbox_tagger.decision_boundary_explanation(word_sequence, sufficient_conditions_print = False)
                explanations.append(explanation)  
            decision_boundary_explanations = explanations

        elif condition_name == 'composite':
            for i in range(len(word_sequences)):
                counterfactual_explanation = decision_boundary_last_step(decision_boundary_explanations[i])
                explanation = make_composite_explanation(LIME_explanations[i], 
                                                    Anchor_explanations[i], 
                                                    counterfactual_explanation, 
                                                    prototype_explanations[i])
                explanations.append(explanation)


        # save pre-condition data - NOTE saving shuffled data
        pre_condition_data_copy = pd.DataFrame.copy(pre_condition_data)        
        pre_condition_data_copy['model'] = predicted_tags
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
        post_condition_data_copy = post_condition_data_copy[['id','context','label','model','explanation','perturbation','user_rating','user_prediction']]
        post_condition_save_path = os.path.join(args.save_dir, 'Tests', 'Counterfactual','counterfactual-post-%s.xlsx' % condition_name)
        post_condition_data_copy.sample(frac=1).to_excel(post_condition_save_path, index = False)

    # reorder columns and save - NOTE saving shuffled data
    master_save_path = os.path.join(save_dir, 'Master-Data-Files','master-counterfactual.xlsx')    
    master_data = master_data[['id','context','perturbation','label','model', 'perturbation_model']]
    master_data.sample(frac=1).to_excel(master_save_path, index = False)

    return



def save_learning_data(args, word_sequences, tag_sequences, name2tagger_dict, batch_size, 
                    LIME_explain_fn, Anchor_explain_fn, composite_explain_fn, explainer_obj,
                    condition_names, constrain_to_classes = None):
    '''
    saves learning data for the forward test learning period(s)
    for each condition in condition_names, save explanations from dev data
    we try to control for the quantity of explanation (as opposed to quality) by balancing counts of instances across conditions
    NOTE by construction, blackbox and prototype model will agree on these examples. see get_balanced_data()

    '''
    print("Getting learning data for forward test...")
    
    # unpack tagger
    blackbox_tagger = name2tagger_dict['blackbox']
    prototype_tagger = name2tagger_dict['prototype']
    data_encoder_obj = blackbox_tagger.data_encoder

    # get batch from data
    batches = get_balanced_batches(word_sequences, tag_sequences, name2tagger_dict, batch_size, constrain_to_classes,
        num_batches = 1)    
    batch = batches[0]

    # unpack the batch
    instance_idx = []
    word_sequences = []    
    tags = []
    for (idx, words, tag) in batch:
        instance_idx.append(idx)
        word_sequences.append(words)
        tags.append(tag)
    text_sequences = [data_row_to_str(data_encoder_obj, data_row) for data_row in word_sequences]  
    word_sequences = np.array(word_sequences)

    # get predictions. NOTE by construction, blackbox and prototype model will agree on these examples. see get_balanced_data()
    blackbox_tagger = name2tagger_dict['blackbox']
    predicted_tags = blackbox_tagger.predict_tags_from_words(word_sequences, constrain_to_classes = constrain_to_classes, quiet = True)

    # classifier_fns for lime and anchor
    lime_classifier_fn = blackbox_tagger.predict_probs_from_words
    anchor_classifier_fn = blackbox_tagger.predict_idx_from_words

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

        # condition-specific variables
        # condition-specific variables
        if condition_name == 'prototype':
            for word_sequence in word_sequences:
                explanation = prototype_tagger.explain_instance(word_sequence, counterfactual_method = args.counterfactual_method)
                explanations.append(explanation)
            prototype_explanations = explanations

        elif condition_name == 'LIME':
            for word_sequence in word_sequences:
                explanation = LIME_explain_fn(word_sequence)
                explanations.append(explanation)
            LIME_explanations = explanations

        elif condition_name == 'Anchor':
            for i, word_sequence in enumerate(word_sequences):
                explanation = Anchor_explain_fn(word_sequence)
                explanations.append(explanation)
            Anchor_explanations = explanations

        elif condition_name == 'omission':
            for word_sequence in word_sequences:
                explanation = blackbox_tagger.saliency_explain_instance(word_sequence, counterfactual_method = args.counterfactual_method)
                explanations.append(explanation)    

        elif condition_name == 'decision_boundary':
            for word_sequence in word_sequences:
                explanation = blackbox_tagger.decision_boundary_explanation(word_sequence, sufficient_conditions_print = False)
                explanations.append(explanation)  
            decision_boundary_explanations = explanations

        elif condition_name == 'composite':
            # for word_sequence in word_sequences:
            #     explanation = composite_explain_fn(word_sequence)
            #     explanations.append(explanation)
            for i in range(len(word_sequences)):
                counterfactual_explanation = decision_boundary_last_step(decision_boundary_explanations[i])
                explanation = make_composite_explanation(LIME_explanations[i], 
                                                    Anchor_explanations[i], 
                                                    counterfactual_explanation, 
                                                    prototype_explanations[i])
                explanations.append(explanation)

        # condition-general variables (or, mostly condition-general)
        condition_data_copy = pd.DataFrame.copy(condition_data)
        condition_data_copy['model'] = predicted_tags
         # no explanations for blackbox
        if condition_name == 'blackbox':
            condition_data_copy = condition_data_copy[['id','context','label','model']]            
        else: 
            condition_data_copy['explanation'] = explanations
            condition_data_copy = condition_data_copy[['id','context','label','model','explanation']]            
            condition_data_copy['user_rating'] = ''
        condition_data_save_path = os.path.join(args.save_dir, 'Tests', 'Forward', 'learning-%s.xlsx' % condition_name)
        condition_data_copy.to_excel(condition_data_save_path, index = False)    
    
    return

    
def save_testing_data(args, word_sequences, tag_sequences, name2tagger_dict, batch_size,
                    LIME_explain_fn, Anchor_explain_fn, composite_explain_fn, condition_names, explainer_obj,
                    constrain_to_classes = None):


    print("Getting balanced data...")
    batches = get_balanced_batches(word_sequences, tag_sequences, name2tagger_dict, batch_size, constrain_to_classes,
        num_batches = 2)

    forward_batch = batches[0]
    counterfactual_batch = batches[1]

    # get another, smaller batch for validity. now it does not matter if they see repeats from forward/counterfactual (assuming this test is done last!)
    # validity_batch_size = 20
    # validity_batches = get_balanced_batches(word_sequences, tag_sequences, name2tagger_dict, validity_batch_size, constrain_to_classes,
    #     num_batches = 1)

    # validity_batch = validity_batches[0]

    print('Getting forward data...')
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
        LIME_explain_fn = LIME_explain_fn,
        Anchor_explain_fn = Anchor_explain_fn,
        composite_explain_fn = composite_explain_fn,
        explainer_obj = explainer_obj)

    print('Getting validity data....')
    save_validity_data(args = args,
        batch = counterfactual_batch, 
        name2tagger_dict = name2tagger_dict, 
        constrain_to_classes = constrain_to_classes, 
        save_dir = args.save_dir, 
        condition_names = condition_names,
        LIME_explain_fn = LIME_explain_fn,
        Anchor_explain_fn = Anchor_explain_fn,
        composite_explain_fn = composite_explain_fn)    

    return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to get HIT data')
    parser.add_argument('--data-name', default='adult',
                        help='Train data in format defined by --data-io param.')
    parser.add_argument('--data-io', '-d', default='reviews', help='Data read file format.')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU device number, 0 by default, -1 means CPU.')
    parser.add_argument('--seed-num', type=int, default=1, help='Random seed number, only used when > 0.')
    parser.add_argument('--save-data', type=str2bool, default=False, help='Save a new dataset split.')
    parser.add_argument('--verbose', type=str2bool, default=True, help='Show additional information.', nargs='?',
                    choices=['yes (default)', True, 'no', False])
 
    parser.add_argument('--saliency-type', default='directional', help = "Saliency map methods. Now defunct")
    parser.add_argument('--batch-size', '-b', type=int, default= 32, help='Batch size, samples.')
    parser.add_argument('--save-dir', '-s', default='data/HIT-data-test', help='Path to dir to save the data.')
    parser.add_argument('--counterfactual-method', default = 'conditional_expected_value', help='method for imputing variables for measuring variable importance')

    args = parser.parse_args()
    start_time = time.time()

    # set seeds
    if args.seed_num > 0:
        np.random.seed(args.seed_num)
        torch.manual_seed(args.seed_num)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed_num)

    # print gpu
    print("Working on gpu: %d" % torch.cuda.current_device())

    # make save dirs
    makedir(args.save_dir)
    makedir(os.path.join(args.save_dir,'Tests'))
    makedir(os.path.join(args.save_dir,'Tests','Forward'))
    makedir(os.path.join(args.save_dir,'Tests','Counterfactual'))
    makedir(os.path.join(args.save_dir,'Tests','Validity'))
    makedir(os.path.join(args.save_dir,'Master-Data-Files'))

    # Load data
    data_io = DataIOFactory.create(args)
    dataset, X_train, Y_train, X_dev, Y_dev, X_test, Y_test = data_io.read_train_dev_test(args)

    # Load taggers
    print("Loading models...")
    blackbox_load_name = 'mlp'
    prototype_load_name_1 = 'proto-20-push80'
    prototype_load_name_2 = 'proto-fixed-kmeans-nosep'
    blackbox_path = os.path.join('saved_models','%s.hdf5' % blackbox_load_name)    
    prototype_path_1 = os.path.join('saved_models','%s.hdf5' % prototype_load_name_1)
    prototype_path_2 = os.path.join('saved_models','%s.hdf5' % prototype_load_name_2)
    blackbox_tagger = TaggerFactory.load(blackbox_path, args.gpu)    
    prototype_tagger = TaggerFactory.load(prototype_path_1, args.gpu)   
    fixed_prototype_tagger = TaggerFactory.load(prototype_path_2, args.gpu)  

    # fit imputation models
    blackbox_tagger.fit_imputation_models(dataset, counterfactual_method = args.counterfactual_method) 
    prototype_tagger.feature_id_to_imputation_model = blackbox_tagger.feature_id_to_imputation_model

    # put taggers in dict
    name2tagger_dict = {
        'blackbox' : blackbox_tagger,
        'prototype' : prototype_tagger,
    }

    # create LIME and ANCHOR explainer obj 
    explainer_obj = anchor_tabular.AnchorTabularExplainer(
        dataset.class_names, dataset.feature_names,
        dataset.data, dataset.categorical_names)
    explainer_obj.fit(dataset.train, dataset.labels_train,
                  dataset.validation, dataset.labels_validation)

    # package functions for LIME and Anchor
    threshold = 0.95
    tau = 0.1
    delta = 0.05
    epsilon_stop = 0.05
    batch_size = 100
    Anchor_explain_fn = utils.get_reduced_explain_fn(
            explainer_obj.explain_anchor_str_short, blackbox_tagger.predict_idx_from_words, threshold=threshold,
            delta=delta, tau=tau, batch_size=batch_size / 2,
            sample_whole_instances=True,
            beam_size=10, epsilon_stop=epsilon_stop)
    LIME_explain_fn = utils.get_reduced_explain_fn(
            explainer_obj.explain_lime_str, blackbox_tagger.predict_probs_from_words, num_features=5,
            use_same_dist=True)
    counterfactual_explanation_fn = blackbox_tagger.decision_boundary_explanation
    similar_case_fn = fixed_prototype_tagger.give_similar_case
    composite_explain_fn = get_composite_explain_fn(LIME_explain_fn, Anchor_explain_fn, counterfactual_explanation_fn, similar_case_fn)

    # condition names
    condition_names = ['blackbox', 'prototype' , 'LIME', 'Anchor', 'omission', 'decision_boundary', 'composite']

    # save learning data for forward test
    save_learning_data(args = args, 
                word_sequences = X_dev, 
                tag_sequences = Y_dev,
                name2tagger_dict = name2tagger_dict, 
                batch_size= 16, 
                explainer_obj = explainer_obj,
                condition_names = condition_names,
                Anchor_explain_fn = Anchor_explain_fn,
                LIME_explain_fn = LIME_explain_fn,
                composite_explain_fn = composite_explain_fn)

    # save test data for all tests
    save_testing_data(args = args, 
                word_sequences = X_test, 
                tag_sequences = Y_test,
                name2tagger_dict = name2tagger_dict, 
                batch_size = args.batch_size, 
                explainer_obj = explainer_obj,
                condition_names = condition_names,
                Anchor_explain_fn = Anchor_explain_fn,
                LIME_explain_fn = LIME_explain_fn,
                composite_explain_fn = composite_explain_fn)


    print('done!')
    print('total runtime: %.2f hours' % ((time.time() - start_time)/3600))

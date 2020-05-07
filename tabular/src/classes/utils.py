"""several auxiliary functions"""

import argparse
import datetime
import itertools
import sys
import torch
import os
import numpy as np
import time

def info(t, name=''):
    print(name, '|', t.type(), '|', t.shape)


def flatten(list_in):
    return [list(itertools.chain.from_iterable(list_item)) for list_item in list_in]


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def get_words_num(word_sequences):
    return sum(len(word_seq) for word_seq in word_sequences)


def get_datetime_str():
    d = datetime.datetime.now()
    return '%02d_%02d_%02d_%02d-%02d_%02d' % (d.year, d.month, d.day, d.hour, d.minute, d.second)


def get_sequences_by_indices(sequences, indices):
    return [sequences[i] for i in indices]


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def argsort_sequences_by_lens(list_in):
    data_num = len(list_in)
    sort_indices = argsort([-len(item) for item in list_in])
    reverse_sort_indices = [-1 for _ in range(data_num)]
    for i in range(data_num):
        reverse_sort_indices[sort_indices[i]] = i
    return sort_indices, reverse_sort_indices


def log_sum_exp(x):
    max_score, _ = torch.max(x, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), -1))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_input_arguments():
    return 'python3 main.py ' + ' '.join([arg for arg in sys.argv[1:]])


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)    


def get_user_input():
    confirmed = False
    while not confirmed:
        user_input = input()
        print("\nDid you mean to say: %s" % user_input)
        print("\nPress any key for Yes or type \'no\' for No")
        confirmation = input()
        confirmation = confirmation.lower().strip().strip('\n')
        if confirmation != 'no':
            print("Input confirmed\n")
            confirmed = True
        else:
            print("Can you re-enter your input:")
    return user_input.lower().strip(' ').strip('\n')


def apply_mask(input_tensor, mask_tensor, gpu=-1):
    input_tensor = input_tensor.cuda()
    mask_tensor = mask_tensor.cuda()
    return input_tensor*mask_tensor.unsqueeze(-1).expand_as(input_tensor)

def prediction_type(pos_neg_dict, tag, predicted_tag):
    pos_neg_tag = pos_neg_dict[tag]
    if pos_neg_tag:
        if predicted_tag == tag:
            return 'TP'
        else:
            return 'FP'
    else:
        if predicted_tag == tag:
            return 'TN'
        else:
            return 'FN'

def float_to_signed_str(number):
    return '+%.2f' % number if number > 0 else '%.2f' % number

def _get_variable_value_human_readable(dataset_obj, data_row, col_id):
    data_row = force_array_1d(data_row)        
    variable_value = int(data_row[col_id])
    variable_str = dataset_obj.categorical_names[col_id][variable_value]
    return force_str(variable_str)

def data_row_to_str(dataset_obj, data_row):
    data_point_list_of_str = [
        '%s = %s' % (dataset_obj.feature_names[col_id], _get_variable_value_human_readable(dataset_obj, data_row, col_id))
        for col_id in range(data_row.shape[-1])
    ]
    return '\n'.join(data_point_list_of_str) + '\n'

def two_data_points_human_readable(dataset_obj, X_1, X_2, header = None):
    
    # validate inpute shape
    X_1 = force_array_1d(X_1)
    X_2 = force_array_1d(X_2)

    # local var
    num_features = X_1.shape[-1]

    # start return_string
    return_string = header if header is not None else "Variable | Data point 1 -- Data point 2\n"

    # add values side by side
    for j in range(num_features):
        var_name = dataset_obj.feature_names[j]
        X_1_readable = _get_variable_value_human_readable(dataset_obj, X_1, j)
        X_2_readable = _get_variable_value_human_readable(dataset_obj, X_2, j)

        # add row to string
        return_string += '%s  |  %s   ---   %s \n' % (var_name, X_1_readable, X_2_readable)

    return return_string



def highlight_differences(dataset_obj, X_1, X_2):
    # returns X_2 str formatted with variables differing from word_seq_1 highlighted
    # print line by line with variable name : variable value --> new_var_value

    # validate input shapes
    X_1 = force_array_1d(X_1)
    X_2 = force_array_1d(X_2)

    where_different = (X_1 != X_2)
    where_different_idx = np.argwhere(where_different).reshape(-1)
    where_different_idx = np.sort(where_different_idx)    

    # show before and after of the variable values in the places where the datapoints differ
    data_point_list_of_str = [
        '%s : %s --> %s' % (dataset_obj.feature_names[col_id], 
                            _get_variable_value_human_readable(dataset_obj, X_1, col_id),
                            _get_variable_value_human_readable(dataset_obj, X_2, col_id))        
        for col_id in where_different_idx
    ]
        
    return '\n'.join(data_point_list_of_str)  



def saliency_list_values(dataset_obj, X_dense, importance_metric):
    '''
    word_sequence: list of str tokens
    importance_metric: np float array indicating signed importance per word

    dataset_obj must have attributes feature_names and categorical_names

    limit to top 6 variables
    
    '''

    # validate input shape
    X_dense = force_array_1d(X_dense)

    # num features
    num_features = X_dense.shape[-1]

    # min_size to show
    scale = np.max(np.abs(importance_metric))
    min_size = min(.1,.5*scale) if scale > .1 else .03

    # start list of tuples
    words_and_vals = []

    # highlight words according to quantiles
    for i in range(num_features):
        if abs(importance_metric[i]) >= min_size:
            variable_value = _get_variable_value_human_readable(dataset_obj, X_dense, i)
            words_and_vals.append((variable_value, importance_metric[i]))

    words_and_vals = sorted(words_and_vals, key = lambda tup: tup[1], reverse = True)

    # take only 6 most extreme variables
    if len(words_and_vals) > 6:
        words_and_vals = words_and_vals[:3] + words_and_vals[-3:]

    # str formatting. put a + on the positive values
    var_importance_list_of_str = [
        '%s | %s' % (feature_name, float_to_signed_str(val))
        for feature_name, val in words_and_vals
    ]
        
    var_importance_str = '\n'.join(var_importance_list_of_str)
    return var_importance_str


def force_ascii(word):
    # return ascii version of sentencePiece str
    if ord(word[0]) >= 128:
        return word[1:]
    else:
        return word

def force_str(x):
    try:
        return x.decode()
    except:
        return x

def force_array_1d(data):
    assert np.max(data.shape) == np.prod(data.shape), "Trying to force array to 1d when it has meaningful extra dimensions"
    return data.squeeze()

def force_array_2d(data):
    # forces arrays of shape (d,) to (1,d)
    if len(data.shape) == 1:
        return data.reshape((1,-1))
    else:
        return data


def sample_perturbations(dataset_obj, X_orig_dense, n_samples = 10000):
    '''
    this is a tight perturbation distribution that we write, since the Anchor distribution never samples very close to the original

    assume that every feature is categorical

    distr: sample uniformly a number of edits per instance from 1-3. randomly sample which features and which values to impute

    '''

    # local vars
    num_features = X_orig_dense.shape[-1]
    feature_idx = np.arange(num_features)
    feature_id_to_num_values = {i : len(dataset_obj.categorical_names[i]) for i in feature_idx}
    feature_id_to_possible_values = {i : np.arange(feature_id_to_num_values[i]) for i in feature_idx}

    # validate input shape
    X_orig_dense = force_array_1d(X_orig_dense)

    # new data to be filled in 
    new_data = np.array([X_orig_dense for i in range(n_samples)])

    # edit each datapoint
    for i in range(n_samples):

        # sample the features to change
        n_changes = np.random.choice([1,2,3])
        features_to_change = np.random.choice(feature_idx, size = n_changes, replace = False)

        # edit the chosen features
        for j in features_to_change:
            possible_values = np.setdiff1d(
                feature_id_to_possible_values[j],
                X_orig_dense[j]
            )
            new_data[i,j] = np.random.choice(possible_values)

    return new_data



def nearest_flipping_perturbation(dataset_obj, X_orig_dense, tagger):
    '''
    sample from a perturbation distrbution around word_sequence, and return the nearest perturbation that receives the opposite class prediction
    return that perturbation as a str, along with a str version with the perturbed words wrapped with double asterisks
    '''

    # validate input shape
    X_orig_dense = force_array_2d(X_orig_dense)

    # define distance function to be euclidean distanc between latent representations
    def distance(v,u):
        # squared euclidean distance
        return np.sum((v-u)*(v-u))

    def edit_distance(X_1, X_2, v,u):
        '''
        the point of this is to measure similarity by the number of variables that change, while tie-breaking based on the latent distance
        note inputs are the same size
        '''
        num_different = np.sum(X_1 != X_2)
        euclidean = distance(v,u)
        return num_different + euclidean

    # could define a distance function that penalizes uses of rare data categories?

    # define the get_latent function
    def get_latents(X_dense, model):
        if hasattr(model, 'prototypes'):
            latents, _ = model.push_forward(X_dense)
            latents = latents.detach().cpu().squeeze().numpy()
        else:
            latents = tagger.get_latents(X_dense).detach().cpu().numpy()
        return latents

    # get the original prediction and latent
    original_prediction = np.array(tagger.predict_tags_from_words(X_orig_dense, quiet = True)[0])
    original_latent = get_latents(X_orig_dense, tagger)    

    # sample perturbations using our perturbation distribution
    X_perturbations_dense = sample_perturbations(tagger.data_encoder, X_orig_dense)

    # get model predictions for perturbations
    perturbation_predictions = np.array(tagger.predict_tags_from_words(X_perturbations_dense, quiet = True))
    
    # gather the oppositely-predicted perturbations and compute their latents
    where_diff_prediction = np.argwhere(original_prediction != perturbation_predictions).reshape(-1)
    diff_perturbations = np.array([X_perturbations_dense[idx] for idx in where_diff_prediction])
    latents = get_latents(diff_perturbations, tagger)

    # get distances between original latent and perturbation latents
    latent_distances = np.zeros(latents.shape[0])
    for j in range(latents.shape[0]):
        latent_distances[j] = edit_distance(X_orig_dense, diff_perturbations[j,:], original_latent, latents[j,:])
    
    # pick closest perturbation
    argmin = np.argmin(latent_distances)
    chosen_perturbation = diff_perturbations[argmin,:]
    chosen_perturbation_highlighted = highlight_differences(dataset_obj, X_orig_dense, chosen_perturbation)

    # print("perturbation latent distances: ", np.quantile(latent_distances,(.01,.1,.5,.9)))
    # print(chosen_perturbation)
    # print(X_orig_dense)
    # import ipdb; ipdb.set_trace()

    return chosen_perturbation, chosen_perturbation_highlighted



def get_path_to_decision_boundary(X_orig_dense, tagger):
    '''
    find the nearest perturbation that gets the opposite prediction as the original
    greedily selected the min-evidence-changing single word change to show
    '''
    tagger.eval()

    # validate input shape
    X_orig_dense = force_array_2d(X_orig_dense)

    # find nearest input of the predicted opposite class
    perturbation, perturbation_str_highlighted, = nearest_flipping_perturbation(
                                                                            dataset_obj = tagger.data_encoder,
                                                                            X_orig_dense = X_orig_dense,
                                                                            tagger = tagger)
    
    # original logits
    orig_logits = tagger.get_logits(X_orig_dense).detach().cpu().squeeze().numpy()
    orig_max_logit = np.max(orig_logits)
    pred_class_id = np.argmax(orig_logits)

    # force array shapes
    X_orig_dense = force_array_1d(X_orig_dense)
    perturbation = force_array_1d(perturbation)
    
    # get locations of the differences in the perturbation and the original
    seq_differ = (X_orig_dense != perturbation)
    where_seq_differ = np.argwhere(seq_differ).reshape(-1)

    # init path_sequences and a word_sequence that we'll iteratively edit
    path_sequences = []
    running_perturbation = X_orig_dense.copy()

    # while we don't have all the steps, make one word changes and greedily select the one word change that makes the minimum adjustment in evidence
    current_logit = orig_max_logit
    while len(path_sequences) < len(where_seq_differ):
        
        # create sequences of one_word_changes on the running_perturbation
        one_word_changes = []
        for idx in where_seq_differ:
            
            # copy down perturb_word and the current sequence
            perturb_word = perturbation[idx]
            current_sequence = running_perturbation.copy()
            
            # if already changed, continue
            if current_sequence[idx] == perturb_word:
                continue
            
            # make one step to the current sequence 
            current_sequence[idx] = perturb_word 
            one_word_changes.append(current_sequence)

        # find min step
        one_step_logits = tagger.get_logits(one_word_changes).detach().cpu().numpy()
        abs_diff_in_logits = np.abs(current_logit - one_step_logits[:,pred_class_id].reshape(-1))
        min_step = np.argmin(abs_diff_in_logits)

        # choose sequences
        chosen_one_word_change = one_word_changes[min_step]
        path_sequences.append(chosen_one_word_change.copy())

        # update running_sequences
        running_perturbation = chosen_one_word_change.copy()

        # update logit
        current_logit = one_step_logits[min_step, pred_class_id]

    # get str highlights of steps along the way
    path_sequences_highlighted = [
        highlight_differences(dataset_obj = tagger.data_encoder, X_1 = X_orig_dense, X_2 = perturbation)
        for perturbation in path_sequences
    ]

    return np.array(path_sequences), path_sequences_highlighted


def decision_boundary_last_step(decision_boundary_explanation):
    db_split = decision_boundary_explanation.split('\n\n')
    header = db_split[0].split('\n')[-1]
    last_conditions = db_split[-1].split('\n')[2:]
    explanation = header + '\n' + '\n'.join(last_conditions)
    return explanation


def data_str_to_array(dataset_obj, data_str):
    '''
    for reverse formatting of data_row prints to dense format
    '''
    feature_names = dataset_obj.feature_names
    categorical_names = dataset_obj.categorical_names
    num_features = len(feature_names)

    data_str_list = data_str.split('\n')
    # now split each row at the first equals sign, and look up the variable integer value
    
    data_array = np.zeros(num_features)
    for i in range(num_features):
        row = data_str_list[i]
        equals_index = row.index('=')
        variable_name = row[:equals_index].strip()
        variable_value = row[(equals_index+1):].strip()
        variable_code = np.argwhere(np.array(categorical_names[i]) == variable_value)
        data_array[i] = variable_code
        
    return data_array


def perturb_str_data(dataset_obj, data_orig_str, perturbations_str):
    '''
    for reverse formatting of data_row prints and pertubation prints
    '''

    perturbations_list = perturbations_str.split('\n')
    num_perturbations = len(perturbations_list)
    feature_names = dataset_obj.feature_names
    categorical_names = dataset_obj.categorical_names
    data_orig_array = data_str_to_array(dataset_obj, data_orig_str)
    num_features = len(data_orig_array)

    data_perturb_array = data_orig_array.copy()

    for i in range(num_perturbations):
        cur_perturb = perturbations_list[i].split(':') # split into (variable name, variable edit)
        perturb_var = cur_perturb[0].strip()
        perturb_col_id = feature_names.index(perturb_var)
        new_value = cur_perturb[1].split('-->')[1].strip() # terrible, i know
        new_value_code = np.argwhere(np.array(categorical_names[perturb_col_id]) == new_value)

        data_perturb_array[perturb_col_id] = new_value_code

    return data_perturb_array




def make_composite_explanation(LIME_explanation, Anchor_explanation, counterfactual_explanation, similar_case_explanation):
    return_list = [
        LIME_explanation + \
        '\n----\n' + \
        Anchor_explanation,
        decision_boundary_last_step(counterfactual_explanation) + \
        '\n----\n' + \
        similar_case_explanation
        ]
    return return_list


def fit_composite_explanation(data_row, LIME_explain_fn, Anchor_explain_fn, counterfactual_explain_fn, similar_case_explain_fn):

    # import ipdb; ipdb.set_trace()

    # validate input shape
    data_row = force_array_1d(data_row)

    # compute explanations
    LIME_exp = LIME_explain_fn(data_row)
    Anchor_exp = Anchor_explain_fn(data_row)
    counterfactual_exp = counterfactual_explain_fn(data_row)
    similar_case_explain = similar_case_explain_fn(data_row)

    # join explanations
    explanation = make_composite_explanation(LIME_exp, Anchor_exp, counterfactual_exp, similar_case_explain)

    return explanation


def get_composite_explain_fn(LIME_explain_fn, Anchor_explain_fn, counterfactual_explain_fn, similar_case_explain_fn):
    def explain_fn(data_row):
        return fit_composite_explanation(data_row, LIME_explain_fn, Anchor_explain_fn, counterfactual_explain_fn, similar_case_explain_fn)
    return explain_fn
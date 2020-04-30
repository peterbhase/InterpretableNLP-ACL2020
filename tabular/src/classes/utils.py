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


def highlight_differences(word_seq_1, word_seq_2):
    # returns seq_2 with words differing from word_seq_1 highlighted
    # assume sequences are same length
    if len(word_seq_1) != len(word_seq_2): # different length texts are hard to highlight. this is a rarely occuring bug
        return ' '.join(word_seq_2)
    where_different = (np.array(word_seq_1) != np.array(word_seq_2))
    text = [word for word in word_seq_2]
    for i in range(len(text)):
        if where_different[i]:
            if text[i] == 'fuckwad': # so this was an interesting encounter
                text[i] = '**<expletive>**'
            else:
                text[i] = '**' + text[i] + '**'
    return ' '.join(text)  



def saliency_highlight(word_sequence, importance_metric, saliency_type):
    '''
    word_sequence: list of str tokens
    importance_metric: np float array indicating signed importance per word

    returns: list of "highlighted" str tokens
    
    min-max scale the importance_metric to be in range [-1,1] and highlight accordingly
    '''
    text = [word for word in word_sequence]

    scale = np.max(np.abs(importance_metric))
    scaled_metric = importance_metric / scale

    bins = [-.7, -.2, .2, .7]

    s_up = '^'
    s_down = '⌄'
    s_up2 = s_up*2
    s_down2 = s_down*2

    # highlight words according to quantiles
    for i, word in enumerate(text):
        p = scaled_metric[i]
        if p < bins[0]:
            text[i] = s_down2 + text[i] + s_down2
        elif p >= bins[0] and p < bins[1]:
            text[i] = s_down + text[i] + s_down
        # don't highlight in the middle bin
        elif p >= bins[2] and p < bins[3]:
            text[i] = s_up + text[i] + s_up
        elif p >= bins[3]:
            text[i] = s_up2 + text[i] + s_up2

    return text


def float_to_signed_str(number):
    return '+%.2f' % number if number > 0 else '%.2f' % number


def saliency_list_values(word_sequence, importance_metric, saliency_type, print_flat = False):
    '''
    word_sequence: list of str tokens
    importance_metric: np float array indicating signed importance per word

    limit to top 6 tokens
    
    '''    

    # copy text
    text = [word for word in word_sequence]
    text_np = np.array(text)

    # any duplicated tokens? need to mark these
    token2count = {token : np.sum(text_np == token) for token in text_np}
    count_suffixes = {1 : 'st', 2 : 'nd', 3 : 'rd', 4 : 'th', 5 : 'th', 6 : 'th', 7 : 'th'}
    # for each word, get the order number. count how many before it are equal to it, yield that + 1
    try:
        index_marked_tokens = [token + " (%s%s)" % (np.sum(text_np[:i] == token)+1, count_suffixes[np.sum(text_np[:i] == token)+1]) if token2count[token] > 1 else token
                                    for (i, token) in enumerate(word_sequence)]
    except:
        import ipdb; ipdb.set_trace()


    # min_size to show
    scale = np.max(np.abs(importance_metric))
    min_size = min(.1,.5*scale) if scale > .1 else .03

    # start list of tuples
    words_and_vals = []

    # highlight words according to quantiles
    for i, word in enumerate(text):
        if abs(importance_metric[i]) >= min_size:
            words_and_vals.append((index_marked_tokens[i], importance_metric[i]))

    words_and_vals = sorted(words_and_vals, key = lambda tup: tup[1], reverse = False)

    # take only 6 most extreme tokens.
    if len(words_and_vals) > 6:
        words_and_vals = words_and_vals[:3] + words_and_vals[-3:]
        
    if print_flat:
        importance_str = ' | '.join(['%s  %s' % (word, float_to_signed_str(val)) for (word, val) in words_and_vals])
    else:
        importance_str = '\n'.join(['%s | %s' % (word, float_to_signed_str(val)) for (word, val) in words_and_vals])

    return importance_str



def replace_word(word_sequence, perturb_slot_id, neighbor_obj, tagger_word_dict = None, method = 'unk'):
    '''
    replace the word at word_sequence[perturb_slot_id] with
        a) 'unk' token
        or 
        b) random tokens of the same pos

    if word pos not in replaceable_pos, simply return the sequence back. this works properly in conjunction with counterfactual word importance. the logit_difference will be 0

    return list of lists. just one inside list if 'unk' token
    '''
    new_sequences = []

    if method == 'unk':
        text_copy = [word for word in word_sequence]
        text_copy[perturb_slot_id] = '<unk>'
        new_sequences.append(text_copy)
        return new_sequences

    elif method == 'neighbors':
        replaceable_pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET']
        perturb_word = word_sequence[perturb_slot_id]
        spacy_tokens = neighbor_obj.nlp(' '.join(word_sequence))
        perturb_token = spacy_tokens[perturb_slot_id]

        # how many neighbors to check for each pos? pretty ad hoc
        pos2k = {pos : 20 if pos in ['ADP','DET'] else 150 for pos in replaceable_pos}

        # if the word's pos isn't in replaceable_pos, continue
        if not hasattr(perturb_token,'pos_'):
            return [word_sequence]
        if perturb_token.pos_ not in replaceable_pos:
            return [word_sequence]

        elif method == 'neighbors':
            # take top_k nearest neighbors in embedding space
            top_k = pos2k[perturb_token.pos_]
            neighbors = [x[0].text for x in neighbor_obj.neighbors(perturb_word)][:top_k]

            # if no neighbors for some reason, return the original sequence
            if len(neighbors) == 0:
                return [word_sequence]

            # remove those not in the model's embedding dict, if model provided
            if tagger_word_dict is not None:
                neighbors = [neighbor for neighbor in neighbors if neighbor in tagger_word_dict]
            
            for i, neighbor in enumerate(neighbors):
                text_copy = [word for word in word_sequence]
                text_copy[perturb_slot_id] = neighbor
                new_sequences.append(text_copy)

    return new_sequences
    



def get_counterfactual_words(text, mask_position, language_model, tokenizer, vocab, valid_idx = None,
                                top_k = 100):
    '''
    get distribution over the word at the mask_position in text, conditioned on the remaining words
    - assumes text is str with tokens separated by spaces
    - assumes language_model is XLNet from transformers package
    - vocab must be words in tokenizer that are also in the tagger vocab
    - valid_idx should give valid words to choose from (corresponding to probablities from the distr.)
    return the top-n-samples words of the distribution 
    '''
    
    # replace the mask_position with with <mask>
    word_sequence = text.split()
    word_sequence[mask_position] = '<mask>'

    # prep input to XLNet language_model
    input_text = ' '.join(word_sequence)
    input_ids = torch.LongTensor(tokenizer.encode(input_text)).unsqueeze(0)  # We will predict the masked token
    mask_id = np.argwhere(input_ids[0] == 6).item()
    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
    perm_mask[:, :, mask_id] = 1.0  # Previous tokens don't see last token
    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
    target_mapping[0, 0, mask_id] = 1.0 
    
    #assert input_ids.shape[-1] <= (len(word_sequence) + 1), "Tokenizer broke up the input into more tokens than we're representing"

    # move all to gpu
    input_ids = input_ids.cuda()
    perm_mask = perm_mask.cuda()
    target_mapping = target_mapping.cuda()
    
    outputs = language_model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
    next_token_logits = outputs[0].detach().cpu().squeeze()  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]    
    if valid_idx is not None:
        next_token_logits = next_token_logits[valid_idx]
    probs = torch.nn.functional.softmax(next_token_logits, dim = -1).numpy()
    
    words_and_probs = [(force_ascii(word), prob) for (word, prob) in zip(vocab, probs)]
    words_and_probs_sorted = sorted(words_and_probs, key = lambda tup : tup[1], reverse = True)
    top_words_and_probs = words_and_probs_sorted[:top_k]
    # print("Top %d words account for %.4f of the probability mass" % (top_k, np.sum([tup[1] for tup in words_and_probs_sorted])))

    # print("model input as tokens:")
    # print(' '.join([tokenizer._convert_id_to_token(idx.item()) for idx in input_ids.view(-1)]))

    # print("mask word: ", word_sequence[mask_position])
    # print(top_words_and_probs[:10])

    return top_words_and_probs


def expected_score(word_sequence, mask_position, class_id, tagger, language_model, tokenizer, vocab, valid_idx = None):
    '''
    word_sequence: list of str tokens
    mask_slot: id of token to be masked in text.split()
    score_f: a function that takes in a list of lists of tokens and outputs scores
    language_model: a language model that gives a distribution p(x_i | x_{-i}) for each i in the sequence
    tokenizer: tokenizes the text to the LM's liking

    for the token in the mask_slot, estimates the expected score_f(text) under the distribution predicted by the LM    

    '''
    
    text = ' '.join(word_sequence)
    words_and_probs = get_counterfactual_words(text, mask_position, language_model, tokenizer, vocab, valid_idx = valid_idx, top_k = 300)
    words = [tup[0] for tup in words_and_probs]
    probs = np.array([tup[1] for tup in words_and_probs])

    # create counterfactual sequences
    counterfactual_sequences = []
    for word in words:
        counterfactual_sequence = [token for token in word_sequence]
        counterfactual_sequence[mask_position] = word
        counterfactual_sequences.append(counterfactual_sequence)

    # compute expected logit
    logits = tagger.get_logits(counterfactual_sequences).detach().cpu().numpy()
    class_logits = logits[:,class_id]
    expected_score = np.sum(class_logits * probs)

    return expected_score


def force_ascii(word):
    # return ascii version of sentencePiece str
    if ord(word[0]) >= 128:
        return word[1:]
    else:
        return word


def jaccard_similarity(sequence1, sequence2):
    '''jaccard difference on sets'''
    set1 = set(sequence1); set2 = set(sequence2)
    return len(set1 & set2) / len(set1 | set2)


def sample_capped_edit_distance_id(text, perturbations, pick_idx, cap = 5):
    # try to randomly sample a perturbation with fewer than cap edits
    # if there are non, take the minimum edit distance
        
    def edit_distance(X_1, X_2):
        '''
        the point of this is to measure similarity by the number of words that change
        note inputs are the same size
        '''

        # occasionally these things are different lengths, which is a weird bug
        try:
            num_different = np.sum(np.array(X_1) != np.array(X_2))
        except:
            num_different = 1 / jaccard_similarity(X_1, X_2)

        return num_different

    # split text into lists
    word_sequence = text.split()
    perturbation_seqs = [seq.split() for seq in perturbations]

    # get edit distances for perturbations
    edit_distances = np.array([edit_distance(word_sequence, perturb_seq) for perturb_seq in perturbation_seqs])

    # subset to edit distances that correspond to perturbations that belong in the pick_idx supplised
    eligible_edit_distances = np.array([
        edit_dist for idx, edit_dist in enumerate(edit_distances) if idx in pick_idx
    ])

    # get idx for eligible edits. see function string
    min_edit_dist = np.min(eligible_edit_distances)
    where_min_edit = np.argwhere(eligible_edit_distances == min_edit_dist).reshape(-1)
    under_cap_edits = np.argwhere(eligible_edit_distances <= cap).reshape(-1)
    eligible_edits = under_cap_edits if sum(under_cap_edits) > 0 else where_min_edit

    # sample
    possible_pick_idx = pick_idx[eligible_edits]
    sample_id = np.random.choice(possible_pick_idx)

    return sample_id


def nearest_flipping_perturbation(perturb_sentence, word_sequence, tagger, neighbors_obj, constrain_to_classes = None, num_attempts = 10):
    '''
    sample from a perturbation distrbution around word_sequence, and return the nearest perturbation that receives the opposite class prediction
    return that perturbation as a str, along with a str version with the perturbed words wrapped with double asterisks
    '''

    text = ' '.join(word_sequence)
    perturbation_found = False
    start = time.time()

    # word2vec_format_path = '../conll_achernodub/embeddings/glove.6B.100d.word2vec.txt'
    # gensim_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_format_path)

    def distance(v,u):
        # squared euclidean distance
        return np.sum((v-u)*(v-u))

    def edit_distance(X_1, X_2, v,u):
        '''
        the point of this is to measure similarity by the number of words that change, while tie-breaking based on the latent distance
        note inputs are the same size
        '''

        # occasionally these things are different lengths, which is a weird bug
        try:
            num_different = np.sum(np.array(X_1) != np.array(X_2))
        except:
            print("Different length inputs to edit_distance")
            print(X_1)
            print(X_2)
            num_different = 1 / jaccard_similarity(X_1, X_2)
        euclidean = distance(v,u)
        return num_different + euclidean


    # get the original prediction and latent
    original_prediction = np.array(tagger.predict_tags_from_words([word_sequence], constrain_to_classes=constrain_to_classes, quiet = True)[0])
    if hasattr(tagger, 'prototypes'):
        latent, _ = tagger.push_forward([word_sequence])
        original_latent = latent.detach().cpu().squeeze().numpy()
    else:
        original_latent = tagger.get_latents([word_sequence]).detach().cpu().squeeze().numpy()
    
    # sampling parameters -- these are incremented if no perturbation of the needed pick_type is found
    top_n = 100
    proba_change = .25 - min(.005 * len(word_sequence), .15) # lower bound at .1, and decrease prob from .3 according to length until then
    temperature = .8
    
    for i in range(num_attempts):        
        # sample predictions
        start=time.time()
        perturbations, _, orig_sequence = perturb_sentence(
                                    text = text, 
                                    present = [], 
                                    n = 1000, 
                                    neighbors = neighbors_obj, 
                                    top_n=top_n,
                                    proba_change = proba_change,
                                    temperature = temperature, 
                                    use_proba=False)           
        perturbations = set(perturbations) # remove duplicates
        perturbations = [pert for pert in perturbations if pert != text] # remove unperturbed examples
        perturbations = [combine_contractions(pert) for pert in perturbations] # combine contractions
        end = time.time()

        # get model predictions for perturbations
        perturbation_predictions = np.array(tagger.predict_tags_from_words(perturbations, constrain_to_classes = constrain_to_classes, quiet = True))
        
        # gather the oppositely-predicted perturbations and compute their latents
        where_diff_prediction = np.argwhere(original_prediction != perturbation_predictions).reshape(-1)

        # continue if none found
        if len(where_diff_prediction) == 0:
            print("Could not find perturbation of needed type on sample %d" % i)
            top_n += 10
            proba_change += .05
            temperature += .4
            continue

        diff_perturbations = [perturbations[idx] for idx in where_diff_prediction]
        if hasattr(tagger, 'prototypes'):
            latents, _ = tagger.push_forward(diff_perturbations)
            latents = latents.detach().cpu().numpy()
        else:
            latents = tagger.get_latents(diff_perturbations).detach().cpu().numpy()

        # get distances between original latent and perturbation latents
        latent_distances = np.zeros(latents.shape[0])
        latent_wmds = np.zeros(latents.shape[0])
        for j in range(latents.shape[0]):
            latent_distances[j] = edit_distance(text.split(), diff_perturbations[j].split(), original_latent, latents[j,:])
            # latent_wmds[j] = word_movers_distance(gensim_model, word_sequence, diff_perturbations[j])

        dist_args = np.argsort(latent_distances)
        
        # pick closest perturbation
        argmin = np.argmin(latent_distances)
        chosen_perturbation = diff_perturbations[argmin]
        chosen_perturbation_highlighted = highlight_differences(text.split(),chosen_perturbation.split())
        perturbation_found = True

        # import ipdb; ipdb.set_trace()

        # perturbation found, so break
        break                        

    # print("Finding example of opposite prediction took %.2f seconds" % (time.time() - start))
    assert perturbation_found is not False, "Failed to get a perturbation of the opposite predicted class"

    return chosen_perturbation, chosen_perturbation_highlighted, orig_sequence



def get_path_to_decision_boundary(perturb_sentence, word_sequence, tagger, neighbors_obj, constrain_to_classes = None, num_attempts = 10):
    '''
    find the nearest perturbation that gets the opposite prediction as the original
    greedily selected the min-evidence-changing single word change to show
    '''
    tagger.eval()

    # find nearest input of the predicted opposite clas
    perturbation_text, perturbation_text_highlighted, orig_sequence = nearest_flipping_perturbation(perturb_sentence, word_sequence, 
                                                                                    tagger = tagger, 
                                                                                    neighbors_obj = neighbors_obj)
    
    # original logits
    orig_logits = tagger.get_logits([word_sequence]).detach().cpu().squeeze().numpy()
    orig_max_logit = np.max(orig_logits)
    pred_class_id = np.argmax(orig_logits)
    
    # get locations of the differences in the perturbation and the original
    perturb_sequence = perturbation_text.split()
    try:
        seq_differ = (np.array(word_sequence) != np.array(perturb_sequence))
        where_seq_differ = np.argwhere(seq_differ).reshape(-1)
    except:
        seq_differ = (np.array(orig_sequence) != np.array(perturb_sequence))
        where_seq_differ = np.argwhere(seq_differ).reshape(-1)


    # init path_sequences and a word_sequence that we'll iteratively edit
    path_sequences = []
    path_sequences_highlighted = []
    running_perturbation = [word for word in word_sequence]
    running_perturbation_highlighted = [word for word in word_sequence]

    # while we don't have all the steps, make one word changes and greedily select the one word change that makes the minimum abs. adjustment in evidence
    current_logit = orig_max_logit
    while len(path_sequences) < len(where_seq_differ):
        
        # create sequences of one_word_changes on the running_perturbation
        one_word_changes = []
        one_word_changes_highlighted = []
        for idx in where_seq_differ:
            
            # copy down perturb_word and the current sequence
            perturb_word = perturb_sequence[idx]
            current_sequence = [word for word in running_perturbation]
            current_sequence_highlighted = [word for word in running_perturbation_highlighted]
            
            # if already changed, continue
            if current_sequence[idx] == perturb_word:
                continue
            
            # make one step to the current sequence 
            current_sequence[idx] = perturb_word 
            current_sequence_highlighted[idx] = "**" + perturb_word + "**"

            one_word_changes.append(current_sequence)
            one_word_changes_highlighted.append(current_sequence_highlighted)

        # find min step
        one_step_logits = tagger.get_logits(one_word_changes).detach().cpu().numpy()
        abs_diff_in_logits = np.abs(current_logit - one_step_logits[:,pred_class_id].reshape(-1))
        min_step = np.argmin(abs_diff_in_logits)

        # choose sequences
        chosen_one_word_change = one_word_changes[min_step]
        chosen_one_word_change_highlighted = one_word_changes_highlighted[min_step]
        path_sequences.append([word for word in chosen_one_word_change])
        path_sequences_highlighted.append([word for word in chosen_one_word_change_highlighted])

        # update running_sequences
        running_perturbation = [word for word in chosen_one_word_change]
        running_perturbation_highlighted = [word for word in chosen_one_word_change_highlighted]

        # update logit
        current_logit = one_step_logits[min_step, pred_class_id]



    return path_sequences, path_sequences_highlighted


def decision_boundary_last_step(decision_boundary_explanation):
    db_split = decision_boundary_explanation.split('\n\n')
    header = db_split[0].split('\n')[-1]
    last_conditions = decision_boundary_explanation.split('----')[-1]
    explanation = header + '\n' + last_conditions
    return explanation



def combine_contractions(word_sequence):
    # pick out contractions in a sequence and combine them, of the following kind
    # - I 've
    # - I 'm
    # these are coded in the tagger but for some reason perturb_sentence breaks them into ' ve and ' m (separate tokens)
    if type(word_sequence) is str:
        word_sequence = word_sequence.split()

    new_sequence = []
    for i, word in enumerate(word_sequence):
        if word == "\'":
            if i != len(word_sequence) - 1:
                if word_sequence[i+1] == 've' or word_sequence[i+1] =='m':
                    combine = word + word_sequence[i+1]
                    new_sequence.append(combine)
                    continue
        if i > 0:
            if (word == 've' or word == 'm') and word_sequence[i-1] == "\'":
                continue

        new_sequence.append(word)

    return ' '.join(new_sequence)



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


def fit_composite_explanation(data_row, LIME_explain_fn, Anchor_explain_fn, counterfactual_explain_fn, similar_case_explain_fn, neighbors_obj):

    # compute explanations
    LIME_exp = LIME_explain_fn(data_row)
    Anchor_exp = Anchor_explain_fn(data_row)
    counterfactual_exp = counterfactual_explain_fn(data_row, neighbors_obj)
    similar_case_explain = similar_case_explain_fn(data_row)

    # join explanations
    explanation = make_composite_explanation(LIME_exp, Anchor_exp, counterfactual_exp, similar_case_explain)

    return explanation


# get one argument LIME
def get_LIME_fn(LIME, tagger):
    def LIME_fn(data_row):
        return LIME.safe_explain_instance(data_row, classifier_fn = tagger.predict_probs_from_words, num_features = 5)
    return LIME_fn

# get one argument Anchor
def get_Anchor_fn(Anchor, tagger, args, short_exp = True):
    def Anchor_fn(data_row):
        return Anchor.safe_explain_instance(data_row, classifier_fn = tagger.predict_idx_from_words, threshold = args.threshold,
                                            short_exp = short_exp)
    return Anchor_fn

# get one argument omission fn from blackbox_tagger
def get_omission_fn(tagger, neighbors_obj):
    def omission_fn(data_row):
        return tagger.explain_instance(data_row, neighbors_obj)
    return omission_fn

def get_decision_boundary_fn(tagger, neighbors_obj):
    def decision_boundary_fn(data_row):
        return tagger.decision_boundary_explanation(data_row, neighbors_obj, sufficient_conditions_print = False)
    return decision_boundary_fn


def get_composite_explain_fn(args, tagger, LIME, Anchor, similar_case_explain_fn, neighbors_obj):
    # returns a composite explain fn that takes a single str argument

    LIME_fn = get_LIME_fn(LIME, tagger)
    Anchor_fn = get_Anchor_fn(Anchor, tagger, args)
  
    # get one argument composite explain fn
    def explain_fn(data_row):
        return fit_composite_explanation(data_row = data_row, 
                                        LIME_explain_fn = LIME_fn, 
                                        Anchor_explain_fn = Anchor_fn, 
                                        counterfactual_explain_fn = tagger.decision_boundary_explanation, 
                                        similar_case_explain_fn = similar_case_explain_fn, 
                                        neighbors_obj = neighbors_obj)

    return explain_fn
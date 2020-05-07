"""abstract base class for all types of taggers"""
import math
import torch
import torch.nn as nn
import os
import copy
import time
import numpy as np
from src.classes.utils import *
import sklearn


class TaggerBase(nn.Module):
    """TaggerBase is an abstract class for tagger models. It implements the tagging functionality for
    different types of inputs (sequences of tokens, sequences of integer indices, tensors). Auxiliary class
    SequencesIndexer is used for input and output data formats conversions. Abstract method `forward` is used in order
    to make these predictions, it has to be implemented in the model classes."""
    def __init__(self, data_encoder, tag_seq_indexer, gpu, batch_size):
        super(TaggerBase, self).__init__()
        self.data_encoder = data_encoder
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

    def dense2sparse(self, X_dense):
        X_sparse = self.data_encoder.transform(X_dense)
        X_sparse_torch = self.tensor_ensure_gpu(torch.Tensor(X_sparse.toarray()))
        return X_sparse_torch

    def predict_idx_from_words(self, X_dense, numpy = True):

        # set tagger to eval and validate input shape
        self.eval()
        X_dense = force_array_2d(X_dense)

        outputs_tensor = self.forward(X_dense) # batch_size x num_class
        predicted_idx = []
        
        for k in range(len(X_dense)):
            curr_output = outputs_tensor[k, :]
            prediction = curr_output.argmax(dim=0).item()                
            predicted_idx.append(prediction)        
        
        return predicted_idx if not numpy else np.array(predicted_idx)

    def inv_predict_idx_from_words(self, X_dense, numpy = True):

        # set tagger to eval and validate input shape
        self.eval()
        X_dense = force_array_2d(X_dense)

        outputs_tensor = self.forward(X_dense) # batch_size x num_class
        predicted_idx = []
        
        for k in range(len(X_dense)):
            curr_output = outputs_tensor[k, :]
            prediction = 1 - curr_output.argmax(dim=0).item()                
            predicted_idx.append(prediction)        
        
        return predicted_idx if not numpy else np.array(predicted_idx)


    def predict_probs_from_words(self, X_dense):
        # returns numpy array of predicted probs, for use in LIME
        
        # set tagger to eval and validate input shape
        self.eval()
        X_dense = force_array_2d(X_dense)

        outputs_tensor = self.forward(X_dense) # batch_size x num_class
        predicted_probs = []

        for k in range(len(X_dense)):
            curr_output = outputs_tensor[k, :]
            prediction = curr_output.argmax(dim=0).item()                            
            probs = torch.nn.functional.softmax(curr_output, dim=0).detach().cpu().numpy()
            predicted_probs.append(probs)

        return np.array(predicted_probs)            


    def predict_tags_from_words(self, X_dense, batch_size=-1, constrain_to_classes = None, quiet = False):
        '''
        they use .extend because curr_output_tag_sequences is a list of lists, so what gets appended is just the list of tags
        word_sequences must be list of lists of tokens

        constrain_to_classes is defunct
        '''
        
        # set tagger to eval and validate input shape
        self.eval()
        if type(X_dense) is list: X_dense = np.array(X_dense)
        X_dense = force_array_2d(X_dense)

        if batch_size == -1:
            batch_size = self.batch_size
        if not quiet:
            print('\n')
        batch_num = math.floor(len(X_dense) / batch_size)
        if len(X_dense) > 0 and len(X_dense) < batch_size:
            batch_num = 1
        output_tag_sequences = list()
        for n in range(batch_num):
            i = n*batch_size
            if n < batch_num - 1:
                j = (n + 1)*batch_size
            else:
                j = len(X_dense)
            curr_output_idx = self.predict_idx_from_words(X_dense[i:j], numpy = False)
            curr_output_tag_sequences = self.tag_seq_indexer.idx2items(curr_output_idx)
            output_tag_sequences.extend(curr_output_tag_sequences)
            if math.ceil(n * 100.0 / batch_num) % 25 == 0 and not quiet:
                print('\r++ predicting, batch %d/%d (%1.2f%%).' % (n + 1, batch_num, math.ceil(n * 100.0 / batch_num)),
                      end='', flush=True)
            # not_O = sum([sum([x != 2 for x in sent]) for sent in curr_output_idx2])
            # print(not_O)
        return output_tag_sequences



    def saliency_map(self, X_dense, counterfactual_method = 'mean', scaling_factor = 1):
        '''

        '''

        # set tagger to eval and validate input shape
        self.eval()
        X_dense = force_array_2d(X_dense)
        
        # forward pass
        logits = self.get_logits(X_dense)
        selected_logit = torch.max(logits) 
        selected_logit = selected_logit.detach().cpu()

        # get class ids for prediction, class to explain, and neg_class
        predicted_tag = self.predict_tags_from_words(X_dense, quiet = True)[0]
        neg_class_id = self.tag_seq_indexer.item2idx_dict['below $50K']
        explain_class_id = torch.argmax(logits.view(-1)).item()

        # obtain importance measure as a difference between the selected class logit for the original input and some counterfactual class logit
        logit_differences = np.zeros(X_dense.shape[1])
        for impute_id in range(X_dense.shape[-1]):
    
            # note depending on counterfactual_method, counterfactual_data could be a single datapoint or many datapoints
            counterfactual_data, prob_weights = self.impute_data(data_row = X_dense, impute_id = impute_id, counterfactual_method = counterfactual_method)
            counterfactual_logits = self.get_logits(counterfactual_data).detach().cpu().numpy()
            class_counterfactual_logits = counterfactual_logits[:,explain_class_id]
            counterfactual_evidence = np.average(class_counterfactual_logits.reshape(-1), 
                                            axis = 0,
                                            weights = prob_weights.reshape(-1))
            logit_differences[impute_id] = selected_logit - counterfactual_evidence

        # set importance metric
        importance_metric = logit_differences

        # quick fix so that saliency maps are consistently directional between classes.
        if explain_class_id == neg_class_id:
            importance_metric = -importance_metric   

        # scale for readability
        importance_metric = scaling_factor * importance_metric

        importance_str = saliency_list_values(self.data_encoder, X_dense, importance_metric)

        return importance_str


    def impute_data(self, data_row, impute_id, counterfactual_method):
        '''
        method is method for variable importance, used in self.saliency_map

        example: if feature 3 has 5 possible values, and impute_id == 3, then this returns a dataset of 5 data points, each possible version of the original data_row
        with feature 3 set to each of the possible values. it also returns the probabilities of each corresponding value of feature 3, obtained according to the imputation method

        see fit_imputation_models for imputation details
        '''
        assert hasattr(self, 'feature_id_to_imputation_model'), "Need to execute self.fit_imputation_models first"
        # assert self.imputation_method == counterfactual_method, "Requesting to use imputation method which this model doesn't have imputation models for"

        # validate input shape
        data_row = force_array_2d(data_row)

        new_data = data_row.copy()
        model = self.feature_id_to_imputation_model[impute_id]

        if counterfactual_method == 'mean':

            # impute with the argmax. consider to have prob 1
            X_fit = np.zeros((1,1))
            new_data[0,impute_id] = model.predict(X_fit)
            probs = np.ones((1,1))

        elif counterfactual_method == 'expected_value':
            
            # possible variable values
            possible_values = np.arange(len(self.data_encoder.categorical_names[impute_id])) # possible integer features values to impute
            num_values = len(possible_values)

            # impute data
            new_data = np.array([data_row.copy() for i in range(num_values)]).squeeze()
            new_data[:, impute_id] = possible_values

            # likelihood of each imputation. only need one "data point" to get empirical distribution
            X_fit = np.zeros((1,1)) # 
            probs = model.predict_proba(X_fit)

        elif counterfactual_method == 'conditional_expected_value':

            # possible variable values
            possible_values = np.arange(len(self.data_encoder.categorical_names[impute_id])) # possible integer features values to impute
            num_values = len(possible_values)

            # impute data
            new_data = np.array([data_row.copy() for i in range(num_values)]).squeeze()
            new_data[:, impute_id] = possible_values

            # likelihood of each imputation, conditioned on remaining covariates
            X_minus_j = np.delete(data_row.copy(), obj = impute_id, axis = 1)
            X_sparse = model.data_encoder.transform(X_minus_j)
            probs = model.predict_proba(X_sparse)

        return new_data, probs


    def fit_imputation_models(self, dataset_obj, counterfactual_method = 'conditional_expected_value'):
        '''
        fits imputation model for each feature, to obtain variable importance values
        each model will have a .predict function that takes a np array as an argument

        mean: impute with majority value
        expected_value: return n samples with data imputed from empirical distribution
        conditional_expected_value: impute with the value predicted from the other covariates using a model

        NOTE that we are imputing the sparse data matrix, rather than dense. categorical values can take real valued imputations
        '''

        assert not hasattr(self, 'feature_id_to_imputation_model'), "Already fit imputation models. Only fit these once"

        self.feature_id_to_imputation_model = {}
        self.imputation_method = counterfactual_method
        num_features = len(dataset_obj.categorical_names)
        num_rows = dataset_obj.train.shape[0]

        # the model will simply output the argmax probability of each class, or be used for sampling from the empirical distribution
        if counterfactual_method == 'mean' or counterfactual_method == 'expected_value':

            for j, feature_name in enumerate(dataset_obj.feature_names):
                
                # fit model
                Y = dataset_obj.train[:,j]
                X_fit = np.zeros((num_rows,1))
                model = sklearn.linear_model.LogisticRegression(C=1e9, multi_class = 'multinomial', solver = 'lbfgs')
                model.fit(X_fit, Y)
                self.feature_id_to_imputation_model[j] = copy.deepcopy(model)


        elif counterfactual_method == 'conditional_expected_value':

            for j, feature_name in enumerate(dataset_obj.feature_names):

                # get data. column indexing
                X_minus_j = np.delete(dataset_obj.train, obj = j, axis = 1)
                X_j = dataset_obj.train[:,j]
                
                # get onehot data_encoder
                orig_cat_idx = [idx for idx in range(num_features) if idx != j]
                n_values = [len(dataset_obj.categorical_names[idx]) for idx in orig_cat_idx]
                data_encoder = sklearn.preprocessing.OneHotEncoder(
                    categorical_features=range(num_features-1),
                    n_values=n_values)
                data_encoder.fit(X_minus_j)

                # transform covariates
                X_sparse = data_encoder.transform(X_minus_j)

                # fit model
                model = sklearn.linear_model.LogisticRegression(C=1e3, multi_class = 'multinomial', solver = 'lbfgs')
                model.fit(X_sparse, X_j)

                # attach data_encoder to the model
                model.data_encoder = data_encoder 

                # store model
                self.feature_id_to_imputation_model[j] = copy.deepcopy(model)


        # check accuracies
        if counterfactual_method == 'conditional_expected_value':
            print("Imputation model accuracies:\n")

            majority_accs = []
            val_accs = []

            for j, feature_name in enumerate(dataset_obj.feature_names):

                # get classes, X, Y, and preds
                model = self.feature_id_to_imputation_model[j]
                possible_classes = np.arange(len(dataset_obj.categorical_names[j]))
                Y = dataset_obj.train[:,j]
                X_fit = model.data_encoder.transform(np.delete(dataset_obj.train, obj = j, axis = 1)) if counterfactual_method == 'conditional_expected_value' \
                        else np.zeros((len(Y),1))
                preds = model.predict(X_fit)

                # print train accuracies
                print("\nFeature: %s" % feature_name)
                print("Train baseline_acc: %.2f  |  Train model_acc: %.2f" % (
                    np.max([np.mean(Y == class_id) for class_id in possible_classes]), # max proportion class
                    np.mean(preds == Y) # accuracy
                ))

                # get valid data and preds
                Y = dataset_obj.validation[:,j]
                X_fit = model.data_encoder.transform(np.delete(dataset_obj.validation, obj = j, axis = 1)) if counterfactual_method == 'conditional_expected_value' \
                        else np.zeros((len(Y),1))
                preds = model.predict(X_fit)

                # print valid accsd
                print("Valid baseline_acc: %.2f  |  Valid model_acc: %.2f" % (
                    np.max([np.mean(Y == class_id) for class_id in possible_classes]), # max proportion class
                    np.mean(preds == Y) # accuracy
                ))

                majority_accs.append(np.max([np.mean(Y == class_id) for class_id in possible_classes]))
                val_accs.append(np.mean(preds==Y))

            # would be nice to print the avg majority-baseline acc here
            print("\nAvg majority class proportion: %.2f" % np.mean(majority_accs))
            print("Avg validation acc: %.2f" % np.mean(val_accs))




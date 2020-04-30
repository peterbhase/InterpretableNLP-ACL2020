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
from anchor import anchor_tabular
from anchor.utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run trained model')
    parser.add_argument('--load-name', help='Path to load from the trained model.',
                        default='mlp')
    parser.add_argument('--data-name', default='adult',
                        help='Train data in format defined by --data-io param.')   
    parser.add_argument('--input', help='Input CoNNL filename.',
                        default='data/NER/CoNNL_2003_shared_task/test.txt')
    parser.add_argument('--output', '-o', help='Output JSON filename.',
                        default='out.json')
    parser.add_argument('--data-io', '-d', default='adult', help='Data read file format.')
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

    # Create DataIO object
    data_io = DataIOFactory.create(args)
    
    # Load text data as lists of lists of words (sequences) and corresponding list of lists of tags
    data_io = DataIOFactory.create(args)
    dataset, X_train, Y_train, X_dev, Y_dev, X_test, Y_test = data_io.read_train_dev_test(args)


    # fit imputation models
    # import ipdb; ipdb.set_trace()     
    # tagger.fit_imputation_models(dataset, counterfactual_method = 'conditional_expected_value')

    # sklearn baselines
    explainer = anchor_tabular.AnchorTabularExplainer(
        dataset.class_names, dataset.feature_names,
        dataset.data, dataset.categorical_names)
    explainer.fit(dataset.train, dataset.labels_train,
                  dataset.validation, dataset.labels_validation)

    encode_fn = explainer.encoder.transform

    import sklearn; import sklearn.neural_network
    import xgboost
    print("Logistic Regression baseline")
    model = sklearn.linear_model.LogisticRegression(C=1e2, solver = 'lbfgs', max_iter = 300)
    model.fit(encode_fn(X_train),Y_train)
    print("Train acc: %.3f" % model.score(encode_fn(X_train), Y_train))
    print("Test acc:  %.3f" % model.score(encode_fn(X_test), Y_test))

    print("XGboost")
    model = xgboost.XGBClassifier(n_estimators=400, nthread=10, seed=1)
    model.fit(encode_fn(X_train),Y_train)
    print("Train acc: %.3f" % model.score(encode_fn(X_train), Y_train))
    print("Test acc:  %.3f" % model.score(encode_fn(X_test), Y_test))

    print("sklearn neural net")
    model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(50,50), max_iter = 300)
    model.fit(encode_fn(X_train),Y_train)
    print("Train acc: %.3f" % model.score(encode_fn(X_train), Y_train))
    print("Test acc:  %.3f" % model.score(encode_fn(X_test), Y_test))

    # data_row = X_train[0]

    # predict_proba_fn = tagger.predict_probs_from_words
    # sample_fn, mapping = explainer.get_sample_fn(data_row, predict_proba_fn,
    #                                        sample_whole_instances=True,
    #                                        encode_before_forward=False)
    # raw, data, _ = sample_fn([], 5000, False)
    
    # DatasetsBank provides storing the different dataset subsets (train/dev/test) and sampling batches
    datasets_bank = DatasetsBankFactory.create(args)
    datasets_bank.add_train_sequences(X_train, Y_train)
    datasets_bank.add_dev_sequences(X_dev, Y_dev)
    datasets_bank.add_test_sequences(X_test, Y_test)

    # Create evaluator
    evaluator = EvaluatorFactory.create(args)

    train_score, dev_score, test_score, test_msg = evaluator.get_evaluation_score_train_dev_test(tagger,
                                                                                                 datasets_bank,
                                                                                                 batch_size=1)
    print('\n train / dev / test | %1.2f / %1.2f / %1.2f.' % (train_score, dev_score, test_score))  
    print(test_msg)

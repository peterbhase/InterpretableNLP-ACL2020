from __future__ import print_function
from math import ceil, floor
import os
from os.path import isfile
import time
import numpy as np
import torch.nn as nn
import sklearn, sklearn.neural_network
from src.classes.report import Report
from src.classes.utils import *
from src.factories.factory_data_io import DataIOFactory
from src.factories.factory_datasets_bank import DatasetsBankFactory
from src.factories.factory_evaluator import EvaluatorFactory
from src.factories.factory_optimizer import OptimizerFactory
from src.factories.factory_tagger import TaggerFactory
from src.seq_indexers.seq_indexer_tag import SeqIndexerTag
from src.seq_indexers.seq_indexer_word import SeqIndexerWord


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning tagger using neural networks')
    parser.add_argument('--data-name', default='adult',
                        help='Train data in format defined by --data-io param.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number, -1  means CPU.')
    parser.add_argument('--model', help='Model type.', choices=['MLP', 'ProtoMLP'],
                        default='MLP')
    parser.add_argument('--load-name', '-l', default=None, help='name of model to be loaded.')
    parser.add_argument('--save-dir', '-s', default='saved_models',
                        help='Path to dir to save the trained model.')
    parser.add_argument('--save-name', default='%s_tagger' % get_datetime_str(),
                        help='Save name for trained model.')
    parser.add_argument('--epoch-num', '-e',  type=int, default=300, help='Number of epochs.')
    parser.add_argument('--min-epoch-num', '-n', type=int, default=1, help='Minimum number of epochs.')
    parser.add_argument('--patience', type=int, default=150, help='Patience for early stopping.')
    parser.add_argument('--evaluator', '-v', default='token-acc', help='Evaluation method.')
    parser.add_argument('--save-best', type=str2bool, default=True, help = 'Save best on dev model as a final model.',
                        nargs='?', choices=['yes', True, 'no (default)', False])
    parser.add_argument('--dropout-ratio', '-r', type=float, default=0, help='Dropout ratio.')
    parser.add_argument('--batch-size', '-b', type=int, default=1000, help='Batch size, samples.')
    parser.add_argument('--opt', '-o', help='Optimization method.', choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--lr-decay', type=float, default=0, help='Learning decay rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Learning momentum rate.')
    parser.add_argument('--dataset-sort', type=str2bool, default=False, help='Sort sequences by length for training.',
                        nargs='?', choices=['yes', True, 'no (default)', False])
    parser.add_argument('--seed-num', type=int, default=10, help='Random seed number, only used when > 0.')
    parser.add_argument('--cross-folds-num', type=int, default=-1,
                        help='Number of folds for cross-validation (optional, for some datasets).')
    parser.add_argument('--cross-fold-id', type=int, default=-1,
                        help='Current cross-fold, 1<=cross-fold-id<=cross-folds-num (optional, for some datasets).')
    parser.add_argument('--verbose', type=str2bool, default=True, help='Show additional information.', nargs='?',
                        choices=['yes (default)', True, 'no', False])
    parser.add_argument('--input-dim', type=int, default=66, help='Input dim of encoded data')
    parser.add_argument('--hidden-dim', type=int, default=50, help='Input dim of encoded data')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='coef for l2 reg on weights')
    parser.add_argument('--num-prototypes-per-class', '-p', type=int, default=20, help='Number of prototypes per class for prototype layers')
    parser.add_argument('--proto-dim', type=int, default=16, help='latent space dimension for prototypes')
    parser.add_argument('--pre-eval', type=str2bool, default=True, help='Eval on test set before training')
    parser.add_argument('--pretrained-model', type=str, default=None, help='name of pretrained model used to init models')
    parser.add_argument('--no-eval-til', type=int, default=0, help='No evaluation until the epoch supplied')
    parser.add_argument('--latent-dim', type=int, default=None, help='Dimension reduction before classification for black-box')
    parser.add_argument('--unfreeze-lin-layer', type=int, default=1000, help='Unfreeze lin layer in proto model after this many epochs')
    parser.add_argument('--unfreeze-feature-extractor', type=int, default=1000, help='Unfreeze early layers in proto model after this many epochs')
    parser.add_argument('--start-epoch', type=int, default=0, help='Unfreeze lin layer after this many epochs')
    parser.add_argument('--kmeans-init', type=str2bool, default=True, help='Logical for initializing prototypes via kmeans on the latent vectors of the blackbox.')
    parser.add_argument('--push-after', type=int, default=20, help='Epoch after which there is a push every epoch')
    parser.add_argument('--max-pool-protos', type=str2bool, default=True, help='Logical for max pooling similarity scores within each class.')
    parser.add_argument('--lambda-sep', type=float, default=1e-1, help='Coefficient of separation cost term')
    parser.add_argument('--pooling-type', type=str, default='attention', help='pooling type for context embeddings')
    parser.add_argument('--save-data', type=str2bool, default=False, help='Save a new dataset split.')
    parser.add_argument('--similarity-epsilon', type=float, default=1e-2, help='Epsilon for similarity function')
    parser.add_argument('--hadamard-importance', type=str2bool, default=False, help='If true, the proto_layer outputs similarity_scores*importance_weights (hadamard).')
    parser.add_argument('--similarity-function-name', type=str, default='gaussian',
                            choices = ['log_inv_distance','gaussian'], help='pooling type for context embeddings')

    args = parser.parse_args()

    if args.seed_num > 0:
        np.random.seed(args.seed_num)
        torch.manual_seed(args.seed_num)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed_num)

    print("Working on gpu %d" % torch.cuda.current_device())
    
    # Load data. X are dense numpy arrays, and Y are list of str class names
    data_io = DataIOFactory.create(args)
    dataset, X_train, Y_train, X_dev, Y_dev, X_test, Y_test = data_io.read_train_dev_test(args)

    # import ipdb; ipdb.set_trace()
    
    # DatasetsBank provides storing the different dataset subsets (train/dev/test) and sampling batches
    datasets_bank = DatasetsBankFactory.create(args)
    datasets_bank.add_train_sequences(X_train, Y_train)
    datasets_bank.add_dev_sequences(X_dev, Y_dev)
    datasets_bank.add_test_sequences(X_test, Y_test)

    # make sklearn OneHotEncoder for data and add some attributes to it -- this is passed to the models
    cat_names = sorted(dataset.categorical_names.keys())
    n_values = [len(dataset.categorical_names[i]) for i in cat_names]
    data_encoder = sklearn.preprocessing.OneHotEncoder(
        categorical_features=cat_names,
        n_values=n_values)
    data_encoder.fit(dataset.data)

    # add dataset info to data_encoder
    data_encoder.feature_names = dataset.feature_names
    data_encoder.categorical_names = dataset.categorical_names

    # Tag_seq_indexer converts lists of lists of tags to lists of lists of integer indices and back
    tag_seq_indexer = SeqIndexerTag(gpu=args.gpu)
    tag_seq_indexer.load_items_from_tag_sequence(dataset.class_names)
    
    # Create or load pre-trained tagger
    if args.load_name is None:        
        tagger = TaggerFactory.create(args, data_encoder, tag_seq_indexer)
        start_epoch = args.start_epoch
    else:        
        load_path = os.path.join('saved_models','%s.hdf5' % args.load_name)
        print("Loading model from %s" % load_path)
        tagger = TaggerFactory.load(load_path, args.gpu)

        report_path = os.path.join('saved_models','%s-report.txt' % args.load_name)
        start_epoch = args.start_epoch

    # init proto model layers
    if args.pretrained_model is not None and args.load_name is None:
        pretrained_path = os.path.join('saved_models','%s.hdf5' % args.pretrained_model)
        tagger.initialize_from_pretrained(pretrained_path)

    # init prototypes empirically via k-means
    if args.kmeans_init and hasattr(tagger, 'prototypes'):     
        tagger.initialize_prototypes_empirical(X_train, Y_train)
        tagger.push(args, dataset, X_train, Y_train, save_prototype_files = False)
    
    # Create evaluator
    evaluator = EvaluatorFactory.create(args)
    
    # Create optimizer
    optimizer, scheduler = OptimizerFactory.create(args, tagger)    
    
    # Prepare report and temporary variables for "save best" strategy
    report_fn = os.path.join('saved_models','%s_report.txt' % args.save_name)
    report = Report(report_fn, args, score_names=('train loss', '%s-train' % args.evaluator, '%s-dev' % args.evaluator, '%s-test' % args.evaluator,
                                                                'cross_ent','l1-pen','sep_loss'))
    # Initialize training variables
    iterations_num = floor(datasets_bank.train_data_num / args.batch_size)
    best_dev_score = -1
    best_epoch = -1
    best_test_score = -1
    best_test_msg = 'N/A' # was 'N\A'
    patience_counter = 0

    # Evaluate tagger   
    if args.pre_eval:# or args.load_name is not None:
        train_score, dev_score, test_score, test_msg = evaluator.get_evaluation_score_train_dev_test(tagger,
                                                                                                     datasets_bank,
                                                                                                     batch_size=100)
        report.write_msg(test_msg)
        print(test_msg)
        best_dev_score = dev_score
        best_epoch = start_epoch - 1

    # import ipdb; ipdb.set_trace()
    print('\nStart training...\n')
    for epoch in range(start_epoch+1, args.epoch_num + 1):
        time_start = time.time()
        loss_sum = 0
        cross_ent_sum = 0
        l1_pen_sum = 0
        sep_loss_sum = 0        
        if hasattr(tagger, 'prototypes'):
            push_dev_losses = []  
            tagger.freeze_unfreeze_parameters(epoch,args)          
            tagger.pushed = False
        num_batches = datasets_bank.train_data_num // args.batch_size # last partial batch is dropped
        tagger.train()        

        if args.lr_decay > 0:
            scheduler.step(epoch) # .step() does not yield the correct LR when resuming training on a previously trained model (i.e. starting at >0 epoch)

        for i, (word_sequences_train_batch, tag_sequences_train_batch) in \
                enumerate(datasets_bank.get_train_batches(args.batch_size)):
            tagger.train()
            tagger.zero_grad()            
            # compute loss
            if not hasattr(tagger,'prototypes'):
                loss = tagger.get_loss(word_sequences_train_batch, tag_sequences_train_batch)
            else:
                outputs_tensor_train_batch_one_hot, distances = tagger.get_logprobs_and_distances(word_sequences_train_batch)
                targets_tensor_train_batch = tagger.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
                cross_entropy = tagger.nll_loss(outputs_tensor_train_batch_one_hot, targets_tensor_train_batch)
                
                sep_loss = tagger.get_sep_loss(distances, word_sequences_train_batch, targets_tensor_train_batch)
                l1_pen = tagger.get_lin_layer_l1()
                loss = cross_entropy + 1/10 * l1_pen + args.lambda_sep * sep_loss

            loss.backward()            
            optimizer.step()

            # batch stats tracking
            loss_sum += loss.item()
            if hasattr(tagger, 'prototypes'):
                cross_ent_sum += cross_entropy.item()
                l1_pen_sum += l1_pen.item()
                sep_loss_sum += sep_loss.item()
            if ceil(i*100.0/iterations_num) % 10 == 0:
                print('\r-- train epoch %d/%d, batch %d/%d (%1.2f%%), avg_loss = %1.2f' % (epoch, args.epoch_num,
                                                                                     i + 1, iterations_num,
                                                                                     ceil(i*100.0/iterations_num),
                                                                                     loss_sum / (i+1) \
                                                                                     ),
                                                                                     end='', flush=True)
        # extra diagnostics
        avg_cross_ent = cross_ent_sum / num_batches
        avg_l1_pen = l1_pen_sum / num_batches
        avg_sep_loss = sep_loss_sum / num_batches     
        if hasattr(tagger, 'prototypes'):            
            print('\n-- more epoch stats: cross_ent = %.2f, l1_pen: %.2f, sep_loss: %.2f' % (avg_cross_ent, avg_l1_pen, avg_sep_loss),
                end='', flush = True)            


        # Evaluate and/or push tagger
        # if epoch >= args.no_eval_til:
        train_score, dev_score, test_score, test_msg = evaluator.get_evaluation_score_train_dev_test(tagger,
                                                                                                     datasets_bank,
                                                                                                     batch_size=100)
        print('\n== eval epoch %d/%d "%s" train / dev / test | %1.2f / %1.2f / %1.2f.' % (epoch, args.epoch_num,
                                                                                        args.evaluator, train_score,
                                                                                        dev_score, test_score))

        # for prototype models, only push when dev_score > best_dev_score and epoch > args.push_after
        if dev_score > best_dev_score and hasattr(tagger, 'prototypes') and epoch > args.push_after:
                before_push_dev_score = dev_score
                tagger.eval()
                tagger.push(args, dataset, X_train, Y_train, save_prototype_files = False)
                tagger.pushed = True

                train_score, dev_score, test_score, test_msg = evaluator.get_evaluation_score_train_dev_test(tagger,
                                                                                                             datasets_bank,
                                                                                                             batch_size=100)
                print('\n== eval epoch %d/%d "%s" train / dev / test | %1.2f / %1.2f / %1.2f.' % (epoch, args.epoch_num,
                                                                                                args.evaluator, train_score,
                                                                                                dev_score, test_score))                            

                push_dev_loss = (before_push_dev_score - dev_score) # positive when score is lowered
                push_dev_losses.append(push_dev_loss)
                print('\n Loss of %.2f dev score points on push\n' % (before_push_dev_score - dev_score))

        report.write_epoch_scores(epoch, (loss_sum / iterations_num, train_score, dev_score, test_score,
                                                            avg_cross_ent, avg_l1_pen, avg_sep_loss))

        # Early stopping & model saving
        if dev_score > best_dev_score:
            if not hasattr(tagger, 'prototypes'):
                best_dev_score = dev_score
                best_test_score = test_score
                best_epoch = epoch
                best_test_msg = test_msg
                patience_counter = 0
                tagger.save_tagger(os.path.join(args.save_dir, '%s.hdf5' % args.save_name))
            # only save pushed models for prototype models    
            else: 
                if tagger.pushed: 
                    best_dev_score = dev_score
                    best_test_score = test_score
                    best_epoch = epoch
                    best_test_msg = test_msg
                    patience_counter = 0
                    tagger.save_tagger(os.path.join(args.save_dir, '%s.hdf5' % args.save_name))
            
            print('## [BEST epoch], %d seconds.\n' % (time.time() - time_start))
        else:
            patience_counter += 1
            print('## [no improvement eval metric on DEV during the last %d epochs (best_metric_dev=%1.2f), %d seconds].\n' %
                                                                                            (patience_counter,
                                                                                             best_dev_score,
                                                                                             (time.time()-time_start)))


        if patience_counter > args.patience and epoch > args.min_epoch_num:
            break
    

    # final load and push for prototype models
    if hasattr(tagger, 'prototypes'):
        load_path = os.path.join('saved_models','%s.hdf5' % args.save_name)
        print("Final load and push for model at %s" % load_path)
        tagger = TaggerFactory.load(load_path, args.gpu)
        tagger.push(args, dataset, X_train, Y_train, save_prototype_files = True)        
    
    # Show and save the final scores
    if args.save_best:
        report.write_final_score('Final eval on test, "save best", best epoch on dev %d, %s, dev = %1.2f | test = %1.2f)' %
                                 (best_epoch, args.evaluator, best_dev_score, best_test_score))
        report.write_msg(best_test_msg)
        report.write_input_arguments()
        report.write_final_line_score(best_test_score)
    else:
        report.write_final_score('Final eval on test, %s test = %1.2f)' % (args.evaluator, test_score))
        report.write_msg(test_msg)
        report.write_input_arguments()        
        report.write_final_line_score(test_score)
        if hasattr(tagger, 'prototypes'):
            push_dev_losses_str = ', '.join([str(score) for score in push_dev_losses])
            report.write_msg('Push dev losses: %s' % push_dev_losses_str)
    if args.verbose:
        report.make_print()
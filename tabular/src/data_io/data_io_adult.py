"""input/output data wrapper for adult UCI dataset"""
import codecs
from anchor.utils import load_dataset
import sklearn
import os
import pickle


class DataIOAdult():
    """DataIONerConnl2003 is an input/output data wrapper for UCI adult dataset"""
    def read_train_dev_test(self, args):
        # returns data as dense np matrices, along with the dataset object which has human-readable info
        
        dataset = load_dataset(args.data_name, balance=True)

        # manual override of class names. <$50k is now 'neg', >$50k is now 'pos'. not a value judgment, just saves time re-coding from sentiment
        # dataset.class_names = ['neg', 'pos']
        # dataset.class_names = ['<=50K','>50K']
        dataset.class_names = ['below $50K','above $50K']

        train_dir = os.path.join('data', args.data_name,'train')
        dev_dir = os.path.join('data', args.data_name,'dev')
        test_dir = os.path.join('data', args.data_name,'test')


        # make data dirs if they do not exist
        try:
            os.mkdir(train_dir)
            os.mkdir(dev_dir)
            os.mkdir(test_dir)
        except:
            pass

        # if saving new data
        if args.save_data:        
            pickle.dump(dataset.train, open(os.path.join(train_dir,'data') + '.pkl','wb'))
            pickle.dump(dataset.labels_train, open(os.path.join(train_dir,'labels') + '.pkl','wb'))
            pickle.dump(dataset.validation, open(os.path.join(dev_dir,'data') + '.pkl','wb'))
            pickle.dump(dataset.labels_validation, open(os.path.join(dev_dir,'labels') + '.pkl','wb'))
            pickle.dump(dataset.test, open(os.path.join(test_dir,'data') + '.pkl','wb'))
            pickle.dump(dataset.labels_test, open(os.path.join(test_dir,'labels') + '.pkl','wb'))

        # if loading already saved data
        print("Loading data from .pkl's in data/%s" % args.data_name)
        X_train = pickle.load( open(os.path.join(train_dir,'data') + '.pkl','rb'))
        Y_train = pickle.load( open(os.path.join(train_dir,'labels') + '.pkl','rb'))
        X_dev = pickle.load( open(os.path.join(dev_dir,'data') + '.pkl','rb'))
        Y_dev = pickle.load( open(os.path.join(dev_dir,'labels') + '.pkl','rb'))
        X_test = pickle.load( open(os.path.join(test_dir,'data') + '.pkl','rb'))
        Y_test = pickle.load( open(os.path.join(test_dir,'labels') + '.pkl','rb'))
        
        if args.verbose:
            for name, data in zip(['train','dev','test'], [dataset.train,dataset.validation,dataset.test]):
                print('Loading from %s: %d samples.' % (name, len(data)))

        # switch to human-readable
        Y_train = [dataset.class_names[y] for y in Y_train]
        Y_dev = [dataset.class_names[y] for y in Y_dev]
        Y_test = [dataset.class_names[y] for y in Y_test]

        return dataset, X_train, Y_train, X_dev, Y_dev, X_test, Y_test


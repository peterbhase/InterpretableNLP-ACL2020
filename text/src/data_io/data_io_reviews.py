"""input/output data wrapper for polarity reviews data at http://www.cs.cornell.edu/people/pabo/movie-review-data/"""
import codecs
from src.classes.utils import get_words_num
import sklearn
import os
import en_core_web_lg
import pickle


class DataIOReviews():
    """DataIONerConnl2003 is an input/output data wrapper for polarity reviews data at http://www.cs.cornell.edu/people/pabo/movie-review-data/"""
    def read_train_dev_test(self, args):

        def tokenize(tokenizer,text):
            sequence = [token.text for token in tokenizer(text.decode('ascii'))]
            return sequence

        path = args.data_dir
        data = []
        labels = []
        f_names = ['rt-polarity.neg', 'rt-polarity.pos']

        train_dir = os.path.join(args.data_dir,'train')
        dev_dir = os.path.join(args.data_dir,'dev')
        test_dir = os.path.join(args.data_dir,'test')
        try:
            os.mkdir(train_dir)
            os.mkdir(dev_dir)
            os.mkdir(test_dir)
        except:
            pass

        if args.save_data:
        
            print("Loading tokenizer...")
            tokenizer = en_core_web_lg.load()
            
            for (l, f) in enumerate(f_names):
                for line in open(os.path.join(path, f), 'rb'):
                    try:
                        line.decode('utf-8')
                    except:
                        continue
                    data.append(line.strip())
                    labels.append('pos' if l else 'neg')

            # seed set in main.py
            train, test, train_labels, test_labels = sklearn.model_selection.train_test_split(data, labels, test_size=.2)
            train, val, train_labels, val_labels = sklearn.model_selection.train_test_split(train, train_labels, test_size=.1)

            word_sequences_train = [tokenize(tokenizer, text) for text in train]
            word_sequences_dev = [tokenize(tokenizer, text) for text in val]
            word_sequences_test = [tokenize(tokenizer, text) for text in test]

            pickle.dump(word_sequences_train, open(os.path.join(train_dir,'word_sequences') + '.pkl','wb'))
            pickle.dump(train_labels, open(os.path.join(train_dir,'labels') + '.pkl','wb'))
            pickle.dump(word_sequences_dev, open(os.path.join(dev_dir,'word_sequences') + '.pkl','wb'))
            pickle.dump(val_labels, open(os.path.join(dev_dir,'labels') + '.pkl','wb'))
            pickle.dump(word_sequences_test, open(os.path.join(test_dir,'word_sequences') + '.pkl','wb'))
            pickle.dump(test_labels, open(os.path.join(test_dir,'labels') + '.pkl','wb'))

        else:
            print("Loading data from .pkl's in %s" % args.data_dir)
            word_sequences_train = pickle.load( open(os.path.join(train_dir,'word_sequences') + '.pkl','rb'))
            train_labels = pickle.load( open(os.path.join(train_dir,'labels') + '.pkl','rb'))
            word_sequences_dev = pickle.load( open(os.path.join(dev_dir,'word_sequences') + '.pkl','rb'))
            val_labels = pickle.load( open(os.path.join(dev_dir,'labels') + '.pkl','rb'))
            word_sequences_test = pickle.load( open(os.path.join(test_dir,'word_sequences') + '.pkl','rb'))
            test_labels = pickle.load( open(os.path.join(test_dir,'labels') + '.pkl','rb'))
        
        if args.verbose:
            for name, word_sequences in zip(['train','dev','test'], [word_sequences_train,word_sequences_dev,word_sequences_test]):
                print('Loading from %s: %d samples, %d words.' % (name, len(word_sequences), get_words_num(word_sequences)))

        return word_sequences_train, train_labels, word_sequences_dev, val_labels, word_sequences_test, \
               test_labels
"""bla"""
# from __future__ import print_function
import copy
import sklearn
import numpy as np
# import string
import os
import sys
import time

if (sys.version_info > (3, 0)):
    def unicode(s, errors=None):
        return s#str(s)

class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)


def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy()
    for src, target in value_map.items():
        ret[ret == src] = target
    return ret
def replace_binary_values(array, values):
    return map_array_values(array, {'0': values[0], '1': values[1]})



def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_features=None,
                     features_to_use=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_features, consider everything < 20 as categorical"""
    if feature_transformations is None:
        feature_transformations = {}
    try:
        data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
    except:
        import pandas
        data = pandas.read_csv(data,
                               header=None,
                               delimiter=delimiter,
                               na_filter=True,
                               dtype=str).fillna(fill_na).values
    if target_idx < 0:
        target_idx = data.shape[1] + target_idx
    ret = Bunch({})
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])
    labels = data[:, target_idx]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    labels = ret.labels
    ret.class_names = list(le.classes_)
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_features:
            categorical_features = ([x if x < target_idx else x - 1
                                     for x in categorical_features])

    if categorical_features is None:
        categorical_features = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_features.append(f)
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)
    ordinal_features = []
    if discretize:
        disc = lime.lime_tabular.QuartileDiscretizer(data,
                                                     categorical_features,
                                                     feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]
        categorical_features = range(data.shape[1])
        categorical_names.update(disc.names)
    for x in categorical_names:
        categorical_names[x] = [y.decode() if type(y) == np.bytes_ else y for y in categorical_names[x]]
    ret.ordinal_features = ordinal_features
    ret.categorical_features = categorical_features
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                  test_size=.2,
                                                  random_state=1)
    train_idx, test_idx = [x for x in splits.split(data)][0]
    ret.train = data[train_idx]
    ret.labels_train = ret.labels[train_idx]
    cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                     test_size=.5,
                                                     random_state=1)
    cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
    cv_idx = test_idx[cv_idx]
    test_idx = test_idx[ntest_idx]

    ret.validation = data[cv_idx]
    ret.labels_validation = ret.labels[cv_idx]
    ret.test = data[test_idx]
    ret.labels_test = ret.labels[test_idx]
    ret.test_idx = test_idx
    ret.validation_idx = cv_idx
    ret.train_idx = train_idx

    # ret.train, ret.test, ret.labels_train, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(data, ret.labels,
    #                                               train_size=0.80))
    # ret.validation, ret.test, ret.labels_validation, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(ret.test, ret.labels_test,
    #                                               train_size=.5))
    ret.data = data
    return ret

class Neighbors:
    def __init__(self, nlp_obj):
        self.nlp = nlp_obj
        self.to_check = [w for w in self.nlp.vocab if w.prob >= -15]
        self.n = {}

    def neighbors(self, word):
        word = unicode(word)
        orig_word = word
        if word not in self.n:
            if word not in self.nlp.vocab:
                self.n[word] = []
            else:
                word = self.nlp.vocab[unicode(word)]
                queries = [w for w in self.to_check
                            if w.is_lower == word.is_lower]
                if word.prob < -15:
                    queries += [word]
                try:
                    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
                # this try/except was necessary but no idea why
                except:
                    # import ipdb; ipdb.set_trace()
                    sims = []
                    for i in range(len(queries)):
                        sims.append(word.similarity(queries[i]))
                    sort_idx = np.argsort([sim for sim in reversed(sims)]) # idx in descending sort order
                    by_similarity = queries[sort_idx]

                    print("what")

                self.n[orig_word] = [(self.nlp(w.orth_)[0], word.similarity(w))
                                     for w in by_similarity[:500]]
                                    #  if w.lower_ != word.lower_]
        return self.n[orig_word]

def perturb_sentence(text, present, n, neighbors, proba_change=0.5,
                     top_n=50, forbidden=[], forbidden_tags=['PRP$'],
                     forbidden_words=['be'],
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP'], use_proba=True, # try not changing determiners
                     #pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], use_proba=True, # I think the proba flag is backwards?
                     temperature=.4): 
    # words is a list of words (must be unicode)
    # present is which ones must be present, also a list
    # n = how many to sample
    # neighbors must be of utils.Neighbors
    # nlp must be spacy
    # proba_change is the probability of each word being different than before
    # forbidden: forbidden lemmas
    # forbidden_tags, words: self explanatory
    # pos: which POS to change

    '''
    it's a long story about why orig_sequence has to be returned. (find the ending in src.classes.utils.get_path_to_decision_boundary)
    for some reason, the tokenizing occuring here is different from the tokenizing done in our src.data_io.data_io_reviews
    '''
    start_time = time.time()
    tokens = neighbors.nlp(unicode(text))    
    orig_sequence = [token.text for token in tokens] 
    # print [x.pos_ for x in tokens]
    eligible = []
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = np.zeros((n, len(tokens)), '|S80')
    data = np.ones((n, len(tokens)))
    raw[:] = [x.text for x in tokens]
    for i, t in enumerate(tokens):
        if i in present:
            continue
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            r_neighbors = [
                (unicode(x[0].text.encode('utf-8'), errors='ignore'), x[1])
                for x in neighbors.neighbors(t.text)
                if x[0].tag_ == t.tag_][:top_n]
            # for neighbor in r_neighbors:
            #     print(neighbor)
            if not r_neighbors:
                continue
            t_neighbors = [x[0] for x in r_neighbors]
            weights = np.array([x[1] for x in r_neighbors])
            if use_proba:
                weights = weights ** (1. / temperature)
                weights = weights / sum(weights)
                # print t.text
                # print sorted(zip(t_neighbors, weights), key=lambda x:x[1], reverse=True)[:10]
                raw[:, i] = np.random.choice(t_neighbors, n,  p=weights,
                                             replace=True)
                # The type of data in raw is byte.
                data[:, i] = raw[:, i] == t.text.encode()
            else: 
                n_changed = np.random.binomial(n, proba_change)
                changed = np.random.choice(n, n_changed, replace=False)
                if t.text in t_neighbors:
                    idx = t_neighbors.index(t.text)
                    weights[idx] = 0
                weights = weights ** (1. / temperature)
                weights = weights / sum(weights)
                try:
                    raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights)
                except:
                    print("random choice when perturbing because weights contained NaNs")
                    raw[changed, i] = np.random.choice(t_neighbors, n_changed)
                data[changed, i] = 0
#         else:
#             print t.text, t.pos_ in pos, t.lemma_ in forbidden, t.tag_ in forbidden_tags, t.text in neighbors
    # print raw
    if (sys.version_info > (3, 0)):
        raw = [' '.join([y.decode() for y in x]) for x in raw]
    else:
        raw = [' '.join(x) for x in raw]
    return raw, data, orig_sequence





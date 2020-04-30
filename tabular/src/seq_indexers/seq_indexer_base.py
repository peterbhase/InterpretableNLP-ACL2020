"""base abstract class for sequence indexers"""
import numpy as np
import torch


class SeqIndexerBase():
    """
    SeqIndexerBase is a base abstract class for sequence indexers. It converts list of lists of string items
    to the list of lists of integer indices and back. Items could be either words, tags or characters.
    """
    def __init__(self, gpu=-1, check_for_lowercase=True, zero_digits=False, pad='<pad>', unk='<unk>',
                 load_embeddings=False, embeddings_dim=0, verbose=False):
        self.gpu = gpu
        self.check_for_lowercase = check_for_lowercase
        self.zero_digits = zero_digits
        self.pad = pad
        self.unk = unk
        self.load_embeddings = load_embeddings
        self.embeddings_dim = embeddings_dim
        self.verbose = verbose
        self.out_of_vocabulary_list = list()
        self.item2idx_dict = dict()
        self.idx2item_dict = dict()
        self.item2counts_dict = dict()
        if load_embeddings:
            self.embeddings_loaded = False
            self.embedding_vectors_list = list()
        if pad is not None:
            self.pad_idx = self.add_item(pad)
            if load_embeddings:
                self.add_emb_vector(self.generate_zero_emb_vector())
        if unk is not None:
            self.unk_idx = self.add_item(unk)
            if load_embeddings:
                self.add_emb_vector(self.generate_random_emb_vector())

    def get_items_list(self):
        return list(self.item2idx_dict.keys())

    def get_items_count(self):
        return len(self.get_items_list())

    def item_exists(self, item):
        return item in self.item2idx_dict.keys()

    def add_item(self, item):
        idx = len(self.get_items_list())
        self.item2idx_dict[item] = idx
        self.idx2item_dict[idx] = item
        return idx

    def count(self, item):
        if item not in self.item2counts_dict:
            self.item2counts_dict[item] = 1
        else:
            self.item2counts_dict[item] += 1

    def get_class_num(self):
        if self.pad is not None and self.unk is not None:
            return self.get_items_count() - 2
        if self.pad is not None or self.unk is not None:
            return self.get_items_count() - 1
        return self.get_items_count()


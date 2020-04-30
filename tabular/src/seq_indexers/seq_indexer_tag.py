"""converts list of lists of string tags to list of lists of integer indices and back"""
from src.seq_indexers.seq_indexer_base import SeqIndexerBase
import numpy as np
import time
import torch


class SeqIndexerTag(SeqIndexerBase):
    """SeqIndexerTag converts list of lists of string tags to list of lists of integer indices and back."""
    def __init__(self, gpu):
        SeqIndexerBase.__init__(self, gpu=gpu, check_for_lowercase=False, zero_digits=False,
                                      pad=None, unk=None, load_embeddings=False, verbose=True)

    def items2idx(self, item_sequence):
        idx_seq = []
        for item in item_sequence:
            if item in self.item2idx_dict:
                idx_seq.append(self.item2idx_dict[item])
            else:
                if self.unk is not None:
                    idx_seq.append(self.item2idx_dict[self.unk])
                else:
                    idx_seq.append(self.item2idx_dict[self.pad])
        return idx_seq

    def idx2items(self, idx_seq):
        item_seq = [self.idx2item_dict[idx] for idx in idx_seq]
        return item_seq

    def items2tensor(self, item_sequences):
        idx = self.items2idx(item_sequences)
        return self.idx2tensor(idx)

    def idx2tensor(self, idx_sequence):
        tensor = torch.LongTensor(idx_sequence)
        if self.gpu >= 0:
            tensor = tensor.cuda(device=self.gpu)
        return tensor


    def add_tag(self, tag):
        if not self.item_exists(tag):
            self.add_item(tag)

    def load_items_from_tag_sequence(self, tag_sequence):
        print("Loading items from tag sequence...")
        assert self.load_embeddings == False
        for tag in tag_sequence:
            self.add_tag(tag)
            self.count(tag)
        
    def print_sorted_tags(self):
        tags_ids = [(tag, idx) for tag,idx in self.item2idx_dict.items()]
        tags_ids.sort(key = lambda tup: tup[1])
        print(tags_ids)

    def print_sorted_counts(self):
        tags_counts = [(tag, count) for tag,count in self.item2counts_dict.items()]
        tags_counts.sort(key = lambda tup: tup[1], reverse = True)
        print(tags_counts)

    def print_class_info(self):
        print(' -- class_num = %d' % self.get_class_num())
        print(' -- ')
        self.print_sorted_tags()
        print('\n -- ')
        self.print_sorted_counts()

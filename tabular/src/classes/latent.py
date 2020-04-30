'''class for saving the human readable information associated with latent vectors'''
import numpy as np

class Latent():
    '''class for saving the human readable information associated with latent vectors'''

    def __init__(self, context, token_id = None, class_id = None, tag = None, global_id = None, 
                    where_different = None, distance = 0, pick_type = None, # used for perturbations
                    blackbox_tag = None, old_proto_tag = None, our_proto_tag = None):
        self.context = context
        self.token_id = token_id
        self.class_id = class_id
        self.tag = tag
        self.global_id = global_id
        self.negative_distance = -distance
        self.blackbox_tag = blackbox_tag
        self.old_proto_tag = old_proto_tag
        self.our_proto_tag = our_proto_tag
        self.blackbox_correct = 1*(tag == blackbox_tag) if blackbox_tag is not None and tag is not None else None
        self.old_proto_correct = 1*(tag == old_proto_tag) if old_proto_tag is not None and tag is not None else None
        self.our_proto_correct = 1*(tag == our_proto_tag) if our_proto_tag is not None and tag is not None else None

        # used when Latent is a perturbation
        self.similarity_score_ = self.similarity_score(distance)
        self.similarity_score_percent = self.similarity_score_ / self.similarity_score(distances=0)
        self.similarity_score_change = self.similarity_score_ - self.similarity_score(distances=0)
        self.where_different = where_different
        self.distance = distance
        self.pick_type = pick_type

    def __str__(self):
        '''print with tokens separated by spaces and prototype word singled out with asterisks: *word*'''
        text = self.to_str()
        return text

    def to_str(self):
        text = [word for word in self.context]
        if self.where_different is not None:
            for i in self.where_different:
                text[i] = '**' + text[i] + '**'
        return ' '.join(text)  

    def __lt__(self, other):
        # since popping from a heap removes the smallest element
        return self.negative_distance < other.negative_distance

    def similarity_score(self, distances, epsilon):
        # similarity score function used by prototype model, here with np rather than torch functions
        return np.log(1 + (1 / (distances + epsilon)))

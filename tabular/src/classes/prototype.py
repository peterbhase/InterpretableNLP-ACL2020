'''prototype class for saving the human readable information associated with prototype vectors. used in tagger.push()'''

class Prototype():
    '''prototype class for saving the human readable information associated with prototype vectors. used in tagger.push()'''

    def __init__(self, prototype_id, context, class_id, tag, global_idx, batch_size = 10,
                    importance_score = None, distance = 0):
        self.prototype_id = prototype_id
        self.context = context
        self.class_id = class_id
        self.tag = tag
        self.global_idx = global_idx
        self.batch_size = batch_size
        self.importance_score = importance_score
        self.negative_distance = -distance
        self.distance = distance

    def __str__(self):
        '''print with tokens separated by spaces and prototype word singled out with asterisks: *word*'''
        return ' '.join(self.context)

    def to_str(self):
        return ' '.join(self.context)  

    def __lt__(self, other):
        # defined this way since popping from a heap removes the smallest element
        # smaller = further away
        return self.negative_distance < other.negative_distance        




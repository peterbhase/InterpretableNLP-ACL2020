'''prototype class for saving the human readable information associated with prototype vectors. used in tagger.push()'''
from src.classes.utils import force_str

class Prototype():
    '''prototype class for saving the human readable information associated with prototype vectors. used in tagger.push()'''

    def __init__(self, dataset_obj, prototype_id, context, class_id, tag, global_idx, batch_size = 10,
                    importance_score = None, distance = 0):
        self.prototype_id = prototype_id
        self.context = context
        self.class_id = class_id
        self.tag = tag
        self.global_idx = global_idx
        self.batch_size = batch_size
        self.importance_score = importance_score
        self.negative_distance = -distance # used with heaps. see self.__lt__
        self.distance = distance
        self.feature_names = dataset_obj.feature_names
        self.categorical_names_dict = dataset_obj.categorical_names # maps 

    def __str__(self):
        '''print with tokens separated by spaces and prototype word singled out with asterisks: *word*'''
        return self.to_str()

    def _get_variable_value_human_readable(self, col_id):
        variable_value = int(self.context[col_id])
        variable_str = self.categorical_names_dict[col_id][variable_value]
        return force_str(variable_str)

    def to_str(self):
        data_point_list_of_str = [
            '%s : %s' % (self.feature_names[col_id], self._get_variable_value_human_readable(col_id))
            for col_id in range(self.context.shape[-1])
        ]
        return '\n'.join(data_point_list_of_str) 

    def __lt__(self, other):
        # defined this way since popping from a heap removes the smallest element
        # smaller = further away
        return self.negative_distance < other.negative_distance        




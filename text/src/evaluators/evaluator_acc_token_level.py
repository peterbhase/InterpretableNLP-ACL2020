"""token-level accuracy evaluator for each class of BOI-like tags"""
from src.evaluators.evaluator_base import EvaluatorBase


class EvaluatorAccuracyTokenLevel(EvaluatorBase):
    """EvaluatorAccuracyTokenLevel is token-level accuracy evaluator for each class"""
    def get_evaluation_score(self, targets_seq, outputs_seq, class2id_dict = None):
        # class2id_dict should map the 'positive' class to 1, for confusion stats
        # if none provided, default to hard-coded maps
        if class2id_dict is None:
            classes = sorted(list(set(targets_seq)))
            class2id_dict = {label:idx for idx,label in enumerate(classes)}
        cnt = 0
        match = 0
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        id2class_dict = {v: k for k,v in class2id_dict.items()}
        for t, o in zip(targets_seq, outputs_seq):
            cnt += 1
            if t == o:
                match += 1
            # positive prediction
            if class2id_dict[o]:
                if t == o:
                    TP += 1
                else:
                    FP += 1
            # negative prediction
            elif not class2id_dict[o]:
                if t == o:
                    TN += 1
                else:
                    FN += 1

        acc = match*100.0/cnt
        msg = '\n*** Token-level accuracy: %1.2f%% ***' % acc
        msg += "\n \t TP: %2.2f | abs: %d" % (TP*100/(TP+FP) if (TP+FP) > 0 else -99.99, TP)
        msg += "\n \t FP: %2.2f | abs: %d" % (FP*100/(TP+FP) if (TP+FP) > 0 else -99.99, FP)
        msg += "\n \t TN: %2.2f | abs: %d" % (TN*100/(TN+FN) if (TN+FN) > 0 else -99.99, TN)
        msg += "\n \t FN: %2.2f | abs: %d" % (FN*100/(TN+FN) if (TN+FN) > 0 else -99.99, FN)
        msg += "\nNeg class: %s, Pos class: %s" % (id2class_dict[0], id2class_dict[1])
        return acc, msg

"""creates various evaluators"""
from src.evaluators.evaluator_acc_token_level import EvaluatorAccuracyTokenLevel


class EvaluatorFactory():
    """EvaluatorFactory contains wrappers to create various evaluators."""
    @staticmethod
    def create(args):
        return EvaluatorAccuracyTokenLevel()
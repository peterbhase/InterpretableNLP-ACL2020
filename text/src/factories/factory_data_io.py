"""creates various data readers/writers"""
from src.data_io.data_io_reviews import DataIOReviews

class DataIOFactory():
    """DataIOFactory contains wrappers to create various data readers/writers."""
    @staticmethod
    def create(args):
        return DataIOReviews()

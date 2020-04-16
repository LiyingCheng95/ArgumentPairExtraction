# 
# @author: Allan
#
from common.sentence import  Sentence
from typing import List

class Instance:
    """
    This class is the basic Instance for a datasample
    """

    def __init__(self, input: Sentence, output: List[str] = None, vec: List[List] = None, type: List[List] = None, review_idx: List[List] = None, reply_idx: List[List] = None, labels_pair: List[List] = None, max_review_id: List[List] = None, max_num_tokens: List[List] = None) -> None:
        """
        Constructor for the instance.
        :param input: sentence containing the words
        :param output: a list of labels
        """
        self.input = input
        self.output = output
        self.elmo_vec = None #used for loading the ELMo vector.
        self.sent_ids = None
        self.char_ids = None
        self.output_ids = None
        self.vec = vec
        self.type = type
        self.review_idx = review_idx
        self.reply_idx = reply_idx
        self.labels_pair = labels_pair
        self.max_review_id = max_review_id
        self.max_num_tokens = max_num_tokens

    def __len__(self):
        return len(self.input)

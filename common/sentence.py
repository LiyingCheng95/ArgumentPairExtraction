# 
# @author: Allan
#

from typing import List

class Sentence:
    """
    The class for the input sentence
    """

    def __init__(self, sents: List[str], ori_sents: List[str] = None, pos_tags:List[str] = None):
        """

        :param words:
        :param pos_tags: By default, it is not required to have the pos tags, in case you need it/
        """
        self.sents = sents
        self.ori_sents = ori_sents
        self.pos_tags = pos_tags

    def __len__(self):
        return len(self.sents)

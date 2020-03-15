# 
# @author: Allan
#

from tqdm import tqdm
from common import Sentence, Instance
from typing import List
from bert_serving.client import BertClient
# import sister
import re


class Reader:

    def __init__(self, digit2zero:bool=True):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        self.digit2zero = digit2zero
        self.vocab = set()

    def read_txt(self, file: str, number: int = -1) -> List[Instance]:
        print("Reading file: " + file)
        insts = []

        # embedder = sister.MeanEmbedding(lang='en')
        with open(file, 'r', encoding='utf-8') as f:
            words = []
            ori_words = []
            labels = []
            vecs = []
            for line in tqdm(f.readlines()):
                line = line.rstrip()
                if line == "":
                    inst = Instance(Sentence(words, ori_words), labels, vecs)
                    ##read vector

                    insts.append(inst)
                    words = []
                    ori_words = []
                    labels = []
                    if len(insts) == number:
                        break
                    continue
                ls = line.split('\t')
                word, label = ls[0],ls[2]
                ori_words.append(word)
                # if self.digit2zero:
                #     word = re.sub('\d', '0', word) # replace digit with 0.
                words.append(word)
                self.vocab.add(word)

                bc = BertClient()
                tmp = []
                tmp.append(word)
                vec = bc.encode(list(tmp))
                vecs.append(vec[0][0])

                # vec=embedder(word)
                # vecs.append(vec)
                # print(vec)

                labels.append(label)
        print("number of sentences: {}".format(len(insts)))
        return insts




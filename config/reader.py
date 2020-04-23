# 
# @author: Allan
#

from tqdm import tqdm
from common import Sentence, Instance
from typing import List
from bert_serving.client import BertClient
import re
import pickle


class Reader:

    def __init__(self, digit2zero:bool=True):
        """
        Read the dataset into Instance
        :param digit2zero: convert the digits into 0, which is a common practice for LSTM-CRF.
        """
        self.digit2zero = digit2zero
        self.vocab = set()
        self.type_vocab = {'Review', 'Reply'}

    def read_txt(self, file: str, number: int = 5) -> List[Instance]:
        print("Reading file: " + file)
        insts = []

        # f_vec = open(file[:8]+'vec_test.pkl', 'rb')
        f_vec = open(file[:8] + 'vec_' + file[8:-3] + 'pkl', 'rb')
        all_vecs = pickle.load(f_vec)
        f_vec.close

        with open(file, 'r', encoding='utf-8') as f:

            sents = []
            ori_sents = []
            labels = []
            types = []
            sent_idx = 0
            review_idx = []
            reply_idx = []
            labels_pair = []
            max_review_id=0
            new_index = 0

            f= f.readlines()

            for line_idx, line in enumerate(tqdm(f)):
                line = line.rstrip()
                if line == "":
                    new_index =0
                    vecs=all_vecs[len(insts)]
                    # max_num_tokens = len(vecs[0])
                    num_tokens = [len(vecs[i]) for i in range(len(vecs))]
                    inst = Instance(Sentence(sents, ori_sents), labels, vecs, types, review_idx, reply_idx, labels_pair, max_review_id,num_tokens)
                    ##read vector
                    print(review_idx,reply_idx,max_review_id,labels_pair)
                    insts.append(inst)
                    sents = []
                    ori_sents = []
                    labels = []
                    types = []
                    sent_idx = 0
                    review_idx = []
                    reply_idx = []
                    labels_pair = []
                    max_review_id=0
                    if len(insts) == number:
                        break
                    continue
                ls = line.split('\t')
                if ls[1]=='O':
                    sent, label, label_pair, type = ls[0], ls[1], 0, ls[-1]
                else:
                    sent, label, label_pair, type = ls[0], ls[1][:2] + '0', int(ls[2][2:]), ls[-1]

                ori_sents.append(sent)
                if type == 'Review':
                    type_id = 0
                    if label[0] != 'O':
                        review_idx.append(sent_idx)
                    # else:
                    #     review_idx.append(0)
                    max_review_id += 1
                else:
                    type_id = 1
                    # print(line_idx,len(f),f[line_idx+1])
                    if line_idx<len(f)-2:
                        if label[0] != 'O' or (label[0]=='O' and ( (new_index>0 and f[line_idx-1].rstrip().split('\t')[1][0] != 'O') or
                                                               (new_index>1 and f[line_idx-2].rstrip().split('\t')[1][0] != 'O') or
                                                               (f[line_idx+1]!='' and f[line_idx+1].rstrip().split('\t')[1][0] != 'O') or
                                                               (f[line_idx+1]!='' and f[line_idx+2]!='' and f[line_idx+2].rstrip().split('\t')[1][0] != 'O' ))):
                            reply_idx.append(sent_idx)
                    elif line_idx==len(f)-2:
                        if f[line_idx + 1].rstrip() != '':
                            if label[0] != 'O' or (label[0] == 'O' and (
                                    (new_index > 0 and f[line_idx - 1].rstrip().split('\t')[1][0] != 'O') or
                                    (new_index > 1 and f[line_idx - 2].rstrip().split('\t')[1][0] != 'O') or
                                    (f[line_idx + 1] != '' and f[line_idx + 1].rstrip().split('\t')[1][0] != 'O'))):
                                reply_idx.append(sent_idx)
                        else:
                            if label[0] != 'O' or (label[0] == 'O' and (
                                    (new_index > 0 and f[line_idx - 1].rstrip().split('\t')[1][0] != 'O') or
                                    (new_index > 1 and f[line_idx - 2].rstrip().split('\t')[1][0] != 'O'))):
                                reply_idx.append(sent_idx)


                    elif line_idx==len(f)-1:
                        if label[0] != 'O' or (label[0]=='O' and ( (new_index>0 and f[line_idx-1].rstrip().split('\t')[1][0] != 'O') or
                                                               (new_index>1 and f[line_idx-2].rstrip().split('\t')[1][0] != 'O'))):
                            reply_idx.append(sent_idx)


                types.append(type_id)

                sent_idx+=1
                new_index+=1


                # if self.digit2zero:
                #     sent = re.sub('\d', '0', sent) # replace digit with 0.
                sents.append(sent)
                self.vocab.add(sent)

                # bc = BertClient()
                # vec = bc.encode([sent])
                # vecs.append(vec[0][0])

                labels.append(label)
                labels_pair.append(label_pair)
        print("number of sentences: {}".format(len(insts)))
        all_vecs = 0
        vecs = 0
        return insts




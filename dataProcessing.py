import csv
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from bert_serving.client import BertClient
import random
from transformers import *
import re
import torch
import numpy as np
from collections import defaultdict



def calc_vec(sents):
	sents=sents.split('\n')
	bc = BertClient()
	tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
	all_vecs = []
	for inst in [sents[i].split('\t')[0] for i in range(len(sents))]:
		tokens = []
		orig_to_tok_index = []  # 0 - >0, 1-> len(all word_piece)
		for j, word in enumerate(inst.split(' ')):
			orig_to_tok_index.append(len(tokens))
			word_tokens = tokenizer.tokenize(word.lower())
			for sub_token in word_tokens:
				# orig_to_tok_index.append(i)
				tokens.append(sub_token)
		vec = bc.encode([tokens], is_tokenized=True)
		all_vecs.append(vec)

	return all_vecs

def sep_data():
	with open("all_data_v2.txt", "r") as full_data:
		full_data = full_data.read().split('\n\n')
		print(len(full_data))
		
		paperid2pair = defaultdict(list)
		for passage_pair in full_data:
			id = passage_pair.split('\t')[-1]
			paperid2pair[id].append(passage_pair)

		# # paperid2pair = {passage_pair.split('\t')[-1]: passage_pair for passage_pair in full_data}
		paperids = list(paperid2pair.keys())

		random.seed(230)
		random.shuffle(paperids)

		split_1 = int(0.814*len(paperids))
		split_2 = int(0.9*len(paperids))
		train_ids = paperids[:split_1]
		dev_ids = paperids[split_1:split_2]
		test_ids = paperids[split_2:]
		
		train_data = [passage for id in train_ids for passage in paperid2pair[id]]
		dev_data = [passage for id in dev_ids for passage in paperid2pair[id]]
		test_data = [passage for id in test_ids for passage in paperid2pair[id]]

		print('ratio for train dev test: {}: {}: {}'.format(len(train_data), len(dev_data), len(test_data)))
		

		# full_data.sort()  # make sure that the filenames have a fixed order before shuffling
		# random.seed(230)
		# random.shuffle(full_data)  # shuffles the ordering of filenames (deterministic given the chosen seed)

		# split_1 = int(0.8 * len(full_data))
		# split_2 = int(0.9 * len(full_data))
		# print(split_1, split_2)
		# train_data = full_data[:split_1]
		# dev_data = full_data[split_1:split_2]
		# test_data = full_data[split_2:]

		print('processing dev vec file ...')
		dev = open('dev.txt', 'w')
		vecs = []
		for i in dev_data:
			dev.write(i + '\n\n')
			vec = calc_vec(i)
			vecs.append([vec[j][0][1:-1] for j in range(len(vec))])
			# vecs.append([vec[i][0] for i in range(len(vec))])
			# print('len of vec: ', len(vecs[-1]))
		dev_vecs = open('vec_dev.pkl', 'wb')
		pickle.dump(vecs, dev_vecs)
		dev_vecs.close()
		dev.close()
		print('dev done')

		print('processing test vec file ...')
		test = open('test.txt', 'w')
		vecs = []
		for i in test_data:
			test.write(i + '\n\n')
			vec = calc_vec(i)
			vecs.append([vec[j][0][1:-1] for j in range(len(vec))])
			# vecs.append([vec[i][0] for i in range(len(vec))])
			# print('len of vec: ', len(vecs[-1]))
		test_vecs = open('vec_test.pkl', 'wb')
		pickle.dump(vecs, test_vecs)
		test_vecs.close()
		test.close()
		print('test done')

		print('processing train vec file ...')
		train = open('train.txt', 'w')
		vecs = []
		for i in train_data:
			try:
				vec = calc_vec(i)
			except:
				continue
			train.write(i + '\n\n')
			vecs.append([vec[j][0][1:-1] for j in range(len(vec))])
			# vecs.append([vec[i][0] for i in range(len(vec))])
			# print('len of vec: ', len(vecs[-1]))
		train_vecs = open('vec_train.pkl', 'wb')
		pickle.dump(vecs, train_vecs)
		train_vecs.close()
		train.close()
		print('train done')




if __name__=='__main__':
	sep_data()



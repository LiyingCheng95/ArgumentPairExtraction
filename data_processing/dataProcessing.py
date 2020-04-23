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
			
def sep_sent(s):
	nlp = English()

	sentencizer = nlp.create_pipe("sentencizer")
	nlp.add_pipe(sentencizer)
	s = s.replace('by Yu et al.', 'by Yu et al..').replace('by Wen et al.', 'by Wen et al..').replace('et. al.','et al').replace('et al.,','et al').replace('et al.','et al')
	# s = re.sub('et al. [^A-Z]]','et al [A-Z]',s)
	doc = nlp(s)
	doc = list(doc.sents)
	return doc

def tagging_sequence(outputfile,corpus,arguments_corpus,category):
	bio=0
	vecs=[]
	sents=[]
	# add_to_next_i = ''
	j2tag = 0
	for index,i in enumerate(corpus):
		flag=0
		temp=''
		# temp.append(index)
		processed_i = str(i).replace('\n','[line_break_token]').replace('\t','[tab_token]')
		if processed_i.strip() == '':
			print(processed_i)
			# add_to_next_i = processed_i
			continue
		
		# try:
		# 	bc = BertClient()
		# 	vec = bc.encode([processed_i])
		# 	vecs.append(vec[0][0])
		# 	sents.append(processed_i)
		# except:
		# 	continue

		sents.append(processed_i)

		# temp+=vec
		# temp+='\t'
		# temp+=category
		# temp+=' '
		temp+=processed_i
		# add_to_next_i=''

		temp+='\t'
		i=str(i).replace('\n','')


		for j in arguments_corpus:
			# print("j: ",j[1])
			# print("i: ",i)
			
			if i in j[1].replace('et. al.','et al ') and i!='':
				# print(j2tag,j[2])
				if j2tag!=j[2] or bio==0:

					temp+='B-'
					temp+=category
					temp+='\t'
					temp+='B-'
					bio=1
				else:
					temp+='I-'
					temp+=category
					temp+='\t'
					temp+='I-'


				temp+=j[2]
				j2tag=j[2]
				temp += '\t'
				temp += category
				temp+='\n'
				outputfile.write(temp)
				flag=1
				break
		if flag==0:
			temp+='O'
			temp+='\t'
			bio=0

			temp+='O'
			temp += '\t'
			temp += category
			temp+='\n'
			outputfile.write(temp)


def main():
	with open('ReviewRebuttalnew.txt', 'w') as outputfile:

		with open('file2.csv') as csvfile2:
			file2 = csv.reader(csvfile2, delimiter=',')
			file2=list(file2)
		
			with open('file1.csv') as csvfile1:
				file1 = csv.reader(csvfile1, delimiter=',') 

				# vecs=[]
				# # sents=[]
			
				for line in file1:

					arguments_review=[ file2[index] for index in [i for i, e in enumerate(file2) if e[0] == line[0] and e[-1]=="Review"] ]
					arguments_reply=[ file2[index] for index in [i for i, e in enumerate(file2) if e[0] == line[0] and e[-1]=="Reply"] ]
					# print(line)
					sep=line[1].find("——————【reply】——————")
					review=line[1][22:sep-2]
					reply=line[1][sep+21:-1]
					review=sep_sent(review)
					tagging_sequence(outputfile,review,arguments_review,'Review')
					reply=sep_sent(reply)
					tagging_sequence(outputfile,reply,arguments_reply,'Reply')
					outputfile.write('\n')

					# sents=rvw_sents+rpl_sents
					# print('len of sents: ', len(sents))
					# bc = BertClient()
					# vec = bc.encode(sents)
					# vecs.append([vec[i][0] for i in range(len(vec))])
					# print('len of vec: ',len(vecs[-1]))


				# f_vecs=open('vecs','wb')
				# pickle.dump(vecs,f_vecs)
				# f_vecs.close()


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
	# print('all_tokens',len(all_tokens))
	# print(len(all_tokens[0]))

	# vec = bc.encode([sents[i].split('\t')[0] for i in range(len(sents))])
	# print(len(vec))
	# print(len(vec[0]))
	# print(len(vec[1]))
	# print(len(vec[0][0]))
	return all_vecs

def sep_data():
	with open("ReviewRebuttalnew.txt", "r") as full_data:
		full_data = full_data.read().split('\n\n')
		print(len(full_data))
		full_data.sort()  # make sure that the filenames have a fixed order before shuffling
		random.seed(230)
		random.shuffle(full_data)  # shuffles the ordering of filenames (deterministic given the chosen seed)

		split_1 = int(0.8 * len(full_data))
		split_2 = int(0.9 * len(full_data))
		print(split_1, split_2)
		train_data = full_data[:split_1]
		dev_data = full_data[split_1:split_2]
		test_data = full_data[split_2:]

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



def generate_testnew():
	with open("test.txt", "r") as f:
		f = f.read().split('\n\n')
		fnew = open('testnew.txt','w')

		for i in f[:-1]:
			reply = []
			inst = i.split('\n')
			for line in inst:
				if line.split('\t')[-1]=='Reply':
					reply.append(line.split('\t')[-2][-1])
			for line in inst:
				print(line)
				line_split = line.split('\t')
				if line_split[-2]=='O':
					fnew.write(line_split[0] + '\t' + line_split[1] + '\t' + line_split[2] + '\t' + line_split[2] + '\t' +
							   line_split[-1] + '\n')
				elif line_split[-1]=='Reply':
					fnew.write(
						line_split[0] + '\t' + line_split[1] + '\t' + line_split[2] + '\t' + line_split[2][0] + '-0\t' +
						line_split[-1] + '\n')
				else:
					review_index = line_split[-2][-1]
					match_index=''
					for reply_index in reply:
						if review_index==reply_index:
							match_index+='1'
						else:
							match_index+='0'
					fnew.write(
						line_split[0] + '\t' + line_split[1] + '\t' + line_split[2] + '\t' + line_split[2][
							0] + '-' + match_index + '\t' +
						line_split[-1] + '\n')

			fnew.write('\n')

	f.close()
	fnew.close()


# main()
sep_data()


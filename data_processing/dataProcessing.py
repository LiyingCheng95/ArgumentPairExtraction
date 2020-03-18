import csv
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from bert_serving.client import BertClient
import random
			
def sep_sent(s):
	nlp = English()
	sentencizer = nlp.create_pipe("sentencizer")
	nlp.add_pipe(sentencizer)
	doc = nlp(s)
	return list(doc.sents)

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
			
			if i in j[1] and i!='':
				if j2tag!=j[2]:
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
	with open('ReviewRebuttal.txt', 'w') as outputfile:

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
	vec = bc.encode([sents[i].split('\t')[0] for i in range(len(sents))])
	return vec




main()
with open("ReviewRebuttal.txt","r") as full_data:
	full_data=full_data.read().split('\n\n')
	print(len(full_data))
	full_data.sort()  # make sure that the filenames have a fixed order before shuffling
	random.seed(230)
	random.shuffle(full_data) # shuffles the ordering of filenames (deterministic given the chosen seed)

	split_1 = int(0.8 * len(full_data))
	split_2 = int(0.9 * len(full_data))
	print(split_1,split_2)
	train_data = full_data[:split_1]
	dev_data = full_data[split_1:split_2]
	test_data = full_data[split_2:]

	dev=open('dev.txt','w')
	vecs = []
	for i in dev_data:
		dev.write(i+'\n\n')
		vec = calc_vec(i)
		vecs.append([vec[i][0] for i in range(len(vec))])
		print('len of vec: ', len(vecs[-1]))
	dev_vecs = open('dev_vecs', 'wb')
	pickle.dump(vecs, dev_vecs)
	dev_vecs.close()
	dev.close()

	test=open('test.txt','w')
	vecs = []
	for i in test_data:
		test.write(i+'\n\n')
		vec = calc_vec(i)
		vecs.append([vec[i][0] for i in range(len(vec))])
		print('len of vec: ', len(vecs[-1]))
	test_vecs = open('test_vecs', 'wb')
	pickle.dump(vecs, test_vecs)
	test_vecs.close()
	test.close()


	train=open('train.txt','w')
	vecs=[]
	for i in train_data:
		try:
			vec=calc_vec(i)
		except:
			continue
		train.write(i + '\n\n')
		vecs.append([vec[i][0] for i in range(len(vec))])
		print('len of vec: ', len(vecs[-1]))
	train_vecs = open('train_vecs', 'wb')
	pickle.dump(vecs, train_vecs)
	train_vecs.close()
	train.close()




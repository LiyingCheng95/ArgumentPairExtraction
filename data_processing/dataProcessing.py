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
			
def sep_sent(s):
	nlp = English()

	# sentencizer = nlp.create_pipe("sentencizer")
	# nlp.add_pipe(sentencizer)
	nlp.add_pipe(nlp.create_pipe('sentencizer'))
	s = s.replace('by Yu et al.', 'by Yu et al..').replace('by Wen et al.', 'by Wen et al..').replace('et. al.','et al').replace('et al.,','et al').replace('et al.','et al')
	# s = re.sub('et al. [^A-Z]]','et al [A-Z]',s)
	doc = nlp(s)
	doc = list(doc.sents)
	# doc = [sent.string for sent in doc.sents]
	return doc

def tagging_sequence(outputfile,corpus,arguments_corpus,category,paperid):
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

		i = i.strip() # Added by TY
		if i == '':
			continue


		for j in arguments_corpus:
			# print("j: ",j[1])
			# print("i: ",i)
			# argument = j[1].replace('et. al.','et al ')
			argument = j[1].replace('by Yu et al.', 'by Yu et al..').replace('by Wen et al.', 'by Wen et al..').replace('et. al.','et al').replace('et al.,','et al').replace('et al.','et al')
			argument = argument.strip() # added by TY
			argument = re.sub(r'^\. ', '', argument)
			try:
				sents = sep_sent(argument)
			except:
				print('argument with problem: ', argument)
				break
			if (i in argument \
				or str(sep_sent(argument)[0]) in i) \
				and i!='':
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
				temp += '\t'
				temp += str(paperid)
				temp+='\n'
				outputfile.write(temp)
				flag=1
				break




			# try:
			# 	# if str(sep_sent(argument)[0]) in i:
			# 	# 	print('corpus: ', i)
			# 	# 	print('first sent of arguemnt: ', str(sep_sent(argument)[0]))
			# 	if (i in argument \
			# 		or str(sep_sent(argument)[0]) in i ) \
			# 		and i!='':
			# 		# print(j2tag,j[2])
			# 		if j2tag!=j[2] or bio==0:

			# 			temp+='B-'
			# 			temp+=category
			# 			temp+='\t'
			# 			temp+='B-'
			# 			bio=1
			# 		else:
			# 			temp+='I-'
			# 			temp+=category
			# 			temp+='\t'
			# 			temp+='I-'


			# 		temp+=j[2]
			# 		j2tag=j[2]
			# 		temp += '\t'
			# 		temp += category
			# 		temp += '\t'
			# 		temp += str(paperid)
			# 		temp+='\n'
			# 		outputfile.write(temp)
			# 		flag=1
			# 		break
			# except:
			# 	print('argument with problem: ', argument)
			# 	if i in argument and i!='':
			# 		# print(j2tag,j[2])
			# 		if j2tag!=j[2] or bio==0:

			# 			temp+='B-'
			# 			temp+=category
			# 			temp+='\t'
			# 			temp+='B-'
			# 			bio=1
			# 		else:
			# 			temp+='I-'
			# 			temp+=category
			# 			temp+='\t'
			# 			temp+='I-'


			# 		temp+=j[2]
			# 		j2tag=j[2]
			# 		temp += '\t'
			# 		temp += category
			# 		temp += '\t'
			# 		temp += str(paperid)
			# 		temp+='\n'
			# 		outputfile.write(temp)
			# 		flag=1
			# 		break
		if flag==0:
			temp+='O'
			temp+='\t'
			bio=0

			temp+='O'
			temp += '\t'
			temp += category
			temp += '\t'
			temp += str(paperid)
			temp+='\n'
			outputfile.write(temp)

def update_file1():
	with open('file1new.csv', 'w') as outputfile:
		outputfile = csv.writer(outputfile)
		with open('iclr_pairs (updated).csv') as iclr_pairs:
			iclr_pairs = csv.reader(iclr_pairs, delimiter=',')
			iclr_pairs = list(iclr_pairs)
		
			with open('file1.csv') as csvfile1:
				file1 = csv.reader(csvfile1, delimiter=',') 

				for line in file1:
					# arguments_review=[ file2[index] for index in [i for i, e in enumerate(file2) if e[0] == line[0] and e[-1]=="Review"] ]
					# arguments_reply=[ file2[index] for index in [i for i, e in enumerate(file2) if e[0] == line[0] and e[-1]=="Reply"] ]
					# # print(line)
					# # print(arguments_review)
					sep=line[1].find("——————【reply】——————")
					review=line[1][22:sep-2]
					reply=line[1][sep+21:-1]
					for row in iclr_pairs:
						if review == row[-1]:
							paperid = row[1]
					line.append(paperid)
					outputfile.writerow(line)


def main():
	with open('ReviewRebuttalnew2.txt', 'w') as outputfile:

		with open('file2.csv') as csvfile2:
			file2 = csv.reader(csvfile2, delimiter=',')
			file2=list(file2)
		
			with open('file1new.csv') as csvfile1:
				file1 = csv.reader(csvfile1, delimiter=',') 

				# vecs=[]
				# # sents=[]
			
				for line in file1:

					arguments_review=[ file2[index] for index in [i for i, e in enumerate(file2) if e[0] == line[0] and e[-1]=="Review"] ]
					arguments_reply=[ file2[index] for index in [i for i, e in enumerate(file2) if e[0] == line[0] and e[-1]=="Reply"] ]
					# print(line)
					# print(arguments_review)
					sep=line[1].find("——————【reply】——————")
					review=line[1][22:sep-2]
					reply=line[1][sep+21:-1]
					review=sep_sent(review)
					tagging_sequence(outputfile,review,arguments_review,'Review',line[-1])
					reply=sep_sent(reply)
					tagging_sequence(outputfile,reply,arguments_reply,'Reply',line[-1])
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
	with open("ReviewRebuttalnew2.txt", "r") as full_data:
		full_data = full_data.read().split('\n\n')
		print(len(full_data))
		
		# paperid2pair = defaultdict(list)
		# for passage_pair in full_data:
		# 	id = passage_pair.split('\t')[-1]
		# 	paperid2pair[id].append(passage_pair)

		# # paperid2pair = {passage_pair.split('\t')[-1]: passage_pair for passage_pair in full_data}
		# paperids = list(paperid2pair.keys())

		# random.seed(230)
		# random.shuffle(paperids)

		# split_1 = int(0.82*len(paperids))
		# split_2 = int(0.9*len(paperids))
		# train_ids = paperids[:split_1]
		# dev_ids = paperids[split_1:split_2]
		# test_ids = paperids[split_2:]
		
		# train_data = [passage for id in train_ids for passage in paperid2pair[id]]
		# dev_data = [passage for id in dev_ids for passage in paperid2pair[id]]
		# test_data = [passage for id in test_ids for passage in paperid2pair[id]]

		# print('ratio for train dev test: {}: {}: {}'.format(len(train_data), len(dev_data), len(test_data)))
		

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

def get_paper_id():
	fnew=open('ReviewRebuttalnew2.txt','r').readlines()
	fold=open('ReviewRebuttalnew.txt','r').readlines()
	fout = open('RR.txt','w')
	for i in range(len(fold)):
		if fold[i]!='\n':
			tmp=fold[i][:-1]+'\t'+fnew[i].split('\t')[-1]
			fout.write(tmp)
		else:
			fout.write('\n')
	fout.close()

def combine():
	fnew=open('train1.txt','r').readlines()
	fold=open('train.txt','r').readlines()
	fout = open('checkI2B.txt','w')
	count=0
	for i in range(len(fnew)-1):
		if fnew[i]!='\n' and fnew[i+1]!='\n':
			if fold[i].split('\t')[1][0]=='O' and fnew[i].split('\t')[1][0]=='B' \
			and (fold[i+1].split('\t')[1][0]!='B' or fnew[i+1].split('\t')[1][0]!='I'):
				count+=1
				print(fnew[i])
				tmp=fnew[i].split('\t')[0]+'\t'+fold[i].split('\t')[1] \
				+ '\t' + fold[i].split('\t')[2] \
				+ '\t' + fold[i].split('\t')[3][:-1] \
				+ '\t' + fnew[i].split('\t')[-1]
				fout.write(tmp)
			else:
				fout.write(fnew[i])
		else:
			fout.write(fnew[i])
	print(count)
	fout.close()

def match_diff_level():
    file_seg = open('ReviewRebuttalnew2.txt','r').read().split('\n\n')
    output = open('ReviewRebuttalnew3.txt','w')

    # file1 = csv.reader(open('file1.csv'), delimiter=',')
        
        # print(len(file_seg),len(file1))
    for idx,i in enumerate(file_seg):
        # print(idx)
        # print(i.split('\n')[0].split('\t')[0])
        # print(list(file1))
        flag =0
        with open('file1.csv') as csvfile1:
            file1 = csv.reader(csvfile1, delimiter=',') 
            for j in file1:
                # print(j[1])
                # print(j)
                if len(i.split('\n')[0].split('\t')[0])>30 and j[1].find(i.split('\n')[0].split('\t')[0].replace('[line_break_token]','\n').split('et al')[0])!=-1:
                    for line in i.split('\n'):
                        # print(line)
                        line+='\t'
                        line+=j[0]
                        line+='\n'
                        output.write(line)
                    output.write('\n')
                    flag +=1 
                    break
                else:
                    # print(i.split('\n')[0].split('\t')[0].replace('[line_break_token]',''))
                    if j[1].find(i.split('\n')[1].split('\t')[0].replace('[line_break_token]','\n').split('et al')[0])!=-1:
                        for line in i.split('\n'):
                            # print(line)
                            line+='\t'
                            line+=j[0]
                            line+='\n'
                            output.write(line)
                        output.write('\n')
                        flag += 1
                        break
                    # else:
                        # print(i.split('\n')[0].split('\t')[0].replace('[line_break_token]',''))
        if flag !=1:
            print(flag,idx)
            print(i.split('\n')[0].split('\t')[0])

    output.close()

def add_id():
	file1 = open('RR.txt','r').readlines()
	csvfile2 = open('file2.csv')
	file2 = csv.reader(csvfile2, delimiter=',')
	file2 = list(file2)
	output = open('ReviewRebuttalnew3.txt','w')
	id = 0
	for i in range(len(file1)):
		if file1[i]!='\n':
			output.write(file2[id][0] + '\t' + file1[i])
		else:
			id+=2
			output.write(file1[i])




def update_label():
	with open('ReviewRebuttalnew2.txt', 'w') as outputfile:

		csvfile2 = open('file2.csv')
		file2 = csv.reader(csvfile2, delimiter=',')
		file2=list(file2)

		for i in range(len(file2)):
			file2[i][1]=file2[i][1].replace('by Yu et al.', 'by Yu et al..').replace('by Wen et al.', 'by Wen et al..').replace('et. al.','et al').replace('et al.,','et al').replace('et al.','et al').replace('\n','[line_break_token]').replace('\t','[tab_token]').split('.')[0]
	
		original =open('ReviewRebuttalnew3.txt','r').readlines()
		flag=0
		next_flag=0
	
		for i in range(len(original)-1):
			flag=0

			if next_flag==1:
				next_flag=0
				continue

			if original[i]=='\n':
				outputfile.write(original[i])
				continue

			if original[i].split('\t')[2]=='O':
				file2_filter = [ file2[index] for index in [i for i, e in enumerate(file2) if e[0] == original[i].split('\t')[0] ] ]
				# print(type(file2_filter))
				for j in range(len(file2_filter)):
					# print(file2[j][0], original[i].split('\t')[0])
					a = file2_filter[j][1].replace('[line_break_token]','').replace('[tab_token]','')
					b = original[i].split('\t')[1].replace('[line_break_token]','').replace('[tab_token]','')
					if a in b and len(a)>=10:
						label='B-'+file2_filter[j][2]
						outputfile.write(original[i].split('\t')[1]+'\t'\
							+ label + '\t' \
							+ 'B-' + original[i].split('\t')[4] +'\t' \
							+ original[i].split('\t')[4] +'\t' \
							+ original[i].split('\t')[5])
						print("--------a--------:",a,"--------b--------:",b)
						flag=1

						if original[i].split('\t')[2]==label:
							next_label='I-'+file2_filter[j][2]
							outputfile.write(original[i+1].split('\t')[1]+'\t'\
								+ nextlabel + '\t' \
								+ 'I-' + original[i+1].split('\t')[4] +'\t' \
								+ original[i+1].split('\t')[4] +'\t' \
								+ original[i+1].split('\t')[5])
							next_flag=1
			if flag==0:
				outputfile.write(original[i])


def match_train():
	file_all = open('ReviewRebuttalnew2.txt','r').readlines()
	file_part = open('train.txt','r').readlines()
	file_new = open('train1.txt','w')
	for i in file_part:
		if i!='\n':
			for j in file_all:
				if i.split('\t')[0] == j.split('\t')[0] \
				and i.split('\t')[-1] == j.split('\t')[-1] \
				and i.split('\t')[-2] == j.split('\t')[-2]:
					file_new.write(j)
					break
		else:
			file_new.write(i)

from collections import defaultdict
def match_train_fast():
	file_all = open('ReviewRebuttalnew2.txt','r').readlines() # confirm correct
	file_old = open('test.txt','r').readlines() # Old, might be wrong
	file_new = open('test1.txt','w') # follow the same order, but the content is from file_all

	dict_all = defaultdict(list)
	for i in range(len(file_all)):
		dict_all[tuple(file_all[i].split('\t')[-2:])].append(i)
	for line in file_old:
		flag = False
		for ind in dict_all[tuple(line.split('\t')[-2:])]: # iterate over limited indices of correct file
			if file_all[ind].split('\t')[0].strip() == line.split('\t')[0].strip():
				file_new.write(file_all[ind])
				flag = True
				break
		if not flag:
			print('not found: ', line)

def check_wrong():
	csvfile2 = open('file2.csv')
	file2 = csv.reader(csvfile2, delimiter=',')
	file2 = list(file2)
	wrong_arguments = [row[1] for row in file2 if re.findall(r'^\S+[.]\s+', row[1])]
	print(wrong_arguments)
	print('count: ', len(wrong_arguments))


if __name__=='__main__':
	# argument = '''. What is missing is better motivating/explaining why these results are interesting/relevant. Sure, the GDBM reproduces certain experimental findings. But have we learned something new about the brain? Is the GDBM particularly well suited as a general model of visual cortex? Does the model make predictions? What about alternative, perhaps simpler models that could have been used instead? Etc.Also, are there related models or theoretical approaches to spontaneous activity?'''
	# argument = re.sub(r'^. ', '', argument)
	# print(argument)
	# print(sep_sent(''', sparse DBNs show worse match to the biological findings as reported in [1]. A quantitative comparison between centered GDBMs and sparse DBNs is still open for future studies.' Where was it shown to be a worse match then?'''))	
	# main()
	# sep_sent('Done')
	# check_wrong()
	sep_data()
	# get_paper_id()
	# print(sep_sent('''.	We have replaced ‚ÄúWord frequency‚Äù with ‚ÄúWord ranking by frequency‚Äù'''))
	# print(sep_sent('''Details about the corresponding association and communication events in the two datasets are provided in Appendix E.1. We uploaded a revised version that contains your suggested changes.'''))
	# combine()
	# update_file1()
	# match_diff_level()
	# add_id()
	# update_label()
	# match_train_fast()


import numpy as np
import torch
from typing import List, Tuple
from common import Instance
import pickle
import torch.optim as optim

import torch.nn as nn



from config import PAD, ContextEmb, Config
from termcolor import colored

def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))

def batching_list_instances(config: Config, insts: List[Instance]):
    train_num = len(insts)
    batch_size = config.batch_size
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(simple_batching(config, one_batch_insts))

    return batched_data

def simple_batching(config, insts: List[Instance]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    """
    batching these instances together and return tensors. The seq_tensors for word and char contain their word id and char id.
    :return
        sent_emb_tensor: Shape: (batch_size, max_seq_len, emb_size)
        # word_seq_tensor: Shape: (batch_size, max_seq_length)
        sent_seq_len: Shape: (batch_size), the length of each paragraph in a batch.
        context_emb_tensor: Shape: (batch_size, max_seq_length, context_emb_size)
        char_seq_tensor: Shape: (batch_size, max_seq_len, max_char_seq_len)
        char_seq_len: Shape: (batch_size, max_seq_len), 
        label_seq_tensor: Shape: (batch_size, max_seq_length)
    """
    batch_size = len(insts)
    batch_data = insts
    # probably no need to sort because we will sort them in the model instead.
    # batch_data = sorted(insts, key=lambda inst: len(inst.input.words), reverse=True) ##object-based not direct copy
    sent_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.sents), batch_data)))
    max_seq_len = sent_seq_len.max()

    # NOTE: Use 1 here because the CharBiLSTM accepts
    char_seq_len = torch.LongTensor([list(map(len, inst.input.sents)) + [1] * (int(max_seq_len) - len(inst.input.sents)) for inst in batch_data])
    max_char_seq_len = char_seq_len.max()

    context_emb_tensor = None
    if config.context_emb != ContextEmb.none:
        emb_size = insts[0].elmo_vec.shape[1]
        context_emb_tensor = torch.zeros((batch_size, max_seq_len, emb_size))

    # emb_size = len(insts[0][0].vec)
    # print('emb_size: ',emb_size)
    emb_size = 768

    # word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_seq_tensor =  torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    review_idx_tensor = torch.full((batch_size, max_seq_len),0, dtype=torch.long)
    reply_idx_tensor = torch.full((batch_size, max_seq_len),0, dtype=torch.long)

    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_char_seq_len), dtype=torch.long)

    sent_emb_tensor = torch.zeros((batch_size, max_seq_len, emb_size), dtype=torch.float32)
    # input = torch.zeros((batch_size, num_sents, emb_size))

    type_id_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

    pair_tensor = torch.zeros((batch_size,max_seq_len,max_seq_len), dtype = torch.float32)
    pair_padding_tensor = torch.zeros((batch_size, max_seq_len, max_seq_len), dtype=torch.float32)

    max_review_tensor = torch.zeros((batch_size), dtype=torch.long)

    for idx in range(batch_size):


        # word_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].word_ids)
        if batch_data[idx].output_ids:
            # print('output_ids:   ', batch_data[idx].output_ids)
            # print('review_idx:   ', batch_data[idx].review_idx)
            # print(sent_seq_len[idx])
            # print("batch_data[idx].max_review_id:   ",batch_data[idx].max_review_id )
            max_review_tensor[idx]=batch_data[idx].max_review_id
            label_seq_tensor[idx, :sent_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)
            review_idx_tensor[idx, :len(batch_data[idx].review_idx)] = torch.LongTensor(batch_data[idx].review_idx)
            reply_idx_tensor[idx, len(batch_data[idx].review_idx):] = torch.LongTensor(batch_data[idx].reply_idx)
            type_id_tensor[idx, :sent_seq_len[idx]] = torch.LongTensor(batch_data[idx].type)
        if config.context_emb != ContextEmb.none:
            context_emb_tensor[idx, :sent_seq_len[idx], :] = torch.from_numpy(batch_data[idx].elmo_vec)
        
        for sent_idx in range(sent_seq_len[idx]):
            sent_emb_tensor[idx, sent_idx, :emb_size] = torch.Tensor(batch_data[idx].vec[sent_idx])

            # print('sent_emb_tensor', sent_emb_tensor[idx, sent_idx, 0])
            char_seq_tensor[idx, sent_idx, :char_seq_len[idx, sent_idx]] = torch.LongTensor(batch_data[idx].char_ids[sent_idx])

            if sent_idx < batch_data[idx].max_review_id:
                for sent_idx2 in range(sent_idx+1, sent_seq_len[idx]):
                    if batch_data[idx].labels_pair[sent_idx] == batch_data[idx].labels_pair[sent_idx2] \
                            and batch_data[idx].labels_pair[sent_idx] != 0 \
                            and batch_data[idx].type[sent_idx] != batch_data[idx].type[sent_idx2]:
                        pair_tensor[idx,sent_idx,sent_idx2]=1.0
                    if batch_data[idx].type[sent_idx]!= batch_data[idx].type[sent_idx2]:
                        pair_padding_tensor[idx,sent_idx,sent_idx2]=1.0
        # print("sum:", pair_padding_tensor)
        pair_tensor[pair_padding_tensor==0] = -100

        # print(pair_tensor[idx,])
        for sentIdx in range(sent_seq_len[idx], max_seq_len):
            char_seq_tensor[idx, sentIdx, 0: 1] = torch.LongTensor([config.char2idx[PAD]])   ###because line 119 makes it 1, every single character should have a id. but actually 0 is enough

    # word_seq_tensor = word_seq_tensor.to(config.device)
    label_seq_tensor = label_seq_tensor.to(config.device)
    char_seq_tensor = char_seq_tensor.to(config.device)
    sent_seq_len = sent_seq_len.to(config.device)
    char_seq_len = char_seq_len.to(config.device)

    # sent_emb_tensor = sent_emb_tensor.to(config.device)
    type_id_tensor = type_id_tensor.to(config.device)

    review_idx_tensor = review_idx_tensor.to(config.device)
    reply_idx_tensor = reply_idx_tensor.to(config.device)

    pair_tensor = pair_tensor.to(config.device)

    return sent_emb_tensor, type_id_tensor, sent_seq_len, context_emb_tensor, char_seq_tensor, char_seq_len, label_seq_tensor, review_idx_tensor, reply_idx_tensor, pair_tensor, pair_padding_tensor, max_review_tensor


def lr_decay(config, optimizer: optim.Optimizer, epoch: int) -> optim.Optimizer:
    """
    Method to decay the learning rate
    :param config: configuration
    :param optimizer: optimizer
    :param epoch: epoch number
    :return:
    """
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer


def load_elmo_vec(file: str, insts: List[Instance]):
    """
    Load the elmo vectors and the vector will be saved within each instance with a member `elmo_vec`
    :param file: the vector files for the ELMo vectors
    :param insts: list of instances
    :return:
    """
    f = open(file, 'rb')
    all_vecs = pickle.load(f)  # variables come out in the order you put them in
    f.close()
    size = 0
    for vec, inst in zip(all_vecs, insts):
        inst.elmo_vec = vec
        size = vec.shape[1]
        assert(vec.shape[0] == len(inst.input.sents))
    return size



def get_optimizer(config: Config, model: nn.Module):
    params = model.parameters()
    if config.optimizer.lower() == "sgd":
        print(
            colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params, lr=config.learning_rate, weight_decay=float(config.l2))
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optim.Adam(params, config.learning_rate)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)



def write_results(filename: str, insts):
    f = open(filename, 'w', encoding='utf-8')
    for inst in insts:
        for i in range(len(inst.input)):
            sents = inst.input.ori_sents
            output = inst.output
            prediction = inst.prediction
            gold2 = inst.gold2
            pred2 = inst.pred2
            assert len(output) == len(prediction)
            f.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(i, sents[i], output[i], prediction[i], gold2[i], pred2[i]))
        f.write("\n")
    f.close()
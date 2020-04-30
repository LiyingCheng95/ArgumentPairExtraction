import argparse
import random
import numpy as np
from config import Reader, Config, ContextEmb, lr_decay, simple_batching, evaluate_batch_insts, get_optimizer, write_results, batching_list_instances, evaluate_batch_insts_e2e
import time
from modelrr.neuralcrf import NNCRF
import torch
from typing import List
from common import Instance
from termcolor import colored
import os
from config.utils import load_elmo_vec
import pickle
import tarfile
import shutil
from tqdm import tqdm

def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="rr")
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt",
                        help="we will be using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=788)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--learning_rate', type=float, default=0.01)  ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10, help="default batch size is 10 (works well)")
    parser.add_argument('--num_epochs', type=int, default=200, help="Usually we set to 10.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--max_no_incre', type=int, default=100, help="early stop when there is n epoch not increasing on dev")

    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=0, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--context_emb', type=str, default="none", choices=["none", "elmo"],
                        help="contextual word embedding")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, epoch: int, train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance]):
    model = NNCRF(config)
    optimizer = get_optimizer(config, model)
    train_num = len(train_insts)
    print("number of instances: %d" % (train_num))
    print(colored("[Shuffled] Shuffle the training instance ids", "red"))
    random.shuffle(train_insts)

    batched_data = batching_list_instances(config, train_insts)
    dev_batches = batching_list_instances(config, dev_insts)
    test_batches = batching_list_instances(config, test_insts)

    best_dev = [-1, 0, -1]
    best_test = [-1, 0, -1]

    model_folder = config.model_folder
    res_folder = "results"
    if os.path.exists("model_files/" + model_folder):
        raise FileExistsError(
            f"The folder model_files/{model_folder} exists. Please either delete it or create a new one "
            f"to avoid override.")
    model_path = f"model_files/{model_folder}/lstm_crf.m"
    config_path = f"model_files/{model_folder}/config.conf"
    res_path = f"{res_folder}/{model_folder}.results"
    print("[Info] The model will be saved to: %s.tar.gz" % (model_folder))
    os.makedirs(f"model_files/{model_folder}", exist_ok= True) ## create model files. not raise error if exist
    os.makedirs(res_folder, exist_ok=True)
    no_incre_dev = 0
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for index in np.random.permutation(len(batched_data)):
            processed_batched_data = simple_batching(config,batched_data[index])
            model.train()
            loss = model(*processed_batched_data)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()

        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        model.eval()
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
        # print(test_insts.prediction)
        if dev_metrics[2] > best_dev[0] or (dev_metrics[2] == best_dev[0] and dev_metrics[-2] > best_dev[-1]): # task 1 & task 2
        # if dev_metrics[-2] > best_dev[-1]: # task 2
        # if dev_metrics[2] > best_dev[0]: # task 1
            print("saving the best model...")
            no_incre_dev = 0
            best_dev[0] = dev_metrics[2]
            best_dev[-1] = dev_metrics[-2]
            best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[-1] = test_metrics[-2]
            best_test[1] = i
            torch.save(model.state_dict(), model_path)
            # Save the corresponding config as well.
            f = open(config_path, 'wb')
            pickle.dump(config, f)
            f.close()
            write_results(res_path, test_insts)
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break

    print("Archiving the best Model...")
    with tarfile.open(f"model_files/{model_folder}/{model_folder}.tar.gz", "w:gz") as tar:
        tar.add(f"model_files/{model_folder}", arcname=os.path.basename(model_folder))

    print("Finished archiving the models")

    print("The best dev: %.2f" % (best_dev[0]))
    print("The corresponding test: %.2f" % (best_test[0]))
    print("Final testing.")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    evaluate_model(config, model, test_batches, "test", test_insts)
    write_results(res_path, test_insts)


def evaluate_model(config: Config, model: NNCRF, batch_insts_ids, name: str, insts: List[Instance]):
    ## evaluation
    tp, fp, tn, fn = 0, 0, 0, 0
    metrics, metrics_e2e = np.asarray([0, 0, 0], dtype=int), np.asarray([0, 0, 0], dtype=int)
    pair_metrics = np.asarray([0, 0, 0], dtype=int)
    batch_idx = 0
    batch_size = config.batch_size
    # print('insts',len(insts))
    for batch in batch_insts_ids:
        # print('batch_idx * batch_size:(batch_idx + 1) * batch_size', batch_idx* batch_size,(batch_idx + 1) * batch_size )
        one_batch_insts = insts[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        processed_batched_data = simple_batching(config, batch)
        # print(len(one_batch_insts))
        batch_max_scores, batch_max_ids, pair_ids = model.decode(processed_batched_data)

        metrics += evaluate_batch_insts(one_batch_insts, batch_max_ids, processed_batched_data[-6], processed_batched_data[2], config.idx2labels)
        # print(processed_batched_data[-1])
        metrics_e2e += evaluate_batch_insts_e2e(one_batch_insts, batch_max_ids, processed_batched_data[-6], processed_batched_data[2], config.idx2labels, processed_batched_data[-8], pair_ids, processed_batched_data[-1])

        word_seq_lens = processed_batched_data[2].tolist()
        for batch_id in range(batch_max_ids.size()[0]):
            # print('batch_max_ids[batch_id]:  ',batch_max_ids[batch_id].size(),batch_max_ids[batch_id])
            length = word_seq_lens[batch_id]
            # prediction = batch_max_ids[batch_id][:length]
            # prediction = torch.flip(prediction,dims = [0])

            gold = processed_batched_data[-6][batch_id][:length]
            # gold = torch.flip(gold, dims=[0])

            # s_id = (prediction == 2).nonzero()
            # b_id = (prediction == 3).nonzero()
            # e_id = (prediction == 4).nonzero()
            # i_id = (prediction == 5).nonzero()
            # pred_id = torch.cat([s_id, b_id, e_id, i_id]).squeeze(1)
            # pred_id,_ = pred_id.sort(0, descending=False)
            # pred_id = pred_id[pred_id < processed_batched_data[-1][batch_id]]

            s_id = (gold == 2).nonzero()
            b_id = (gold == 3).nonzero()
            e_id = (gold == 4).nonzero()
            i_id = (gold == 5).nonzero()
            gold_id = torch.cat([s_id, b_id, e_id, i_id]).squeeze(1)
            gold_id, _ = gold_id.sort(0, descending=False)
            gold_id = gold_id[gold_id < processed_batched_data[-1][batch_id]]

            # argu_id = torch.LongTensor(list(set(gold_id.tolist()).intersection(set(pred_id.tolist()))))
            argu_id = torch.LongTensor(list(set(gold_id.tolist())))
            # print('gold_id', gold_id, 'pred_id', pred_id, 'argu_id', argu_id)

            # print(pair_ids[batch_id].size(), batch[-3][batch_id].size())
            one_batch_insts[batch_id].pred2 = pair_ids[batch_id].squeeze(2)
            one_batch_insts[batch_id].gold2 = processed_batched_data[-3][batch_id]

            # print(one_batch_insts[batch_id].gold2)
            # print(torch.sum(one_batch_insts[batch_id].pred2, dim=1))

            pred2 = one_batch_insts[batch_id].pred2[argu_id]
            gold2 = one_batch_insts[batch_id].gold2[argu_id]


            # print('argu_id:  ',argu_id.size(),argu_id)
            # print('one_batch_insts[batch_id].pred2:  ',one_batch_insts[batch_id].pred2.size(),one_batch_insts[batch_id].pred2)

            gold_pairs = gold2.flatten()
            pred_pairs = pred2.flatten()

            # print(gold_pairs,pred_pairs)
            sum_table = gold_pairs + pred_pairs
            # print(sum_table.size(),sum_table[:100])
            sum_table_sliced = sum_table[sum_table >= 0]
            # print(sum_table_sliced.size(),sum_table_sliced)
            tp_tmp = len(sum_table_sliced[sum_table_sliced == 2])
            tn_tmp = len(sum_table_sliced[sum_table_sliced == 0])
            tp += tp_tmp
            tn += tn_tmp
            ones = len(gold_pairs[gold_pairs == 1])
            zeros = len(gold_pairs[gold_pairs == 0])
            fp += (zeros - tn_tmp)
            fn += (ones - tp_tmp)
            # print(tp,tp_tmp,tn,tn_tmp,ones,zeros,fp,fn)


        batch_idx += 1
    print('tp, fp, fn, tn: ', tp, fp, fn, tn)
    precision_2 = 1.0 * tp / (tp + fp) * 100 if tp + fp != 0 else 0
    recall_2 = 1.0 * tp / (tp + fn) * 100 if tp + fn != 0 else 0
    f1_2 = 2.0 * precision_2 * recall_2 / (precision_2 + recall_2) if precision_2 + recall_2 != 0 else 0
    acc = 1.0 *(tp+tn)/(fp+fn+tp+tn) * 100 if fp+fn+tp+tn!=0 else 0
    p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    p_e2e, total_predict_e2e, total_entity_e2e = metrics_e2e[0], metrics_e2e[1], metrics_e2e[2]
    precision_e2e = p_e2e * 1.0 / total_predict_e2e * 100 if total_predict_e2e != 0 else 0
    recall_e2e = p_e2e * 1.0 / total_entity_e2e * 100 if total_entity_e2e != 0 else 0
    fscore_e2e = 2.0 * precision_e2e * recall_e2e / (precision_e2e + recall_e2e) if precision_e2e != 0 or recall_e2e != 0 else 0


    print("Task1: [%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore), flush=True)
    print("Task2: [%s set] Precision: %.2f, Recall: %.2f, F1: %.2f, acc: %.2f" % (name, precision_2, recall_2, f1_2, acc), flush=True)
    print("Overall: [%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision_e2e, recall_e2e, fscore_e2e), flush=True)
    return [precision, recall, fscore, precision_2, recall_2, f1_2, acc, precision_e2e, recall_e2e, fscore_e2e]


def main():
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    reader = Reader(conf.digit2zero)
    set_seed(opt, conf.seed)

    trains = reader.read_txt(conf.train_file, conf.train_num)
    devs = reader.read_txt(conf.dev_file, conf.dev_num)
    tests = reader.read_txt(conf.test_file, conf.test_num)

    if conf.context_emb != ContextEmb.none:
        print('Loading the ELMo vectors for all datasets.')
        conf.context_emb_size = load_elmo_vec(conf.train_file + "." + conf.context_emb.name + ".vec", trains)
        load_elmo_vec(conf.dev_file + "." + conf.context_emb.name + ".vec", devs)
        load_elmo_vec(conf.test_file + "." + conf.context_emb.name + ".vec", tests)

    conf.use_iobes(trains)
    conf.use_iobes(devs)
    conf.use_iobes(tests)
    conf.build_label_idx(trains + devs + tests)

    conf.build_word_idx(trains, devs, tests)
    conf.build_emb_table()

    conf.map_insts_ids(trains)
    conf.map_insts_ids(devs)
    conf.map_insts_ids(tests)

    print("num chars: " + str(conf.num_char))
    # print(str(config.char2idx))

    print("num words: " + str(len(conf.word2idx)))
    # print(config.word2idx)
    train_model(conf, conf.num_epochs, trains, devs, tests)


if __name__ == "__main__":
    main()

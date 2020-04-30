
import numpy as np
from overrides import overrides
from typing import List
from common import Instance
import torch
from sklearn.metrics import precision_score, recall_score, f1_score


class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))

class Span_e2e:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type
        # return self.left == other.left and self.right == other.right and ((self.type[self.type+other.type >= 0]) == (other.type[self.type+other.type >= 0])).all()

    def __hash__(self):
        return hash((self.left, self.right, self.type))



def evaluate_batch_insts(batch_insts: List[Instance],
                         batch_pred_ids: torch.LongTensor,
                         batch_gold_ids: torch.LongTensor,
                         word_seq_lens: torch.LongTensor,
                         idx2label: List[str]) -> np.ndarray:
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """
    p = 0
    p_task2 = 0
    total_entity = 0
    total_predict = 0
    word_seq_lens = word_seq_lens.tolist()
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        output = [idx2label[l] for l in output]
        prediction =[idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction
        #convert to span

        # gold
        output_spans = set()
        start = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                output_spans.add(Span(start, end, output[i][2:]))
            if output[i].startswith("S-"):
                output_spans.add(Span(i, i, output[i][2:]))


        # predict
        predict_spans = set()
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
            if prediction[i].startswith("S-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))

        total_entity += len(output_spans)
        total_predict += len(predict_spans)
        p += len(predict_spans.intersection(output_spans))


    # In case you need the following code for calculating the p/r/f in a batch.
    # (When your batch is the complete dataset)
    # precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    # recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    return np.asarray([p, total_predict, total_entity], dtype=int)

def evaluate_batch_insts_e2e_old(batch_insts: List[Instance],
                         batch_pred_ids: torch.LongTensor,
                         batch_gold_ids: torch.LongTensor,
                         word_seq_lens: torch.LongTensor,
                         idx2label: List[str],
                         pair_gold: torch.Tensor,
                         pair_predict: torch.Tensor) -> np.ndarray:
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """
    p = 0
    total_entity = 0
    total_predict = 0
    word_seq_lens = word_seq_lens.tolist()
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        output = [idx2label[l] for l in output]
        prediction = [idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction
        # convert to span

        # gold
        output_spans = set()
        start = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                pairs = pair_gold[idx][start: end+1][pair_gold[idx][start: end+1]+ pair_predict[idx].squeeze(2)[start: end+1]>=0].tolist()
                pairs = ''.join(str(int(e)) for e in pairs)
                output_spans.add(Span_e2e(start, end, pairs))
            if output[i].startswith("S-"):
                pairs = pair_gold[idx][i][pair_gold[idx][i] + pair_predict[idx].squeeze(2)[i] >= 0].tolist()
                pairs = ''.join(str(int(e)) for e in pairs)
                output_spans.add(Span_e2e(i, i, pairs))


        # predict
        predict_spans = set()
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                pairs = pair_predict[idx].squeeze(2)[start: end+1][pair_gold[idx][start: end+1]+ pair_predict[idx].squeeze(2)[start: end+1]>=0].tolist()
                pairs = ''.join(str(int(e)) for e in pairs)
                predict_spans.add(Span_e2e(start, end, pairs))
            if prediction[i].startswith("S-"):
                pairs = pair_predict[idx].squeeze(2)[i][pair_gold[idx][i] + pair_predict[idx].squeeze(2)[i] >= 0].tolist()
                pairs = ''.join(str(int(e)) for e in pairs)
                predict_spans.add(Span_e2e(i, i, pairs))

        total_entity += len(output_spans)
        total_predict += len(predict_spans)
        p += len(predict_spans.intersection(output_spans))


    # In case you need the following code for calculating the p/r/f in a batch.
    # (When your batch is the complete dataset)
    # precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    # recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    return np.asarray([p, total_predict, total_entity], dtype=int)



def evaluate_batch_insts_e2e(batch_insts: List[Instance],
                         batch_pred_ids: torch.LongTensor,
                         batch_gold_ids: torch.LongTensor,
                         word_seq_lens: torch.LongTensor,
                         idx2label: List[str],
                         pair_gold: torch.Tensor,
                         pair_predict: torch.Tensor,
                         num_review: torch.Tensor) -> np.ndarray:
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """
    p = 0
    total_entity = 0
    total_predict = 0
    word_seq_lens = word_seq_lens.tolist()
    # print(num_review)
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1]
        output = [idx2label[l] for l in output]
        prediction = [idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction
        # convert to span

        # gold
        output_spans = set()
        start = -1

        pair_gold1 = pair_gold[idx]
        pair_pred1 = pair_predict[idx].squeeze(2)
        for i in range(num_review[idx], len(output)):
            # if pair_gold1[i] ==-100:
            #     pair_gold1[i] = 0

            if output[i].startswith("O"):
                pair_gold1[i] = -100
                pair_pred1[i] = -100

        for i in range(num_review[idx]):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                # print('gold 2', pair_gold1[start: end+1])
                pair_index = [j for j, e in enumerate(pair_gold1[start].tolist()) if e == 1]
                # print('gold2',pair_index)
                pairs = ''.join(str(int(e)) for e in pair_index)
                output_spans.add(Span_e2e(start, end, pairs))
                # print('gold',pairs)
            if output[i].startswith("S-"):
                pair_index = [j for j, e in enumerate(pair_gold1[i].tolist()) if e == 1]
                pairs = ''.join(str(int(e)) for e in pair_index)
                output_spans.add(Span_e2e(i, i, pairs))
                # print('gold',pairs)

        # predict
        predict_spans = set()
        for i in range(num_review[idx]):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                # print('pred2',pair_pred1[start: end + 1])
                pair_index = [j for j, e in enumerate(pair_pred1[start].tolist()) if e == 1]
                pairs = ''.join(str(int(e)) for e in pair_index)
                predict_spans.add(Span_e2e(start, end, pairs))
                # print('pred', pairs)
            if prediction[i].startswith("S-"):
                # print('pred1', pair_pred1[i])
                pair_index = [j for j, e in enumerate(pair_pred1[i].tolist()) if e == 1]
                pairs = ''.join(str(int(e)) for e in pair_index)
                predict_spans.add(Span_e2e(i, i, pairs))
                # print('pred', pairs)

        # print('gold',output_spans)
        # print('pred',predict_spans)
        total_entity += len(output_spans)
        total_predict += len(predict_spans)
        p += len(predict_spans.intersection(output_spans))
        print(total_entity, total_predict, p)


    # In case you need the following code for calculating the p/r/f in a batch.
    # (When your batch is the complete dataset)
    # precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    # recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    return np.asarray([p, total_predict, total_entity], dtype=int)





def evaluate_pairs(batch_insts: List[Instance],
                         pair_ids: torch.LongTensor,
                         gold_ids: torch.LongTensor,
                         ) -> np.ndarray:
    """
    not in use
    """

    pair_ids = pair_ids.view(-1,1)
    gold_ids = gold_ids.view(-1,1)
    precision = precision_score(gold_ids,pair_ids)
    recall = recall_score(gold_ids,pair_ids)
    f1 = f1_score(gold_ids,pair_ids)
    return f1, precision, recall

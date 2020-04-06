

import torch.nn as nn
import torch
import torch.nn.functional as F

from config import log_sum_exp_pytorch, START, STOP, PAD
from typing import Tuple
from overrides import overrides

class LinearCRF(nn.Module):


    def __init__(self, config, print_info: bool = True):
        super(LinearCRF, self).__init__()

        self.label_size = config.label_size
        self.device = config.device
        self.use_char = config.use_char_rnn
        self.context_emb = config.context_emb

        self.label2idx = config.label2idx
        self.labels = config.idx2labels
        self.start_idx = self.label2idx[START]
        self.end_idx = self.label2idx[STOP]
        self.pad_idx = self.label2idx[PAD]

        # initialize the following transition (anything never -> start. end never -> anything. Same thing for the padding label)
        init_transition = torch.randn(self.label_size, self.label_size).to(self.device)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0

        self.transition = nn.Parameter(init_transition)

        final_hidden_dim = config.hidden_dim
        # self.pair2score = nn.Linear(final_hidden_dim * 2, 1).to(self.device)
        self.pair2score_first = nn.Linear(final_hidden_dim * 2, final_hidden_dim).to(self.device)
        # self.pair2score_second = nn.Linear(final_hidden_dim, 100).to(self.device)
        self.pair2score_final = nn.Linear(final_hidden_dim, 2).to(self.device)

    @overrides
    def forward(self, lstm_scores, pair_scores, word_seq_lens, tags, mask, pairs, pair_padding):
        """
        Calculate the negative log-likelihood
        :param lstm_scores:
        :param word_seq_lens:
        :param tags:
        :param mask:
        :return:
        """
        all_scores=  self.calculate_all_scores(lstm_scores= lstm_scores)
        unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, mask)

        pair_loss = self.calculate_pair_loss(y = pairs.view(-1).long(), pair_scores = pair_scores, pair_padding = pair_padding)
        # all_pair_scores = self.calculate_pair_scores(pair_scores= pair_scores)
        # unlabed_pair_score = self.forward_pair_unlabeled(all_pair_scores, word_seq_lens)
        # labeled_pair_score = self.forward_labeled(all_pair_scores, word_seq_lens, tags, mask)

        return unlabed_score, labeled_score, pair_loss

    def calculate_pair_loss(self, y: torch.Tensor, pair_scores: torch.Tensor, pair_padding: torch.Tensor) -> torch.Tensor:
        # print(pair_scores)

        pair_scores = pair_scores.view(-1, 2)
        criterion = nn.CrossEntropyLoss(reduction='mean')
        # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.1]), reduction='sum')
        # criterion = nn.BCEWithLogitsLoss(reduction='sum')
        # loss = criterion(pair_scores, y.unsqueeze(3))
        loss = criterion(pair_scores, y)
        return loss


    def forward_unlabeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels x num_labels) from (lstm scores + transition scores).
        :param word_seq_lens: (batch_size)
        :return: The score for all the possible structures.
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)

        alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :] ## the first position of all labels = (the transition from start - > all labels) + current emission.

        for word_idx in range(1, seq_len):
            ## batch_size, self.label_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx-1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        ## final score for the unlabeled network in this batch, with size: 1
        return torch.sum(last_alpha)

    def backward(self, lstm_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Backward algorithm. A benchmark implementation which is ready to use.
        :param lstm_scores: shape: (batch_size, sent_len, label_size) NOTE: the score from LSTMs, not `all_scores` (which add up the transtiion)
        :param word_seq_lens: shape: (batch_size,)
        :return: Backward variable
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        beta = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)

        ## reverse the view of computing the score. we look from behind
        rev_score = self.transition.transpose(0, 1).view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                    lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)

        ## The code below, reverse the score from [0 -> length]  to [length -> 0].  (NOTE: we need to avoid reversing the padding)
        perm_idx = torch.zeros(batch_size, seq_len).to(self.device)
        for batch_idx in range(batch_size):
            perm_idx[batch_idx][:word_seq_lens[batch_idx]] = torch.range(word_seq_lens[batch_idx] - 1, 0, -1)
        perm_idx = perm_idx.long()
        for i, length in enumerate(word_seq_lens):
            rev_score[i, :length] = rev_score[i, :length][perm_idx[i, :length]]

        ## backward operation
        beta[:, 0, :] = rev_score[:, 0, self.end_idx, :]
        for word_idx in range(1, seq_len):
            before_log_sum_exp = beta[:, word_idx - 1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + rev_score[:, word_idx, :, :]
            beta[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ## Following code is used to check the backward beta implementation
        last_beta = torch.gather(beta, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size) - 1).view(batch_size, self.label_size)
        last_beta += self.transition.transpose(0, 1)[:, self.start_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_beta = log_sum_exp_pytorch(last_beta.view(batch_size, self.label_size, 1)).view(batch_size)

        # This part if optionally, if you only use `last_beta`.
        # Otherwise, you need this to reverse back if you also need to use beta
        for i, length in enumerate(word_seq_lens):
            beta[i, :length] = beta[i, :length][perm_idx[i, :length]]

        return torch.sum(last_beta)

    def forward_labeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, tags: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the scores for the gold instances.
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param tags: (batch, seq_len)
        :param masks: batch, seq_len
        :return: sum of score for the gold sequences Shape: (batch_size)
        '''
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]

        ## all the scores to current labels: batch, seq_len, all_from_label?
        currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, self.label_size, 1)).view(batchSize, -1, self.label_size)
        if sentLength != 1:
            tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tags[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,  endTagIds).view(batchSize)
        score = torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresEnd)
        if sentLength != 1:
            score += torch.sum(tagTransScoresMiddle.masked_select(masks[:, 1:]))
        return score

    def calculate_all_scores(self, lstm_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate all scores by adding up the transition scores and emissions (from lstm).
        Basically, compute the scores for each edges between labels at adjacent positions.
        This score is later be used for forward-backward inference
        :param lstm_scores: emission scores.
        :return:
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)
        return scores

    def calculate_pair_scores(self, pair_scores: torch.Tensor) -> torch.Tensor:
        batch_size = pair_scores.size(0)
        seq_len = pair_scores.size(1)
        scores = pair_scores.view(batch_size, seq_len, 1, 1, 2).expand(batch_size, seq_len, seq_len, 2, 2)
        return scores

    def decode(self, features, wordSeqLengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.viterbi_decode(all_scores, wordSeqLengths)
        return bestScores, decodeIdx

    def viterbi_decode(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use viterbi to decode the instances given the scores and transition parameters
        :param all_scores: (batch_size x max_seq_len x num_labels)
        :param word_seq_lens: (batch_size)
        :return: the best scores as well as the predicted label ids.
               (batch_size) and (batch_size x max_seq_len)
        """
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        # sent_len =
        scoresRecord = torch.zeros([batchSize, sentLength, self.label_size]).to(self.device)
        idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64).to(self.device)
        mask = torch.ones_like(word_seq_lens, dtype=torch.int64).to(self.device)
        startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64).to(self.device)
        decodeIdx = torch.LongTensor(batchSize, sentLength).to(self.device)

        scores = all_scores
        # scoresRecord[:, 0, :] = self.getInitAlphaWithBatchSize(batchSize).view(batchSize, self.label_size)
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]  ## represent the best current score from the start, is the best
        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, sentLength):
            ### scoresIdx: batch x from_label x to_label at current index.
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize, self.label_size,
                                                                                  self.label_size) + scores[:, wordIdx, :, :]
            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)  ## the best previous label idx to crrent labels
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1, self.label_size)).view(batchSize, self.label_size)

        lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(batchSize, self.label_size)  ##select position
        lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(sentLength - 1):
            lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens - distance2Last - 1, mask).view(batchSize, 1, 1).expand(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)

        # print("bestScores, decodeIdx:  ", bestScores,decodeIdx)
        return bestScores, decodeIdx

    def pair_decode(self, pair_scores: torch.Tensor, max_review_id: torch.Tensor, decodeIdx: torch.Tensor, wordSeqLengths: torch.Tensor, review_idx: torch.Tensor, reply_idx: torch.Tensor)  -> torch.Tensor :
        # batch_size = decodeIdx.size()[0]
        # max_review_size = max_review_id.max()
        # max_seq_len = decodeIdx.size()[1]
        # max_reply_size = max_seq_len - max_review_id.max().min

        # review_idx = torch.zeros((batch_size,max_review_size), dtype=torch.long)
        # reply_idx = torch.zeros((batch_size, max_reply_size), dtype=torch.long)

        # review_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        # reply_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        # for batch_idx in range(batch_size):
        #     # review_ids[batch_idx,:] = decodeIdx[batch_idx][:max_review_id[batch_idx]]
        #     i=0
        #     for idx in range(max_review_id[batch_idx]):
        #         # if decodeIdx[batch_idx][idx] in (2,3,4,5):
        #         review_idx[batch_idx,i]= idx
        #         i+=1
        #     i=0
        #     for idx in range(max_review_id[batch_idx]+1,max_review_size):
        #         reply_idx[batch_idx,i]= idx
        #         i+=1
        #
        # # review_index
        # review_idx = review_idx.to(self.device)
        # reply_idx = reply_idx.to(self.device)
        # print('feature_out.size()  ',feature_out.size(),review_idx, reply_idx)
        # print('review_idx.unsqueeze(2).expand(feature_out.size()): ',reply_idx.unsqueeze(2).expand(feature_out.size()))



        # lstm_review_rep = torch.gather(feature_out, 1, review_idx.unsqueeze(2).expand(feature_out.size()))
        # lstm_reply_rep = torch.gather(feature_out, 1, reply_idx.unsqueeze(2).expand(feature_out.size()))
        # batch_size, max_review, hidden_dim = lstm_review_rep.size()
        # max_reply = lstm_reply_rep.size()[1]
        #
        # # print(feature_out)
        # # print(lstm_reply_rep)
        # # print("max_review, max_reply:  ",max_review,max_reply)
        # lstm_review_rep = lstm_review_rep.unsqueeze(2).expand(batch_size, max_review, max_reply, hidden_dim)
        # lstm_reply_rep = lstm_reply_rep.unsqueeze(1).expand(batch_size, max_review, max_reply, hidden_dim)
        # lstm_pair_rep = torch.cat([lstm_review_rep, lstm_reply_rep], dim=-1)
        #
        # # batch_size, max_seq, hidden_dim = feature_out.size()
        # #
        # # lstm_review_rep = feature_out.unsqueeze(2).expand(batch_size, max_seq, max_seq, hidden_dim)
        # # lstm_reply_rep = feature_out.unsqueeze(1).expand(batch_size, max_seq, max_seq, hidden_dim)
        # # lstm_pair_rep = torch.cat([lstm_review_rep, lstm_reply_rep], dim=-1)
        #
        # # pair_scores = self.pair2score(lstm_pair_rep)
        #
        # x = self.pair2score_first(lstm_pair_rep)
        # y = F.relu(x)
        # # y = self.pair2score_second(y)
        # # y = F.relu(y)
        # pair_scores = self.pair2score_final(y)
        # pair_scores = F.log_softmax(pair_scores, dim=3)
        # # pair_scores = pair_scores.view(-1, 2)
        # # print(pair_scores.size())



        pairIdx = torch.argmax(pair_scores, dim=3).unsqueeze(3)

        # sigmoid = nn.Sigmoid()
        # pair_scores=sigmoid(pair_scores)
        # t=0.5
        # # print(pair_scores)
        # pairIdx = (pair_scores>t).float()
        # print("pairIdx:  ", pairIdx, pairIdx.size())
        return pairIdx



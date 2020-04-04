# 
# @author: Allan
#

import torch
import torch.nn as nn

from config import START, STOP, PAD, log_sum_exp_pytorch
from model.charbilstm import CharBiLSTM
from modelrr.bilstm_encoder import BiLSTMEncoder
from modelrr.linear_crf_inferencer import LinearCRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from config import ContextEmb
from typing import Tuple
from overrides import overrides


class NNCRF(nn.Module):

    def __init__(self, config, print_info: bool = True):
        super(NNCRF, self).__init__()
        self.device = config.device
        self.encoder = BiLSTMEncoder(config, print_info=print_info)
        self.inferencer = LinearCRF(config, print_info=print_info)

    @overrides
    def forward(self, sent_emb_tensor: torch.Tensor,
                    type_id_tensor: torch.Tensor,
                    sent_seq_lens: torch.Tensor,
                    batch_context_emb: torch.Tensor,
                    chars: torch.Tensor,
                    char_seq_lens: torch.Tensor,
                    tags: torch.Tensor,
                    review_index: torch.Tensor,
                    reply_index: torch.Tensor,
                    pairs: torch.Tensor,
                    pair_padding: torch.Tensor,
                        max_review_id: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param batch_context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param tags: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """
        # print("sents: ",sents)
        _,lstm_scores,pair_scores = self.encoder(sent_emb_tensor, type_id_tensor, sent_seq_lens, batch_context_emb, chars, char_seq_lens,tags,review_index, reply_index, pairs,pair_padding, max_review_id)
        # print("lstm_scores: ", lstm_scores)
        # lstm_scores = self.encoder(sent_emb_tensor, sent_seq_lens, chars, char_seq_lens)
        batch_size = sent_emb_tensor.size(0)
        sent_len = sent_emb_tensor.size(1)
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long).view(1, sent_len).expand(batch_size, sent_len).to(self.device)
        mask = torch.le(maskTemp, sent_seq_lens.view(batch_size, 1).expand(batch_size, sent_len)).to(self.device)
        unlabed_score, labeled_score, pair_loss =  self.inferencer(lstm_scores, pair_scores, sent_seq_lens, tags, mask, pairs,pair_padding)
        # print('unlabed_score:  ',unlabed_score.size(),unlabed_score)
        # print('labeled_score:  ',labeled_score.size(),labeled_score)
        print('loss:', unlabed_score - labeled_score, pair_loss)
        return (unlabed_score - labeled_score) + 0.1*pair_loss
        # return pair_loss

    def decode(self, batchInput: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        wordSeqTensor, typeTensor, wordSeqLengths, batch_context_emb, charSeqTensor, charSeqLengths, tagSeqTensor, review_index, reply_index, pairs, pair_padding, max_review_id = batchInput
        feature_out,features, pair_scores = self.encoder(wordSeqTensor, typeTensor, wordSeqLengths, batch_context_emb,charSeqTensor,charSeqLengths, tagSeqTensor, review_index, reply_index, pairs, pair_padding, max_review_id)
        bestScores, decodeIdx = self.inferencer.decode(features, wordSeqLengths)
        # print ('decodeIdx:  ', decodeIdx)
        pairIdx = self.inferencer.pair_decode(feature_out, max_review_id, decodeIdx, wordSeqLengths)


        # print(bestScores, decodeIdx)
        return bestScores, decodeIdx, pairIdx

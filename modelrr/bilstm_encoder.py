
import torch
import torch.nn as nn

from config import ContextEmb
from modelrr.charbilstm import CharBiLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from allennlp.nn.util import batched_index_select

from overrides import overrides

class BiLSTMEncoder(nn.Module):

    def __init__(self, config, print_info: bool = True):
        super(BiLSTMEncoder, self).__init__()

        self.label_size = config.label_size
        self.device = config.device
        self.use_char = config.use_char_rnn
        self.context_emb = config.context_emb

        self.label2idx = config.label2idx
        self.labels = config.idx2labels

        self.input_size = config.embedding_dim
        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size
        if self.use_char:
            self.char_feature = CharBiLSTM(config, print_info=print_info)
            self.input_size += config.charlstm_hidden_dim

        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.word_embedding), freeze=False).to(self.device)
        self.word_drop = nn.Dropout(config.dropout).to(self.device)

        self.type_embedding = nn.Embedding(3,20).to(self.device)

        if print_info:
            print("[Model Info] Input size to LSTM: {}".format(self.input_size))
            print("[Model Info] LSTM Hidden Size: {}".format(config.hidden_dim))

        self.lstm = nn.LSTM(self.input_size, config.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True).to(self.device)
        # print('lstm: ',self.lstm)

        self.drop_lstm = nn.Dropout(config.dropout).to(self.device)
        # print('drop_lstm: ',self.drop_lstm)

        final_hidden_dim = config.hidden_dim

        if print_info:
            print("[Model Info] Final Hidden Size: {}".format(final_hidden_dim))

        self.hidden2tag = nn.Linear(final_hidden_dim, self.label_size).to(self.device)

        self.pair2score = nn.Linear(final_hidden_dim * 2, 1).to(self.device)

    @overrides
    def forward(self, sent_emb_tensor: torch.Tensor,
                      type_id_tensor: torch.Tensor,
                      sent_seq_lens: torch.Tensor,
                      batch_context_emb: torch.Tensor,
                      char_inputs: torch.Tensor,
                      char_seq_lens: torch.Tensor,
                      tags: torch.Tensor,
                      review_index: torch.Tensor,
                      reply_index: torch.Tensor,
                        pairs: torch.Tensor,
                        pair_padding_tensor: torch.Tensor,
                        max_review_id: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_seq_tensor: (batch_size, sent_len)   NOTE: The word seq actually is already ordered before come here.
        :param word_seq_lens: (batch_size, 1)
        :param batch_context_emb: (batch_size, sent_len, context embedding) ELMo embedings
        :param char_inputs: (batch_size * sent_len * word_length)
        :param char_seq_lens: numpy (batch_size * sent_len , 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """

        # word_emb = self.word_embedding(word_seq_tensor)
        # if self.context_emb != ContextEmb.none:
        #     word_emb = torch.cat([word_emb, batch_context_emb.to(self.device)], 2)
        # if self.use_char:
        #     char_features = self.char_feature(char_inputs, char_seq_lens)
        #     word_emb = torch.cat([word_emb, char_features], 2)
        # print(type_id_tensor)
        sent_emb_tensor = sent_emb_tensor.to(self.device)
        type_emb = self.type_embedding(type_id_tensor)
        # print('type_id_tensor: ', type_id_tensor.size())

        # sent_rep = sent_emb_tensor
        sent_rep = torch.cat([sent_emb_tensor,type_emb],2)


        sent_rep = self.word_drop(sent_rep)

        # print("word rep length: ",sent_rep.shape)

        sorted_seq_len, permIdx = sent_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = sent_rep[permIdx]
        type_id = type_id_tensor[permIdx]
        tag_id = tags[permIdx]
        # print('type_id: ', type_id)

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
        lstm_out, _ = self.lstm(packed_words, None)
        # print('lstm_out1: ',lstm_out.size())
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        # print('lstm_out2: ', lstm_out.size())
        feature_out = self.drop_lstm(lstm_out)
        # print('feature_out: ',feature_out)

        feature_out = feature_out[recover_idx]

        # max_review_len = 0
        # max_reply_len =0
        # max_pair_len = 0
        # for lstm_idx, lstm_inst in enumerate(feature_out):
        #     # review_len = lstm_inst.count(1)
        #     # reply_len = lstm_inst.count(2)
        #     review_len = 0
        #     reply_len = 0
        #     for lstm_sent_idx, lstm_sent_rep in enumerate(lstm_inst):
        #         # print("first time:  ",  type_id[lstm_idx][lstm_sent_idx])
        #         if type_id[lstm_idx][lstm_sent_idx]==1 and tag_id[lstm_idx][lstm_sent_idx] in (2,3,4,5):
        #             review_len+=1
        #         if type_id[lstm_idx][lstm_sent_idx] == 2:
        #             reply_len+=1
        #     pair_len = review_len * reply_len
        #     # print('review_len: ',review_len)
        #     if  pair_len > max_pair_len:
        #         max_pair_len = pair_len
        #     if review_len>max_review_len:
        #         max_review_len = review_len
        #     if reply_len>max_reply_len:
        #         max_reply_len = reply_len
        #
        # print("max review and reply len", max_review_len,max_reply_len, max_pair_len)
        #
        # batch_size = feature_out.size()[0]
        # hidden_dim = feature_out.size()[-1]
        #
        # lstm_review_rep = torch.zeros((batch_size, max_review_len, hidden_dim), dtype=torch.float32)
        # lstm_reply_rep = torch.zeros((batch_size, max_reply_len, hidden_dim), dtype=torch.float32)
        # lstm_pair_rep = torch.zeros((batch_size, max_review_len, max_reply_len, hidden_dim * 2), dtype=torch.float32)
        #
        #
        # for lstm_idx, lstm_inst in enumerate(feature_out):
        #     review_idx = 0
        #     reply_idx = 0
        #     pair_idx = 0
        #     # print(type_id[lstm_idx])
        #     # print(lstm_inst.size(),lstm_inst)
        #     for lstm_review_idx, lstm_review in enumerate(lstm_inst):
        #         # print(type_id[lstm_idx][lstm_review_idx])
        #         if type_id[lstm_idx][lstm_review_idx] == 2:
        #             # print('test reply')
        #             lstm_reply_rep[lstm_idx, reply_idx, :] = lstm_review
        #             reply_idx+=1
        #         if type_id[lstm_idx][lstm_review_idx]==1 and tag_id[lstm_idx][lstm_review_idx] in (2,3,4,5):
        #             lstm_review_rep[lstm_idx,review_idx,:] = lstm_review
        #             review_idx += 1
        #             # print('test2: ', type_id[lstm_idx])
        #             reply2_idx=0
        #             for lstm_reply_idx, lstm_reply in enumerate(lstm_inst):
        #
        #                 if type_id[lstm_idx][lstm_reply_idx] == 2:
        #                     # print("lstm_idx,review_idx, reply_idx, pair_idx: ", lstm_idx,review_idx, reply_idx, pair_idx)
        #                     # print("lstm_review,lstm_reply  ", lstm_review.size(),lstm_reply.size(), lstm_review,lstm_reply)
        #                     # print("torch.cat([lstm_review,lstm_reply],2):  ", torch.cat([lstm_review,lstm_reply],2).size(), torch.cat([lstm_review,lstm_reply],2))
        #                     # print(review_idx,reply_idx,reply2_idx)
        #                     # print(lstm_idx)
        #                     lstm_pair_rep[lstm_idx,review_idx-1,reply2_idx,:] = torch.cat([lstm_review,lstm_reply],0)
        #                     pair_idx+=1
        #                     reply2_idx+=1

        ## pip install allennlp

        ## type id: batch_size x sent_len = 6
        ## type_id[0]= (1, 1, 1, 2 ,2 , 0)
        # review_index = (batch_size x max_review_len)
        ## review_index[0] = (0, 1, 2)
        # reply_index = (batch_size x max_reply_len)
        # reply_index[0] = (3,4)


        # print('feature_out:   ', feature_out.size())
        # print('review_index: ',review_index.size())
        lstm_review_rep = torch.gather(feature_out, 1, review_index.unsqueeze(2).expand(feature_out.size()))
        lstm_reply_rep = torch.gather(feature_out, 1, reply_index.unsqueeze(2).expand(feature_out.size()))
        batch_size, max_review, hidden_dim = lstm_review_rep.size()
        max_reply = lstm_reply_rep.size()[1]
        # print("batch_size, max_review, hidden_dim:  ", batch_size, max_review, hidden_dim)

        # lstm_review_rep = batched_index_select(feature_out, review_index)
        # lstm_reply_rep = batched_index_select(feature_out, reply_index)
        lstm_review_rep = lstm_review_rep.unsqueeze(2).expand(batch_size,max_review,max_reply,hidden_dim)
        lstm_reply_rep = lstm_reply_rep.unsqueeze(1).expand(batch_size,max_review,max_reply,hidden_dim)
        lstm_pair_rep = torch.cat([lstm_review_rep, lstm_reply_rep], dim=-1)



        # print(lstm_pair_rep.size(),pairs.size())




        # lstm_pair_rep = torch.zeros(lstm_review_rep.size()[0],lstm_review_rep.size()[1],lstm_reply_rep.size()[1],lstm_review_rep.size()[2]*2)
        #
        # for idx in range(lstm_review_rep.size()[0]):
        #     for rvw_idx in range(lstm_review_rep[idx].size()[0]):
        #         for rpl_idx in range(lstm_reply_rep[idx].size()[0]):
        #             lstm_pair_rep[idx,rvw_idx,rpl_idx,:]=torch.cat([lstm_review_rep[idx][rvw_idx],lstm_reply_rep[idx][rpl_idx]],0)

        #
        # print('review:  ', lstm_review_rep.size(),lstm_review_rep)
        # print('reply:  ', lstm_reply_rep.size(),lstm_reply_rep)
        # print('pair:  ', lstm_pair_rep.size(), lstm_pair_rep)


        score = self.pair2score(lstm_pair_rep)
        # print("score_size:  ",score.size())

        score = score * pair_padding_tensor.unsqueeze(3)

        outputs = self.hidden2tag(feature_out)
        # print('outputs: ',outputs.size())

        return feature_out,outputs,score



import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, target_size, drop_out=0.1):
        super(BiLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=2,
            dropout=drop_out,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_size * 2, target_size)
        self.crf = CRF(target_size, batch_first=True)

    # only BiLSTM and linear
    def forward(self, inputs_ids):
        embeddings = self.embedding(inputs_ids)
        sequence_output, _ = self.bilstm(embeddings)
        tag_scores = self.classifier(sequence_output)
        return tag_scores

    # get loss
    def forward_with_crf(self, input_ids, input_tags, input_mask):
        tag_scores = self.forward(input_ids)
        loss = self.crf(tag_scores, input_tags, input_mask) * (-1) # 相当于计算一个交叉信息熵
        return tag_scores, loss

    # 相当于维特比解码，做预测，丢给他一个句子，让他返回一个label对于的id
    def decode(self, input_word):
        out = self.forward(input_word)
        predicted_index = self.crf.decode(out)
        return predicted_index


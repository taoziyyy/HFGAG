import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.Bi_lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)  # 使用双向Lstm
        self.linear = nn.Linear(2 * hidden_size, out_size)

    def forward(self, sentence_tensor, lengths):
        emb = self.embedding(sentence_tensor)
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.Bi_lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True, )
        scores = self.linear(output)

        return scores

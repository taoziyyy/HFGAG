# -*- coding: utf-8 -*-
"""
@Time ： 2023/7/4 下午3:55
@Auth ： zhangtao
@File ：GPT_2_classification.py
@IDE ：PyCharm
@Desc: 
"""
import torch
import torch.nn as nn
from transformers import GPT2Model


class ClassificationModel(nn.Module):
    def __init__(self, num_classes, hidden_size, num_layers, bidirectional=True):
        super(ClassificationModel, self).__init__()

        # 加载预训练的GPT-2模型
        self.gpt2 = GPT2Model.from_pretrained("gpt2")

        # 添加BILSTM层
        self.lstm = nn.LSTM(self.gpt2.config.hidden_size, hidden_size, num_layers, bidirectional=bidirectional,
                            batch_first=True)

        # 添加注意力机制
        self.attention = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)

        # 添加全连接层进行文本分类
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_classes)

    def forward(self, input_ids):
        # 获取GPT-2模型的输出
        gpt2_output = self.gpt2(input_ids)[0]

        # 使用BILSTM层进行特征提取
        lstm_output, _ = self.lstm(gpt2_output)

        # 使用注意力机制获取加权的特征表示
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        weighted_sum = torch.sum(attention_weights * lstm_output, dim=1)

        # 将加权的特征表示应用于全连接层
        output = self.fc(weighted_sum)
        # output = self.fc(gpt2_output[:, 0, :])
        return output

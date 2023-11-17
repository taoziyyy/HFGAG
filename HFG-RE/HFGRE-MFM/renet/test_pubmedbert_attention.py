import os

import numpy as np
import time
import datetime
import random
import sys
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, AdamW, \
    get_linear_schedule_with_warmup, BertModel
import torch.nn as nn
from raw import load_documents
from utils.sequence_utils import *
import argparse
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import KFold
# from transformers import AutoModelForSequenceClassification
# from transformers import get_linear_schedule_with_warmup
# from transformers import AutoConfig
# from transformers import AdamW
from sklearn.metrics import f1_score, recall_score, precision_score


class Config(object):
    """Holds Model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
    window_sizes = [2, 3, 4, 5]
    filter_size = 25
    word_embed_size = 200
    num_word = 82949
    feature_embed_size = 4
    total_feature_size = word_embed_size + feature_embed_size
    sentence_size = 52
    token_size = 175
    batch_size = 128
    label_size = 1
    hidden_size = 100
    max_epochs = 50
    early_stopping = 2
    dropout = 0.1
    lr = 0.001
    l2 = 0.0001


class GlobalAttentionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GlobalAttentionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
        self.attention = nn.Linear(768, 1)  # 全局注意力层
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # BERT模型输出的所有隐层状态
        attention_weights = self.attention(hidden_states).squeeze(dim=-1)  # 计算全局注意力权重
        attention_weights = torch.softmax(attention_weights, dim=1)  # 注意力权重归一化
        pooled_output = torch.matmul(attention_weights.unsqueeze(dim=1), hidden_states).squeeze(dim=1)  # 使用注意力权重加权汇聚特征
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def read_pos_labels(label_path):
    pos_labels = pd.read_csv(label_path)
    pos_labels.pmid = pos_labels.pmid.astype(str)
    pos_labels.geneId = pos_labels.geneId.astype(str)
    return pos_labels


def get_numpy_labels(gdas, pos_labels):
    label_df = pos_labels.drop_duplicates()
    gdas = pd.merge(gdas, label_df, on=['pmid', "geneId", "follicleId"], how="left")
    s = pd.Series(gdas.label)
    s.fillna(0, inplace=True)
    y = s.values
    return y, gdas


def write_2list_to_file(lst, output_path):
    with open(output_path, 'w') as f:
        for sublist in lst:
            for subsublist in sublist:
                f.write(' '.join(subsublist) + ' ')
            f.write('\n')


def write_1list_to_file(lst, output_path):
    with open(output_path, 'w') as f:
        for sublist in lst:
            f.write(' '.join(sublist) + '\n')


def read_file_to_list(file_path):
    result = []
    with open(file_path, 'r') as f:
        for line in f:
            # Remove any leading/trailing white space and new lines
            line = line.strip()
            # Append the cleaned line to the result1 list
            result.append(line)
    return result


def Run(args):
    """Test NER Model implementation.

    When debugging, set max_epochs in the Config object to 1,
    so you can rapidly iterate.
    """

    # 初始化日志记录器
    logger = logging.getLogger(__name__)
    if args.model_name == "dmis-lab/biobert-v1.1":
        filename = f'../result/log/train-biobert.log'
    elif args.model_name == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":
        filename = f'../result/log/train-pubmedbert-attention.log'
    else:
        filename = f'../result/log/train' + args.model_name + '-{str(datetime.date.today())}.log'

    logger.setLevel(logging.INFO)
    logging.basicConfig(filename=filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    print("Loading data...")
    logging.info("Loading data...")

    text_path = args.ab_fn
    sentence_path = args.stc_fn
    ner_path = args.ner_fn
    label_path = args.label_fn
    out_fn = args.output_dir

    gdas, x_word_seq, x_feature = load_documents(text_path, sentence_path, ner_path)
    write_2list_to_file(x_word_seq, "../test/seq.txt")
    new_seq = read_file_to_list('../test/seq.txt')
    new_sequences = Generate_sequences_data(new_seq, gdas)
    pos_labels = read_pos_labels(label_path)
    y, total_y = get_numpy_labels(gdas, pos_labels)
    final_sequences = [' '.join([str(item) for item in row]) for row in new_sequences]
    dataset = pd.DataFrame({'Data': final_sequences, 'Label': y})
    texts = list(dataset.Data)
    label = np.array(dataset.Label.astype('int'))
    number_label = dataset.Label.astype('category').cat.categories
    index_list = [index for index, value in enumerate(texts)]

    # output_path = args.output_dir
    # X_train, y_train, X_val, y_val = split_data(x_word_seq, gdas, y, 1)

    print("Finished loading data.")
    logging.info("Finished loading data.")
    print("Begin predicting...")
    logging.info("Begin predicting...")


    # Tokenize Data
    model_name = args.model_name
    # model_name = "bert-base-cased"
    # model_name = "dmis-lab/biobert-v1.1"
    # model_name = "distilbert-base-uncased"
    # model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    MAX_INPUT_LENGTH = 512
    batch_size = 5
    pre_result = []
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    inputs = tokenizer(texts, padding=True, max_length=MAX_INPUT_LENGTH, return_tensors="pt")
    inputs["input_ids"] = inputs["input_ids"][:, :MAX_INPUT_LENGTH]  # Truncate input_ids if necessary
    inputs["attention_mask"] = inputs["attention_mask"][:,
                               :MAX_INPUT_LENGTH]  # Truncate attention_mask if necessary
    # inputs["token_type_ids"] = inputs["token_type_ids"][:, :MAX_INPUT_LENGTH]
    labels = torch.tensor(label, dtype=torch.long)

    # Train 划分每个数据集中的训练集和验证集
    pre_set = TensorDataset(torch.index_select(inputs.input_ids, 0, torch.tensor(index_list)),
                               torch.index_select(inputs.attention_mask, 0, torch.tensor(index_list)),
                               labels[index_list])
    # train DataLoader
    pre_loader = DataLoader(
        pre_set,
        sampler=RandomSampler(pre_set),
        batch_size=batch_size
    )

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    print("Loading Model...")
    logging.info("Loading Model...")

    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda:1")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(2))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    model = GlobalAttentionClassifier(num_classes=2)

    # 加载模型的参数和权重
    model.load_state_dict(torch.load("../model/save_model/pubmedbert-attention4model.pth"))
    model.cuda(1)
    # 在测试数据上进行预测
    model.eval()
    # 假设你的测试数据已经准备好 test_loader
    for batch in pre_loader:

        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device)}

        outputs = model(**inputs)

        # Move logits and labels to CPU
        logits = outputs.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        pre_result.extend(pred_flat)

    print(pre_result)
    gdas['pre_label'] = pre_result
    # 将 DataFrame 保存为 CSV 文件
    gdas.to_csv("../result/pre_result/gfas.csv", index=False)  # 设置 index=False 表示不保存行索引

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RENET: A Deep Learning Approach for Extracting Gene-Disease Associations from Literature")

    parser.add_argument('--ab_fn', type=str, default=None,
                        help="file of abstracts")

    parser.add_argument('--stc_fn', type=str, default=None,
                        help="file of splitted sentences")

    parser.add_argument('--ner_fn', type=str, default=None,
                        help="annotation of disease/gene entities")

    parser.add_argument('--label_fn', type=str, default=None,
                        help="true G-D-A associations by DisGeNet")

    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output Model")

    parser.add_argument('--model_name', type=str, default=None,
                        help="model_name")

    parser.add_argument('--batch_size', type=int, default=None,
                        help="batch_size")

    parser.add_argument('--epochs', type=int, default=None,
                        help="epochs")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)

# -*- coding: utf-8 -*-
"""
@Time ： 2023/7/21 下午8:51
@Auth ： zhangtao
@File ：train_bert-crf_ner.py
@IDE ：PyCharm
@Desc: 
"""
import pandas as pd
import numpy as np
from torchcrf import CRF
from tqdm import tqdm, trange
import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
import transformers
import torch
from transformers import BertModel, BertPreTrainedModel
import torchcrf
from seqeval.metrics import f1_score, accuracy_score, recall_score, precision_score
from transformers import BertForTokenClassification
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel

print(torch.__version__)
print(transformers.__version__)

def vet_frases(dataframe):
  sentences = []
  sentences_aux = []
  labels = []
  labels_aux = []
  for word, label in zip(dataframe.word.values, dataframe.label.values):
    if (word == ''):
        continue
    sentences_aux.append(word)
    labels_aux.append(label)
    if (word == '.'):
      sentences.append(sentences_aux)
      labels.append(labels_aux)

      sentences_aux = []
      labels_aux = []
  return sentences, labels

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []
    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,labels=None):
        outputs =self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores

MAX_LEN = 128
bs = 5
epochs = 20
max_grad_norm = 1.0

# 打开文件
with open("2023-data/train.txt", "r") as file:
    # 读取文件内容
    lines = file.readlines()
# 分割每行并创建 DataFrame
train = [line.strip().split("   ") for line in lines]
train = pd.DataFrame(train, columns=["word", "label"]).fillna('')

with open("2023-data/dev.txt", "r") as file:
    # 读取文件内容
    lines = file.readlines()
# 分割每行并创建 DataFrame
devel = [line.strip().split("   ") for line in lines]
devel = pd.DataFrame(devel, columns=["word", "label"]).fillna('')

with open("2023-data/test.txt", "r") as file:
    # 读取文件内容
    lines = file.readlines()
# 分割每行并创建 DataFrame
test = [line.strip().split("   ") for line in lines]
test = pd.DataFrame(test, columns=["word", "label"]).fillna('')
print("train", train)

# creat dataset
train_sentences, train_labels = vet_frases(train)
print(train_sentences[0])
print(train_labels[0])
test_sentences, test_labels = vet_frases(test)
print(test_sentences[0])
print(test_labels[0])
devel_sentences, devel_labels = vet_frases(devel)
print(devel_sentences[0])
print(devel_labels[0])

# creat label
tag_values = list(set(train["label"].values))
tag_values.append("PAD")
tag_values = [x for x in tag_values if x]
print(sorted(tag_values))
tag2idx = {t: i for i, t in enumerate(tag_values)}
print(len(tag_values))

# set GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


train_tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(train_sentences, train_labels)
]
test_tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(test_sentences, test_labels)
]
devel_tokenized_texts_and_labels = [
    tokenize_and_preserve_labels(sent, labs)
    for sent, labs in zip(devel_sentences, devel_labels)
]
print(train_tokenized_texts_and_labels[0])

train_tokenized_texts = [token_label_pair[0] for token_label_pair in train_tokenized_texts_and_labels]
train_labels = [token_label_pair[1] for token_label_pair in train_tokenized_texts_and_labels]
test_tokenized_texts = [token_label_pair[0] for token_label_pair in test_tokenized_texts_and_labels]
test_labels = [token_label_pair[1] for token_label_pair in test_tokenized_texts_and_labels]
devel_tokenized_texts = [token_label_pair[0] for token_label_pair in devel_tokenized_texts_and_labels]
devel_labels = [token_label_pair[1] for token_label_pair in devel_tokenized_texts_and_labels]
print(train_tokenized_texts[0])

train_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in train_tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")
test_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in test_tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")
devel_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in devel_tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", value=0.0,
                          truncating="post", padding="post")
train_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in train_labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")
test_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in test_labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")
devel_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in devel_labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")
train_attention_masks = [[float(i != 0.0) for i in ii] for ii in train_input_ids]
test_attention_masks = [[float(i != 0.0) for i in ii] for ii in test_input_ids]
devel_attention_masks = [[float(i != 0.0) for i in ii] for ii in devel_input_ids]
tr_inputs, tr_tags, tr_masks = shuffle(train_input_ids, train_tags, train_attention_masks, random_state=2020)
val_inputs, val_tags, val_masks = shuffle(devel_input_ids, devel_tags, devel_attention_masks, random_state=2020)
test_inputs, test_tags, test_masks = shuffle(test_input_ids, test_tags, test_attention_masks, random_state=2020)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
test_inputs = torch.tensor(test_inputs)
test_tags = torch.tensor(test_tags)
test_masks = torch.tensor(test_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

test_data = TensorDataset(test_inputs, test_masks, test_tags)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=bs)

# creat model
config = BertConfig.from_pretrained("bert-base-cased", num_labels=len(tag2idx), output_attentions=False, output_hidden_states=False)
model = BertCrfForNer.from_pretrained(
    "bert-base-cased",
    config=config
)

model.cuda()
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []
validation_accuracy, validation_f1 = [], []
validation_recall, validation_precision = [], []
for _ in trange(epochs, desc="Epoch"):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set.

    # Put the model into training mode.
    model.train()
    # Reset the total loss for this epoch.
    total_loss = 0

    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        # b_input_ids, b_input_mask, b_labels = batch
        inputs = {"input_ids": batch[0], "attention_mask": batch[1].to(torch.bool), "token_type_ids": None, "labels": batch[2]}
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(**inputs)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print()
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)


    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    # Put the model into evaluation mode
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1].to(torch.bool), "token_type_ids": None, "labels": batch[2]}

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(**inputs)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = batch[2].to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    print("Validation loss: {}".format(eval_loss))
    validation_loss_values.append(eval_loss)
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation Recall: {}".format(recall_score(pred_tags, valid_tags)))
    print("Validation Precision: {}".format(precision_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()
    validation_accuracy.append(accuracy_score(pred_tags, valid_tags))
    validation_f1.append(f1_score(pred_tags, valid_tags))
    validation_recall.append(recall_score(pred_tags, valid_tags))
    validation_precision.append(precision_score(pred_tags, valid_tags))
result_data = {'train_loss': loss_values, 'val_loss': validation_loss_values, 'val_f1': validation_f1, 'val_pre': validation_precision, 'val_recall': validation_recall}
# 保存数据到文件
with open('save_model/bert-crf_metrics.pkl', 'wb') as file:
    pickle.dump(result_data, file)
model.save_pretrained("save_model/bert-crf_model_ner")

# # 从文件中加载数据
# with open('metrics.pkl', 'rb') as file:
#     loaded_data = pickle.load(file)
#
# # 获取加载后的数据
# f1_list = loaded_data['f1']
# loss_list = loaded_data['loss']
#
# # 打印加载后的数据
# print(f1_list)
# print(loss_list)
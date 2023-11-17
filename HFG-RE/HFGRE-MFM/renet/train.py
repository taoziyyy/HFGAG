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
    get_linear_schedule_with_warmup

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


def split_data(word_seq, gdas, y, train_val_split = 0.7):

    train_size = int(len(y) * train_val_split)

    train_word_seq = word_seq[:train_size]
    train_gdas = gdas[:train_size]
    X_train = train_word_seq, train_gdas
    y_train = y[:train_size]

    val_word_seq = word_seq[train_size:]
    val_gdas = gdas[train_size:]
    X_val = val_word_seq, val_gdas
    y_val = y[train_size:]

    return X_train, y_train, X_val, y_val

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
    data_set = 1
    training_f1_stats = []
    validation_f1_stats = []
    training_accuracy_stats = []
    validation_accuracy_stats = []
    training_recall_stats = []
    validation_recall_stats = []
    training_precision_stats = []
    validation_precision_stats = []
    training_time_stats = []
    validation_time_stats = []
    # "bert-base-cased"
    # "distilbert-base-uncased"
    # "dmis-lab/biobert-v1.1"

    # 初始化日志记录器
    logger = logging.getLogger(__name__)
    if args.model_name == "dmis-lab/biobert-v1.1":
        filename = f'../result/log/train-biobert.log'
    elif args.model_name == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":
        filename = f'../result/log/train-pubmedbert.log'
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

    training_sizes = [200, 500, 1000, 1500, 2000, 2535]
    Xs = []
    Ys = []
    for s in training_sizes:
        Xs.append(texts[0:s])
        Ys.append(label[0:s])

    # output_path = args.output_dir
    # X_train, y_train, X_val, y_val = split_data(x_word_seq, gdas, y, 1)

    print("Finished loading data.")
    logging.info("Finished loading data.")
    print("Begin training...")
    logging.info("Begin training...")
    # config = Config()

    # 遍历不同尺寸的6个数据集
    for xx, yy in zip(Xs, Ys):
        kf = KFold(n_splits=5)
        batch_size = args.batch_size
        train_dataset_index = []
        test_dataset_index = []
        for train_index, test_index in kf.split(xx):
            train_dataset_index.append(train_index)
            test_dataset_index.append(test_index)

        # Tokenize Data
        model_name = args.model_name
        # model_name = "bert-base-cased"
        # model_name = "dmis-lab/biobert-v1.1"
        # model_name = "distilbert-base-uncased"
        # model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        MAX_INPUT_LENGTH = 512
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        inputs = tokenizer(xx, padding=True, max_length=MAX_INPUT_LENGTH, return_tensors="pt")
        inputs["input_ids"] = inputs["input_ids"][:, :MAX_INPUT_LENGTH]  # Truncate input_ids if necessary
        inputs["attention_mask"] = inputs["attention_mask"][:, :MAX_INPUT_LENGTH]  # Truncate attention_mask if necessary
        # inputs["token_type_ids"] = inputs["token_type_ids"][:, :MAX_INPUT_LENGTH]
        labels = torch.tensor(yy, dtype=torch.long)

        # Train 划分每个数据集中的训练集和验证集
        train_set1 = TensorDataset(torch.index_select(inputs.input_ids, 0, torch.tensor(train_dataset_index[0])),
                                   torch.index_select(inputs.attention_mask, 0, torch.tensor(train_dataset_index[0])),
                                   labels[train_dataset_index[0]])

        train_set2 = TensorDataset(torch.index_select(inputs.input_ids, 0, torch.tensor(train_dataset_index[1])),
                                   torch.index_select(inputs.attention_mask, 0, torch.tensor(train_dataset_index[1])),
                                   labels[train_dataset_index[1]])

        train_set3 = TensorDataset(torch.index_select(inputs.input_ids, 0, torch.tensor(train_dataset_index[2])),
                                   torch.index_select(inputs.attention_mask, 0, torch.tensor(train_dataset_index[2])),
                                   labels[train_dataset_index[2]])

        train_set4 = TensorDataset(torch.index_select(inputs.input_ids, 0, torch.tensor(train_dataset_index[3])),
                                   torch.index_select(inputs.attention_mask, 0, torch.tensor(train_dataset_index[3])),
                                   labels[train_dataset_index[3]])

        train_set5 = TensorDataset(torch.index_select(inputs.input_ids, 0, torch.tensor(train_dataset_index[4])),
                                   torch.index_select(inputs.attention_mask, 0, torch.tensor(train_dataset_index[4])),
                                   labels[train_dataset_index[4]])
        # test
        test_set1 = TensorDataset(torch.index_select(inputs.input_ids, 0, torch.tensor(test_dataset_index[0])),
                                  torch.index_select(inputs.attention_mask, 0, torch.tensor(test_dataset_index[0])),
                                  labels[test_dataset_index[0]])

        test_set2 = TensorDataset(torch.index_select(inputs.input_ids, 0, torch.tensor(test_dataset_index[1])),
                                  torch.index_select(inputs.attention_mask, 0, torch.tensor(test_dataset_index[1])),
                                  labels[test_dataset_index[1]])

        test_set3 = TensorDataset(torch.index_select(inputs.input_ids, 0, torch.tensor(test_dataset_index[2])),
                                  torch.index_select(inputs.attention_mask, 0, torch.tensor(test_dataset_index[2])),
                                  labels[test_dataset_index[2]])

        test_set4 = TensorDataset(torch.index_select(inputs.input_ids, 0, torch.tensor(test_dataset_index[3])),
                                  torch.index_select(inputs.attention_mask, 0, torch.tensor(test_dataset_index[3])),
                                  labels[test_dataset_index[3]])

        test_set5 = TensorDataset(torch.index_select(inputs.input_ids, 0, torch.tensor(test_dataset_index[4])),
                                  torch.index_select(inputs.attention_mask, 0, torch.tensor(test_dataset_index[4])),
                                  labels[test_dataset_index[4]])

        # train DataLoader
        train_loader1 = DataLoader(
            train_set1,
            sampler=RandomSampler(train_set1),
            batch_size=batch_size
        )

        train_loader2 = DataLoader(
            train_set2,
            sampler=RandomSampler(train_set2),
            batch_size=batch_size
        )

        train_loader3 = DataLoader(
            train_set3,
            sampler=RandomSampler(train_set3),
            batch_size=batch_size
        )

        train_loader4 = DataLoader(
            train_set4,
            sampler=RandomSampler(train_set4),
            batch_size=batch_size
        )

        train_loader5 = DataLoader(
            train_set5,
            sampler=RandomSampler(train_set5),
            batch_size=batch_size
        )
        # test DataLoader
        test_loader1 = DataLoader(
            test_set1,
            sampler=RandomSampler(test_set1),
            batch_size=batch_size
        )

        test_loader2 = DataLoader(
            test_set2,
            sampler=RandomSampler(test_set2),
            batch_size=batch_size
        )

        test_loader3 = DataLoader(
            test_set3,
            sampler=RandomSampler(test_set3),
            batch_size=batch_size
        )

        test_loader4 = DataLoader(
            test_set4,
            sampler=RandomSampler(test_set4),
            batch_size=batch_size
        )

        test_loader5 = DataLoader(
            test_set5,
            sampler=RandomSampler(test_set5),
            batch_size=batch_size
        )

        train_set = [train_loader1, train_loader2, train_loader3, train_loader4, train_loader5]
        test_set = [test_loader1, test_loader2, test_loader3, test_loader4, test_loader5]

        iter = 0  # counter

        total_t0 = time.time()
        training_row_f1 = []
        validation_row_f1 = []
        training_row_accuracy = []
        validation_row_accuracy = []
        training_row_recall = []
        validation_row_recall = []
        training_row_precision = []
        validation_row_precision = []
        training_row_time = []
        validation_row_time = []

        # 分别将一个数据集的five对训练、验证集进行训练
        for train, test in zip(train_set, test_set):
            iter += 1
            # Download the pre-trained Model
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = 2
            config.output_attentions = True
            config.return_dict = False
            config.finetuning_task = "SST-2"

            # Number of training epochs. The BERT authors recommend between 2 and 4.
            epochs = args.epochs
            total_steps = len(test) * epochs

            def flat_accuracy(preds, labels):
                pred_flat = np.argmax(preds, axis=1).flatten()
                labels_flat = labels.flatten()
                return np.sum(pred_flat == labels_flat) / len(labels_flat)

            def flat_f1_score(preds, labels):
                pred_flat = np.argmax(preds, axis=1).flatten()
                labels_flat = labels.flatten()
                return f1_score(labels_flat, pred_flat, average='weighted')

            def flat_recall_score(preds, labels):
                pred_flat = np.argmax(preds, axis=1).flatten()
                labels_flat = labels.flatten()
                return recall_score(labels_flat, pred_flat, average='weighted', zero_division=1)

            def flat_precision_score(preds, labels):
                pred_flat = np.argmax(preds, axis=1).flatten()
                labels_flat = labels.flatten()
                return precision_score(labels_flat, pred_flat, average='weighted', zero_division=1)
            def format_time(elapsed):
                '''
                Takes a time in seconds and returns a string hh:mm:ss
                '''
                # Round to the nearest second.
                elapsed_rounded = int(round((elapsed)))

                # Format as hh:mm:ss
                return str(datetime.timedelta(seconds=elapsed_rounded))

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


            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            # Tell pytorch to run this Model on the GPU.
            model.cuda(1)
            torch.cuda.empty_cache()

            optimizer = AdamW(model.parameters(),
                              lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                              eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                              )

            # Create the learning rate scheduler.
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,  # Default value in run_glue.py
                                                        num_training_steps=total_steps)


            epoch_training_row_f1 = []
            epoch_validation_row_f1 = []
            epoch_training_row_accuracy = []
            epoch_validation_row_accuracy = []
            epoch_training_row_recall = []
            epoch_validation_row_recall = []
            epoch_training_row_precision = []
            epoch_validation_row_precision = []
            epoch_training_row_time = []
            epoch_validation_row_time = []

            # # 指定保存checkpoint的文件名和路径
            # checkpoint_path = '../model/checkpoint/'+ model_name + str(iter) + 'checkpoint.pth'

            # 指定保存模型的文件路径
            if model_name == "dmis-lab/biobert-v1.1":
                model_path = '../model/save_model/biobert-model.pth'
            elif model_name == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":
                model_path = '../model/save_model/pubmedbert-model.pth'
            else:
                model_path = '../model/save_model/' + model_name + str(iter) + 'model.pth'


            # 加载之前的checkpoint（如果有的话）
            start_epoch = 0
            # if os.path.exists(checkpoint_path):
            #     print("Checkpoint file exists.")
            #     checkpoint = torch.load(checkpoint_path)
            #     model.load_state_dict(checkpoint['model_state_dict'])
            #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #     start_epoch = checkpoint['epoch']
            #     loss = checkpoint['loss']
            #     print("Loaded checkpoint from '{}' (epoch {})".format(checkpoint_path, start_epoch))
            # else:
            #     print("Checkpoint file does not exist.")


            # For each epoch...
            for epoch_i in range(0, epochs):

                print("")
                logging.info('')
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
                logging.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
                print('Training...')
                logging.info('Training...')

                # Measure training time
                t0 = time.time()

                # Reset the total loss for this epoch.
                total_train_loss = 0
                total_train_accuracy = 0
                total_train_f1 = 0
                total_train_recall = 0
                total_train_precision = 0
                model.train()

                # For each batch of training data...
                for step, batch in enumerate(train):

                    # Progress update every 40 batches.
                    if step % 40 == 0 and not step == 0:
                        # Calculate elapsed time in minutes.
                        elapsed = format_time(time.time() - t0)

                        # Report progress.
                        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train), elapsed))
                        logging.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train), elapsed))
                    # Unpack elements in DataLoader and copy each tensors to the GPU
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)

                    # print('b_input_ids:', b_input_ids.shape)
                    # print('b_input_mask:', b_input_mask.shape)
                    # print('b_labels:', b_labels.shape)
                    # clear any previously calculated gradients


                    loss, logits, attentions = model(input_ids=b_input_ids,
                                                     attention_mask=b_input_mask,
                                                     labels=b_labels)
                    total_train_loss += loss.item()
                    model.zero_grad()
                    # Perform a backward pass to calculate the gradients.
                    loss.backward()

                    # Clip the norm of the gradients to 1.0.
                    # This is to help prevent the "exploding gradients" problem.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Update parameters and take a step using the computed gradient.
                    optimizer.step()

                    # Update the learning rate.
                    scheduler.step()

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()

                    # Calculate the accuracy for this batch of train sentences, and
                    # accumulate it over all batches.
                    total_train_accuracy += flat_accuracy(logits, label_ids)
                    total_train_f1 += flat_f1_score(logits, label_ids)
                    total_train_recall += flat_recall_score(logits, label_ids)
                    total_train_precision += flat_precision_score(logits, label_ids)

                # Report the final accuracy for this training run.
                avg_train_accuracy = total_train_accuracy / len(train)
                # Report the final f1_score for this training run.
                avg_train_f1 = total_train_f1 / len(train)
                avg_train_recall = total_train_recall / len(train)
                avg_train_precision = total_train_precision / len(train)
                # Calculate the average loss over all of the batches.
                avg_train_loss = total_train_loss / len(train)
                # Measure how long this epoch took.
                training_time = format_time(time.time() - t0)

                # count
                print(" Dataset: {0:.2f}  Cross Validation Round(train,test): {1:.2f}".format(data_set, iter))
                print("  Average training loss: {0:.2f}".format(avg_train_loss))
                print("  Training Accuracy: {0:.2f}".format(avg_train_accuracy))
                print("  Training recall: {0:.2f}".format(avg_train_recall))
                print("  Training precision: {0:.2f}".format(avg_train_precision))
                print("  Training F1-Score: {0:.2f}".format(avg_train_f1))
                print("  Training epoch took: {:}".format(training_time))
                logging.info(" Dataset: {0:.2f}  Cross Validation Round(train,test): {1:.2f}".format(data_set, iter))
                logging.info("  Average training loss: {0:.2f}".format(avg_train_loss))
                logging.info("  Training Accuracy: {0:.2f}".format(avg_train_accuracy))
                logging.info("  Training recall: {0:.2f}".format(avg_train_recall))
                logging.info("  Training precision: {0:.2f}".format(avg_train_precision))
                logging.info("  Training F1-Score: {0:.2f}".format(avg_train_f1))
                logging.info("  Training epoch took: {:}".format(training_time))

                epoch_training_row_accuracy.append(avg_train_accuracy)
                epoch_training_row_recall.append(avg_train_recall)
                epoch_training_row_precision.append(avg_train_precision)
                epoch_training_row_f1.append(avg_train_f1)
                epoch_training_row_time.append(training_time)

                ### Validation
                print("")
                print("Running Validation...")
                logging.info("Running Validation...")

                t0 = time.time()

                # Put the Model in evaluation mode
                model.eval()

                # Tracking variables
                total_eval_accuracy = 0
                total_eval_loss = 0
                total_eval_f1 = 0
                total_eval_recall = 0
                total_eval_precision = 0
                nb_eval_steps = 0

                # Evaluate data for one epoch
                for batch in test:
                    b_input_ids = batch[0].to(device)
                    b_input_mask = batch[1].to(device)
                    b_labels = batch[2].to(device)

                    # Tell pytorch not to bother with constructing the compute graph during
                    with torch.no_grad():
                        (loss, logits, attentions) = model(input_ids=b_input_ids,
                                                           attention_mask=b_input_mask,
                                                           labels=b_labels)

                    # Accumulate the validation loss.
                    total_eval_loss += loss.item()

                    # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()

                    # Calculate the accuracy for this batch of test sentences, and
                    # accumulate it over all batches.
                    total_eval_accuracy += flat_accuracy(logits, label_ids)
                    total_eval_f1 += flat_f1_score(logits, label_ids)
                    total_eval_recall += flat_recall_score(logits, label_ids)
                    total_eval_precision += flat_precision_score(logits, label_ids)

                # Report the final accuracy for this validation run.
                avg_val_accuracy = total_eval_accuracy / len(test)
                avg_val_recall = total_eval_recall / len(test)
                avg_val_precision = total_eval_precision / len(test)
                # Report the final f1_score for this validation run.
                avg_val_f1 = total_eval_f1 / len(test)
                # Calculate the average loss over all of the batches.
                avg_val_loss = total_eval_loss / len(test)
                # Measure how long the validation run took.
                validation_time = format_time(time.time() - t0)

                # count
                epoch_validation_row_f1.append(avg_val_f1)
                epoch_validation_row_accuracy.append(avg_val_accuracy)
                epoch_validation_row_recall.append(avg_val_recall)
                epoch_validation_row_precision.append(avg_val_precision)
                epoch_validation_row_time.append(validation_time)
                print("  Validation Loss: {0:.2f}".format(avg_val_loss))
                print("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))
                print("  Validation Recall: {0:.2f}".format(avg_val_recall))
                print("  Validation Precision: {0:.2f}".format(avg_val_precision))
                print("  Validation F1-Score: {0:.2f}".format(avg_val_f1))
                print("  Validation took: {:}".format(validation_time))
                logging.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
                logging.info("  Validation Accuracy: {0:.2f}".format(avg_val_accuracy))
                logging.info("  Validation Recall: {0:.2f}".format(avg_val_recall))
                logging.info("  Validation Precision: {0:.2f}".format(avg_val_precision))
                logging.info("  Validation F1-Score: {0:.2f}".format(avg_val_f1))
                logging.info("  Validation took: {:}".format(validation_time))

                # # 保存checkpoint
                # checkpoint = {
                #     'epoch': epoch_i + 1,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': loss,
                # }
                # torch.save(checkpoint, checkpoint_path)
            # epoch end
            # 保存模型
            torch.save(model.state_dict(), model_path)

            training_row_accuracy.append(epoch_training_row_accuracy[-1])
            validation_row_accuracy.append(epoch_validation_row_accuracy[-1])
            training_row_recall.append(epoch_training_row_recall[-1])
            validation_row_recall.append(epoch_validation_row_recall[-1])
            training_row_precision.append(epoch_training_row_precision[-1])
            validation_row_precision.append(epoch_validation_row_precision[-1])
            training_row_f1.append(epoch_training_row_f1[-1])
            validation_row_f1.append(epoch_validation_row_f1[-1])
            training_row_time.append(epoch_training_row_time[-1])
            validation_row_time.append(epoch_validation_row_time[-1])

        training_f1_stats.append(training_row_f1)
        validation_f1_stats.append(validation_row_f1)
        training_accuracy_stats.append(training_row_accuracy)
        validation_accuracy_stats.append(validation_row_accuracy)
        training_recall_stats.append(training_row_recall)
        validation_recall_stats.append(validation_row_recall)
        training_precision_stats.append(training_row_precision)
        validation_precision_stats.append(validation_row_precision)
        training_time_stats.append(training_row_time)
        validation_time_stats.append(validation_row_time)



        data_set += 1

    ### From the cells above, these vectors are filled with measurement values ###

    print("training_f1_stats", training_f1_stats)
    print("validation_f1_stats", validation_f1_stats)
    print("training_accuracy_stats", training_accuracy_stats)
    print("validation_accuracy_stats", validation_accuracy_stats)
    print("training_recall_stats", training_recall_stats)
    print("validation_recall_stats", validation_recall_stats)
    print("training_precision_stats", training_precision_stats)
    print("validation_precision_stats", validation_precision_stats)
    print("training_time_stats", training_time_stats)
    print("validation_time_stats", validation_time_stats)

    training_sizes = [200, 500, 1000, 1500, 2000, 2535]

    f_n_train = []
    for n in training_sizes:
        for x in range(5):
            f_n_train.append(n)

    f_train = []
    for f in training_f1_stats:
        for ff in f:
            f_train.append(ff)

    f_test = []
    for f2 in validation_f1_stats:
        for ff2 in f2:
            f_test.append(ff2)

    records = f_train + f_test
    type1 = ['Train'] * 30
    type2 = ['Test'] * 30
    type_ = type1 + type2
    n = f_n_train + f_n_train

    model = ['BERT'] * (30 * 2)
    measure = ['F1'] * (30 * 2)

    data_plot_f1 = pd.DataFrame({'n': n, 'Data': records, 'Type': type_, 'Model': model, 'Measure': measure})
    if model_name == "dmis-lab/biobert-v1.1":
        data_plot_f1.to_excel('../result/biobert' + str(epochs) + '_F1.xlsx', index=False)
    elif model_name == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":
        data_plot_f1.to_excel('../result/pubmedbert' + str(epochs) + '_F1.xlsx', index=False)
    else:
        data_plot_f1.to_excel('../result/' + args.model_name + str(epochs) + '_F1.xlsx', index=False)
    ## Construct Learning Curve ##
    plt.rcParams["figure.figsize"] = (10, 6)

    sns.set_theme(style="darkgrid")
    ax = sns.pointplot(x="n", y="Data", hue="Type", data=data_plot_f1)
    ax.set(xlabel='Sample Size', ylabel='F1-Score', title=args.model_name + "Learning Curve - F1 Score", )
    if model_name == "dmis-lab/biobert-v1.1":
        plt.savefig('../result/biobert' + str(epochs) + '_F1_Learning_Curve.pdf')
    elif model_name == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext":
        plt.savefig('../result/pubmedbert' + str(epochs) + '_F1_Learning_Curve.pdf')
    else:
        plt.savefig('../result/' + args.model_name + str(epochs) + '_F1_Learning_Curve.pdf')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="RENET: A Deep Learning Approach for Extracting Gene-Disease Associations from Literature")

    parser.add_argument('--ab_fn', type=str, default = None,
            help="file of abstracts")

    parser.add_argument('--stc_fn', type=str, default=None,
            help="file of splitted sentences")

    parser.add_argument('--ner_fn', type=str, default=None,
            help="annotation of disease/gene entities")
    
    parser.add_argument('--label_fn', type=str, default=None,
            help="true G-D-A associations by DisGeNet")

    parser.add_argument('--output_dir', type=str, default = None,
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
        
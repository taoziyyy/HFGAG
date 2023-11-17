import os

from torch import nn
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import datetime
import random
import sys
import warnings
import torch.distributed as dist
import seaborn as sns
import matplotlib.pyplot as plt

from raw import load_documents
from utils.sequence_utils import *
import argparse
import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
import logging

class Config(object):
    """Holds Model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """

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
    kf = KFold(n_splits=3)
    batch_size = 6
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:11117', rank=args.local_rank,
                            world_size=args.world_size)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    torch.cuda.set_device(0)
    # 初始化日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 只有处理器0的进程才会输出日志信息
    if torch.distributed.get_rank() == 0:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_set = 1

    logger.info("Loading data...")

    text_path = args.ab_fn
    sentence_path = args.stc_fn
    ner_path = args.ner_fn
    label_path = args.label_fn
    out_fn = args.output_dir

    gdas, x_word_seq, x_feature = load_documents(text_path, sentence_path, ner_path)
    write_2list_to_file(x_word_seq, "/home/zhangtao/experimentalData/pythonCode/RE/gda-extraction-pytorch/test/seq.txt")
    new_seq = read_file_to_list('/home/zhangtao/experimentalData/pythonCode/RE/gda-extraction-pytorch/test/seq.txt')
    new_sequences = Generate_sequences_data(new_seq, gdas)
    pos_labels = read_pos_labels(label_path)
    y, total_y = get_numpy_labels(gdas, pos_labels)
    final_sequences = [' '.join([str(item) for item in row]) for row in new_sequences]
    final_sequence = [final_sequence[:512] for final_sequence in final_sequences]
    dataset = pd.DataFrame({'Data': final_sequence, 'Label': y})
    texts = list(dataset.Data)

    label = np.array(dataset.Label.astype('int'))
    number_label = dataset.Label.astype('category').cat.categories


    logger.info("Finished loading data.")
    logger.info("Begin training...")
    # config = Config()


    train_dataset_index = []
    test_dataset_index = []
    for train_index, test_index in kf.split(texts):
        train_dataset_index.append(train_index)
        test_dataset_index.append(test_index)


    # Tokenize Data
    # model_name = "bert-base-cased"
    # model_name = "dmis-lab/biobert-v1.1"
    # model_name = "distilbert-base-uncased"
    model_name = 'gpt2'
    MAX_INPUT_LENGTH = 512
    labels_ids = {'neg': 0, 'pos': 1}
    n_labels = len(labels_ids)

    logger.info('Loading configuraiton...')
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=n_labels)

    # Get Model's tokenizer.
    logger.info('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    # default to left padding
    tokenizer.padding_side = "left"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH)
    labels = torch.tensor(label, dtype=torch.long)

    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        # device = torch.device("cuda")
        # torch.cuda.set_device(args.local_rank)  # cpu cuda cuda:index
        torch.cuda.set_device(0)
        logger.info("taotao:args.local_rank---------------%s", args.local_rank)
        if torch.cuda.device_count() > 1:
            logger.info('There are %d GPU(s) available.', torch.cuda.device_count())
        else:
            logger.info("Too few GPU!")
        # os.environ["CUDA_VISIBLE_DIVICES"] = "0,1,2,3"
        # logger.info('We will use the GPU:', torch.cuda.get_device_name(1))
    # If not...
    else:
        logger.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

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

    train_sampler1 = torch.utils.data.distributed.DistributedSampler(train_set1, num_replicas=args.world_size, rank=args.local_rank)
    train_sampler2 = torch.utils.data.distributed.DistributedSampler(train_set2, num_replicas=args.world_size, rank=args.local_rank)
    train_sampler3 = torch.utils.data.distributed.DistributedSampler(train_set3, num_replicas=args.world_size, rank=args.local_rank)

    # train DataLoader
    train_loader1 = DataLoader(
        train_set1,
        sampler=train_sampler1,
        batch_size=batch_size,
        # collate_fn=gpt2_classificaiton_collator,
        num_workers=1)

    train_loader2 = DataLoader(
        train_set2,
        sampler=train_sampler2,
        batch_size=batch_size,
        # collate_fn=gpt2_classificaiton_collator,
        num_workers=1)
    train_loader3 = DataLoader(
        train_set3,
        sampler=train_sampler3,
        batch_size=batch_size,
        # collate_fn=gpt2_classificaiton_collator,
        num_workers=1)

    test_loader1 = DataLoader(
        test_set1,
        sampler=SequentialSampler(test_set1),
        batch_size=batch_size,
        # collate_fn=gpt2_classificaiton_collator,
        num_workers=1)

    test_loader2 = DataLoader(
        test_set2,
        sampler=SequentialSampler(test_set2),
        batch_size=batch_size,
        # collate_fn=gpt2_classificaiton_collator,
        num_workers=1)

    test_loader3 = DataLoader(
        test_set3,
        sampler=SequentialSampler(test_set3),
        batch_size=batch_size,
        # collate_fn=gpt2_classificaiton_collator,
        num_workers=1)

    train_set = [train_loader1, train_loader2, train_loader3]
    test_set = [test_loader1, test_loader2, test_loader3]

    # Get the actual Model.
    logger.info('Loading Model...')
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name,
                                                          config=model_config).cuda(args.local_rank)
    # resize Model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix Model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    # Tell pytorch to run this Model on the GPU.
    model = DDP(model, device_ids=[args.local_rank])
    torch.cuda.empty_cache()

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    epochs = 20
    total_steps = len(train_loader1) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)


    iter = 0  # counter

    total_t0 = time.time()
    training_row_f1 = []
    validation_row_f1 = []
    training_row_accuracy = []
    validation_row_accuracy = []
    training_row_time = []
    validation_row_time = []
    # writer = SummaryWriter()
    set_seed(123)

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(np.array_equal(pred_flat, labels_flat)) / len(labels_flat)

    def flat_f1_score(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, pred_flat, average='weighted')

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

    epoch_training_row_f1 = []
    epoch_validation_row_f1 = []
    epoch_training_row_accuracy = []
    epoch_validation_row_accuracy = []
    epoch_training_row_time = []
    epoch_validation_row_time = []

    # For each epoch...
    for epoch_i in range(0, epochs):
        # Tracking variables.
        predictions_labels = []

        logger.info("")
        logger.info('======== Epoch %s / %s ========', epoch_i + 1, epochs)
        logger.info('Training...')

        # Measure training time
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0
        total_train_accuracy = 0
        total_train_f1 = 0
        model.train()
        train_predictions_labels = []
        true_labels1=[]
        true_labels2=[]
        val_predictions_labels = []

        # For each batch of training data...
        for step, batch in enumerate(train_loader1):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                logger.info("Batch %s  of  %s.  Elapsed: %s.", step, len(train_loader1), elapsed)

            batch = {"input_ids": batch[0].type(torch.long).cuda(args.local_rank),
                     "attention_mask": batch[1].type(torch.long).cuda(args.local_rank),
                     "labels": batch[2].type(torch.long).cuda(args.local_rank)
                     }

            model.zero_grad()

            outputs = model(**batch)
            # loss, logits, attentions = Model(batch)
            loss, logits = outputs[:2]
            total_train_loss += loss.item()

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
            train_predictions_labels += logits.argmax(axis=-1).flatten().tolist()
            true_labels1 += batch["labels"].to('cpu').flatten().tolist()

            # Calculate the accuracy for this batch of train sentences, and
            # accumulate it over all batches.
            # total_train_accuracy += flat_accuracy(predict_content, true_labels)
        total_train_accuracy = accuracy_score(true_labels1, train_predictions_labels)
        total_train_f1 = f1_score(true_labels1, train_predictions_labels)

        # Report the final accuracy for this training run.
        # avg_train_accuracy = total_train_accuracy / len(train_loader1)
        # # Report the final f1_score for this training run.
        # avg_train_f1 = total_train_f1 / len(train_loader1)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_loader1)
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        # count
        logger.info(" Dataset: %s  Cross Validation Round(train,test): %s", data_set, iter)
        logger.info("  Average training loss: %s", avg_train_loss)
        logger.info("  Training F1-Score: %s", total_train_f1)
        logger.info("  Training Accuracy: %s", total_train_accuracy)
        logger.info("  Training epoch took: %s", training_time)

        epoch_training_row_f1.append(total_train_f1)
        epoch_training_row_accuracy.append(total_train_accuracy)
        epoch_training_row_time.append(training_time)

        ### Validation
        logger.info("")
        logger.info("Running Validation...")

        t0 = time.time()

        # Put the Model in evaluation mode
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        total_eval_f1 = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in test_loader1:
            batch = {"input_ids": batch[0].type(torch.long).cuda(args.local_rank),
                     "attention_mask": batch[1].type(torch.long).cuda(args.local_rank),
                     "labels": batch[2].type(torch.long).cuda(args.local_rank)
                     }

            # Tell pytorch not to bother with constructing the compute graph during
            with torch.no_grad():
                outputs = model(**batch)

                # Accumulate the validation loss.
                loss, logits = outputs[:2]
                total_train_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                predict_content = logits.argmax(axis=-1).flatten().tolist()
                true_labels2 = batch["labels"].to('cpu').flatten().tolist()

                # Calculate the accuracy for this batch of train sentences, and
                # accumulate it over all batches.
                # total_train_accuracy += flat_accuracy(predict_content, true_labels)
                total_train_accuracy += accuracy_score(true_labels2, predict_content)
                total_train_f1 += f1_score(true_labels2, predict_content)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(test_loader1)
        # Report the final f1_score for this validation run.
        avg_val_f1 = total_eval_f1 / len(test_loader1)
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(test_loader1)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        # count
        epoch_validation_row_f1.append(avg_val_f1)
        epoch_validation_row_accuracy.append(avg_val_accuracy)
        epoch_validation_row_time.append(validation_time)
        logger.info("  Validation Loss: %s", avg_val_loss)
        logger.info("  Validation F1-Score: %s", avg_val_f1)
        logger.info("  Validation Accuracy: %s", avg_val_accuracy)
        logger.info("  Validation took: %s", validation_time)

    # epoch end

    training_row_f1.append(epoch_training_row_f1[-1])
    validation_row_f1.append(epoch_validation_row_f1[-1])
    training_row_accuracy.append(epoch_training_row_accuracy[-1])
    validation_row_accuracy.append(epoch_validation_row_accuracy[-1])
    training_row_time.append(epoch_training_row_time[-1])
    validation_row_time.append(epoch_validation_row_time[-1])


    # writer.close()



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
    parser.add_argument('--local_rank', type=int, default=None,
                        help="local_rank")
    parser.add_argument('--world_size', type=int, default=None,
                        help="world_size")
    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        # parser.print_help()
        sys.exit(1)

    Run(args)

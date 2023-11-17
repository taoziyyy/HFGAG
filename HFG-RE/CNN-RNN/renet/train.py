
import os
import getpass
import sys
import time
import cPickle
import numpy as np
import tensorflow as tf
from raw import load_documents
from cnn_gru import CRNNModel
import argparse
import pandas as pd
from sklearn.model_selection import KFold

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  window_sizes = [2, 3, 4, 5]
  filter_size = 25
  word_embed_size = 400
  num_word = 82949
  feature_embed_size = 4
  total_feature_size = word_embed_size + feature_embed_size
  sentence_size = 52
  token_size = 175
  batch_size = 64
  label_size = 1
  hidden_size = 100
  max_epochs = 200
  early_stopping = 2
  dropout = 0.1
  lr = 0.001
  l2 = 0.0001
def split_data(word_seq, features, y, train_val_split = 0.7):
    
    train_size = int(len(y) * train_val_split)
    
    train_word_seq = word_seq[:train_size]
    train_feature = features[:train_size]
    X_train = train_word_seq, train_feature
    y_train = y[:train_size]

    val_word_seq = word_seq[train_size:]
    val_feature = features[train_size:]
    X_val = val_word_seq, val_feature
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
    return y

def Run(args):
    """Test NER model implementation.

    When debugging, set max_epochs in the Config object to 1
    so you can rapidly iterate.
    """
    print "Loading data..."
    basedir = os.path.dirname(__file__)
    with open(basedir + "/word_index") as fp:
        word_index = cPickle.load(fp)
        
    text_path = args.ab_fn
    sentence_path = args.stc_fn
    ner_path = args.ner_fn
    label_path = args.label_fn
    out_fn = args.output_dir

    gdas, x_word_seq, x_feature = load_documents(text_path, sentence_path, ner_path, word_index)
    # print(gdas)
    pos_labels = read_pos_labels(label_path)
    y = get_numpy_labels(gdas, pos_labels)

    print "Finished loading data."
    print "Begin training..."

    output_path = args.output_dir
    X_train, y_train, X_val, y_val = split_data(x_word_seq, x_feature, y, 0.75)

    # when
    config = Config()
    model = CRNNModel(config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
      best_val_loss = float('inf')
      best_train_loss = float('inf')
      best_val_epoch = 0
      best_train_epoch = 0

      session.run(init)
      for epoch in xrange(Config.max_epochs):
        print 'Epoch {}'.format(epoch)
        start = time.time()

        train_loss, train_acc, train_prec, train_recall, train_f1 = model.run_epoch(session, X_train, y_train)
        val_loss, val_acc, val_prec, val_recall, val_f1,_ = model.evaluate(session, X_val, y_val)

        print 'Training loss: {0:.4f}'.format(train_loss)
        print 'Training acc: {0:.4f}'.format(train_acc)
        print 'Training prec: {0:.4f}'.format(train_prec)
        print 'Training recall: {0:.4f}'.format(train_recall)
        print 'Training f1: {0:.4f}'.format(train_f1)

        print 'Validation loss: {0:.4f}'.format(val_loss)
        print 'Validation acc: {0:.4f}'.format(val_acc)
        print 'Validation prec: {0:.4f}'.format(val_prec)
        print 'Validation recall: {0:.4f}'.format(val_recall)
        print 'Validation f1: {0:.4f}'.format(val_f1)
        # if val_loss < best_val_loss:
        #   best_val_loss = val_loss
        #   best_val_epoch = epoch
        #
        #   saver.save(session, output_path + 'rel.weights')
        # if epoch - best_val_epoch > config.early_stopping:
        #   break
        # if train_loss < best_train_loss:
        #   best_train_loss = train_loss
        #   best_train_epoch = epoch
        #
        #   saver.save(session, output_path + 'rel.weights')
        # if epoch - best_train_epoch > config.early_stopping:
        #   break
        print 'Total time: {}'.format(time.time() - start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="A Deep Learning Approach for Extracting Gene-Follicle Associations from Literature")

    parser.add_argument('--ab_fn', type=str, default = None,
            help="file of abstracts")

    parser.add_argument('--stc_fn', type=str, default=None,
            help="file of splitted sentences")

    parser.add_argument('--ner_fn', type=str, default=None,
            help="annotation of disease/gene entities")
    
    parser.add_argument('--label_fn', type=str, default=None,
            help="true G-F-A associations by FolGeNet")

    parser.add_argument('--output_dir', type=str, default = None,
            help="Output model")
    
    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)
        
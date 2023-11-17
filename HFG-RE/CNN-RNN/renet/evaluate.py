
import os
import getpass
import sys
import time
import cPickle
import numpy as np
import tensorflow as tf
from cnn_gru import CRNNModel
from raw import load_documents
import argparse
import pandas as pd

class Config(object):
  """Holds model hyperparams and data information.

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
  batch_size = 32
  label_size = 1
  hidden_size = 100
  max_epochs = 25
  early_stopping = 2
  dropout = 0.1
  lr = 0.001
  l2 = 0.0

def evaluate(model_path, X_test, y_test):
    """Test NER model implementation.

    When debugging, set max_epochs in the Config object to 1
    so you can rapidly iterate.
    """

    config = Config()
    model = CRNNModel(config)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
      best_val_loss = float('inf')
      best_val_epoch = 0

      session.run(init)
      
      saver.restore(session, model_path +'rel.weights')

      print '=-=-='
      loss, acc, prec, recall, f1, preds = model.evaluate(session, X_test, y_test)
      if y_test is not None:
          print 'Test loss: {0:.4f}'.format(loss)
          print 'Test acc: {0:.4f}'.format(acc)
          print  'Test prec: {0:.4f}'.format(prec)
          print 'Test recall: {0:.4f}'.format(recall)
          print 'Test f1: {0:.4f}'.format(f1)
    return preds

def read_pos_labels(label_path):
    pos_labels = pd.read_csv(label_path)
    pos_labels.pmid = pos_labels.pmid.astype(str)
    pos_labels.geneId = pos_labels.geneId.astype(str)
    return pos_labels

def get_numpy_labels(gdas, pos_labels):
    gdas = pd.merge(gdas, pos_labels, on=['pmid', "geneId", "follicleId"], how="left")
    gdas = gdas.fillna(0)
    y = gdas.label.values
    return y

def Run(args):
    basedir = os.path.dirname(__file__)
    with open(basedir + "/word_index") as fp:
        word_index = cPickle.load(fp)
        
    text_path = args.ab_fn
    sentence_path = args.stc_fn
    ner_path = args.ner_fn
    label_path = args.label_fn
    out_fn = args.output_fn
    model_dir = args.model_dir
    
    gdas, x_word_seq, x_feature = load_documents(text_path, sentence_path, ner_path, word_index)  
    X_test = x_word_seq, x_feature
    if label_path != None:
        pos_labels = read_pos_labels(label_path)
        y_test = get_numpy_labels(gdas, pos_labels)
    else:
        y_test = None
    
    predicts = evaluate(model_dir, X_test, y_test)
    
    gdas['predict'] = np.concatenate(predicts).astype(int)
    
    gdas.to_csv(out_fn, index=False, sep="\t")
    

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

    parser.add_argument('--model_dir', type=str, default=None,
            help="Input a RENET model")

    parser.add_argument('--output_fn', type=str, default = None,
            help="Output G-D-A classification results")
    
    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)
    
        
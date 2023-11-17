import os
import getpass
import sys
import time
import cPickle

import numpy as np
import tensorflow as tf
from utils.data_iterator import data_iterator
from model import LanguageModel
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Input, Dropout, Layer, Softmax, Dot, Concatenate


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
    max_epochs = 24
    early_stopping = 2
    dropout = 0.1
    lr = 0.001
    l2 = 0.0001


class Attention(Layer):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = Dense(hidden_size, activation='tanh')
        self.attention_softmax = Softmax(axis=1)
        self.dot_product = Dot(axes=(1, 1))

    def call(self, encoder_outputs, decoder_hidden):
        # Attention weights calculation
        attention_hidden_layer = self.attention_weights(decoder_hidden)
        score = self.dot_product([encoder_outputs, attention_hidden_layer])
        attention_weights = self.attention_softmax(score)

        # Context vector calculation
        context_vector = attention_weights * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

class CRNNModel(LanguageModel):

    def __weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def __bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def add_placeholders(self):
        """Generate placeholder variables to represent the input tensors"""

        self.input_word_seq_placeholder = tf.placeholder(tf.int32,
                                                         (None, self.config.sentence_size, self.config.token_size))
        self.input_tag_feature_placeholder = tf.placeholder(tf.int32,
                                                            (None, self.config.sentence_size, self.config.token_size))
        self.labels_placeholder = tf.placeholder(tf.float32, (None, self.config.label_size))
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.is_training_placeholder = tf.placeholder(tf.bool)

    def create_feed_dict(self, input_batch, dropout, is_training, label_batch=None):
        """Creates the feed_dict for softmax classifier.
    
    Args:
      input_batch: A batch of input data.
      label_batch: A batch of label data.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
        feed_dict = {self.input_word_seq_placeholder: input_batch[0],
                     self.input_tag_feature_placeholder: input_batch[1],
                     self.is_training_placeholder: is_training,
                     self.dropout_placeholder: dropout}
        if label_batch is not None:
            feed_dict[self.labels_placeholder] = label_batch
        return feed_dict

    def add_embedding(self):
        """Add embedding layer that maps from vocabulary to vectors.

    Creates an embedding tensor (of shape (len(self.wv), embed_size). Use the
    input_placeholder to retrieve the embeddings for words in the current batch.

    Returns:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    """
        self._feature_embedding = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], \
                                            [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]], dtype=np.float32)
        self.word_embedding = tf.Variable(
            tf.random_uniform((self.config.num_word, self.config.word_embed_size), -0.05, 0.05))
        self.feature_embedding = tf.Variable(tf.constant(self._feature_embedding), dtype=tf.float32)
        # The embedding lookup is currently only implemented for the CPU
        with tf.device('/cpu:0'):
            word_embedding_out = tf.nn.embedding_lookup(self.word_embedding, self.input_word_seq_placeholder)
            feature_embedding_out = tf.nn.embedding_lookup(self.feature_embedding, self.input_tag_feature_placeholder)
        embedding_out = tf.concat([word_embedding_out, feature_embedding_out], -1)
        return tf.reshape(embedding_out, shape=(-1, self.config.token_size, self.config.total_feature_size))

    def add_cnn(self, embedding_out):
        """Adds the 1-hidden-layer NN.
    
    Args:
      window: tf.Tensor of shape (-1, window_size*embed_size)
    Returns:
      output: tf.Tensor of shape (batch_size, label_size)
    """
        with tf.variable_scope("ConvolutionLayer"):
            self.W_convs = []
            self.b_convs = []
            for i, window_size in enumerate(self.config.window_sizes):
                w_weight = tf.get_variable("W" + str(i),
                                           shape=[window_size, self.config.total_feature_size, self.config.filter_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
                b_weight = tf.get_variable("b" + str(i), shape=[self.config.filter_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
                self.W_convs.append(w_weight)
                self.b_convs.append(b_weight)

        self.conv_outs = []
        for i, W_conv in enumerate(self.W_convs):
            conv_out = tf.nn.conv1d(embedding_out, W_conv, stride=1, padding='VALID') + self.b_convs[i]
            conv_out = tf.layers.batch_normalization(conv_out, axis=1, training=self.is_training_placeholder)
            conv_out = tf.reduce_max(conv_out, -2)
            self.conv_outs.append(conv_out)
            tf.add_to_collection("regularization", tf.reduce_sum(tf.square(W_conv)))

        output = tf.reshape(tf.concat(self.conv_outs, -1),
                            shape=(-1, self.config.sentence_size,
                                   len(self.config.window_sizes) * self.config.filter_size))

        output = tf.nn.relu(output)

        return output

    def length(self):
        used = tf.sign(tf.reduce_max(self.input_word_seq_placeholder, 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    def add_rnn_attention(self, cnn_out, rnn_type="GRU", attention_bool=False):

        if rnn_type == "GRU":
            with tf.variable_scope("GRULayer"):
                length = self.length()
                cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=1 - self.dropout_placeholder)

            output, _ = tf.nn.dynamic_rnn(
                cell,
                cnn_out,
                dtype=tf.float32,
                sequence_length=length,
            )

        elif rnn_type == "BIGRU":
            with tf.variable_scope("BIGRULayer"):
                length = self.length()
                fw_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size, name='gru', reuse=tf.AUTO_REUSE,
                                                 activation=tf.nn.tanh)
                fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=1 - self.dropout_placeholder)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=fw_cell, inputs=cnn_out,
                                                         sequence_length=length, dtype=tf.float32)
            output_fw, output_bw = outputs
            output = tf.concat([output_fw, output_bw], axis=-1)

        elif rnn_type == "LSTM":
            with tf.variable_scope("LSTMLayer"):
                length = self.length()
                cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, state_is_tuple=True)
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=1 - self.dropout_placeholder)

            output, _ = tf.nn.dynamic_rnn(
                cell,
                cnn_out,
                dtype=tf.float32,
                sequence_length=length,
            )

        elif rnn_type == "BILSTM":
            with tf.variable_scope("LSTMLayer"):
                length = self.length()
                fw_cell = tf.contrib.rnn.BasicLSTMCell(self.config.hidden_size, name='lstm', reuse=tf.AUTO_REUSE,
                                                       activation=tf.nn.tanh)
                fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=1 - self.dropout_placeholder)

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=fw_cell, inputs=cnn_out,
                                                         sequence_length=length, dtype=tf.float32)
            output_fw, output_bw = outputs
            output = tf.concat([output_fw, output_bw], axis=-1)

        if attention_bool:
            # Attention mechanism
            attention_context = tf.layers.dense(output, self.config.hidden_size, activation=tf.nn.tanh)
            attention_weights = tf.layers.dense(attention_context, 1, activation=tf.nn.softmax)
            attention_output = tf.reduce_sum(output * attention_weights, axis=1)
            return attention_output
        else:
            batch_size = tf.shape(output)[0]
            max_length = tf.shape(output)[1]
            out_size = int(output.get_shape()[2])

            index = tf.range(0, batch_size) * max_length + (length - 1)
            flat = tf.reshape(output, [-1, out_size])

            partitions = tf.reduce_sum(tf.one_hot(index, tf.shape(flat)[0], dtype='int32'), 0)
            # Selecting the elements we want to choose.
            last_timestamps = tf.dynamic_partition(flat, partitions, 2)  # (batch_size, n_dim)
            relevant_output = last_timestamps[1]

            return relevant_output
    def add_classifier(self, rnn_out, bidirectional_rnn = False):

        if bidirectional_rnn:
            matrix_size = len(self.config.window_sizes) * self.config.filter_size * 2
        else:
            matrix_size = len(self.config.window_sizes) * self.config.filter_size

        with tf.variable_scope("DenseLayer"):
            self.U1 = self.__weight_variable([matrix_size, matrix_size])  # (200,200)
            self.b1 = self.__bias_variable([matrix_size])  # (200, )

            self.U2 = self.__weight_variable([matrix_size, 1])  # (200, 1)
            self.b2 = self.__bias_variable([1])  # (1, )

        tf.add_to_collection("regularization", tf.reduce_sum(tf.square(self.U1)))
        tf.add_to_collection("regularization", tf.reduce_sum(tf.square(self.b1)))
        tf.add_to_collection("regularization", tf.reduce_sum(tf.square(self.U2)))
        tf.add_to_collection("regularization", tf.reduce_sum(tf.square(self.b2)))

        dense_out = tf.nn.relu(tf.matmul(rnn_out, self.U1) + self.b1)  # (200,) * (200, 200) + (200, )
        # dense_out = tf.layers.batch_normalization(dense_out, training=self.is_training_placeholder)

        output = tf.matmul(dense_out, self.U2) + self.b2  # (200)

        return output

    def add_loss_op(self, pred):
        """Adds cross_entropy_loss ops to the computational graph.

    Args:
      pred: A tensor of shape (batch_size, n_classes)
    Returns:
      loss: A 0-d tensor (scalar)
    """
        loss = self.config.l2 * sum(tf.get_collection("regularization"))
        loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=self.labels_placeholder))
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

    Args:
      loss: Loss tensor, from cross_entropy_loss.
    Returns:
      train_op: The Op for training.
    """
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
        return train_op

    def __init__(self, config):
        """Constructs the network using the helper functions defined above."""
        self.config = config
        self.add_placeholders()
        self.embedding_out = self.add_embedding()
        self.cnn_out = self.add_cnn(self.embedding_out)
        self.rnn_out = self.add_rnn_attention(self.cnn_out, "BILSTM", False)  # GRU/BIGRU/LSTM/BILSTM + attention = False

        self.y = self.add_classifier(self.rnn_out, True)  # bidirectional_rnn = False
        self.loss = self.add_loss_op(self.y)
        self.predictions = tf.nn.sigmoid(self.y)

        one_hot_prediction = tf.round(self.predictions)
        correct_prediction = tf.equal(
            self.labels_placeholder, one_hot_prediction)
        self.correct_predictions = tf.reduce_sum(tf.cast(correct_prediction, 'int32'))

        self.true_positives = tf.reduce_sum(one_hot_prediction * self.labels_placeholder)
        self.predicted_positives = tf.reduce_sum(one_hot_prediction)
        self.possible_positives = tf.reduce_sum(self.labels_placeholder)

        self.train_op = self.add_training_op(self.loss)

    def run_epoch(self, session, input_data, input_labels,
                  shuffle=True, verbose=True):
        data_X, data_X_feature = input_data
        data_y = input_labels
        dp = self.config.dropout
        # We're interested in keeping track of the prec and recall
        total_loss = []
        total_correct_examples = 0
        total_true_positives = 0
        total_predicted_positives = 0
        total_possible_positives = 0
        total_processed_examples = 0
        total_steps = len(data_X) / self.config.batch_size

        # Do not output the do_feature
        for step, (x, x_feature_tag, y) in enumerate(
                data_iterator(data_X, data_X_feature, data_y, batch_size=self.config.batch_size,
                              shuffle=shuffle)):
            feed = self.create_feed_dict(input_batch=(x, x_feature_tag), dropout=dp, is_training=True, label_batch=y)
            loss, total_correct, total_true_pos, total_pred_pos, total_possible_pos, _ = session.run(
                [self.loss,
                 self.correct_predictions,
                 self.true_positives,
                 self.predicted_positives,
                 self.possible_positives,
                 self.train_op],
                feed_dict=feed)
            total_processed_examples += len(x)
            total_correct_examples += total_correct
            total_true_positives += total_true_pos
            total_predicted_positives += total_pred_pos
            total_possible_positives += total_possible_pos

            total_loss.append(loss)
            ##
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{0} / {1} : loss = {2:.4f}, acc = {3:.4f}, prec = {4:.4f}, recall = {5:.4f}'.format(
                    step, total_steps, np.mean(total_loss), total_correct_examples / float(total_processed_examples),
                                                            total_true_positives / float(total_predicted_positives),
                                                            total_true_positives / float(total_possible_positives)))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()
        prec = total_true_positives / float(total_predicted_positives)
        recall = total_true_positives / float(total_possible_positives)
        f1 = (2 * prec * recall)/(prec + recall)
        return np.mean(total_loss), total_correct_examples / float(total_processed_examples), \
               prec, recall, f1

    def evaluate(self, session, X, y=None):
        """Make predictions from the provided model."""
        # If y is given, the loss is also calculated
        # We deactivate dropout by setting it to 1
        dp = 0
        losses = []
        results = []
        data_X, data_X_feature = X
        if np.any(y):
            data = data_iterator(data_X, data_X_feature, y, batch_size=self.config.batch_size,
                                 shuffle=False)
        else:
            data = data_iterator(data_X, data_X_feature, batch_size=self.config.batch_size,
                                 shuffle=False)

        total_loss = []
        total_correct_examples = 0
        total_true_positives = 0
        total_predicted_positives = 0
        total_possible_positives = 0
        total_processed_examples = 0
        total_steps = len(data_X) / self.config.batch_size

        for step, (x, x_feature_tag, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=(x, x_feature_tag), dropout=dp, is_training=False, label_batch=y)
            if y is not None:
                loss, total_correct, total_true_pos, total_pred_pos, total_possible_pos, preds = session.run(
                    [self.loss,
                     self.correct_predictions,
                     self.true_positives,
                     self.predicted_positives,
                     self.possible_positives,
                     self.predictions],
                    feed_dict=feed)

                total_correct_examples += total_correct
                total_true_positives += total_true_pos
                total_predicted_positives += total_pred_pos
                total_possible_positives += total_possible_pos
                total_loss.append(loss)
            else:
                preds = session.run(self.predictions, feed_dict=feed)

            predicted_indices = np.round(preds)
            results.extend(predicted_indices)
            total_processed_examples += len(x)

            # print predicted_indices
        if y is None:
            return None, None, None, None, results

        prec = total_true_positives / float(total_predicted_positives)
        recall = total_true_positives / float(total_possible_positives)
        f1 = (2 * prec * recall) / (prec + recall)
        return np.mean(total_loss), total_correct_examples / float(total_processed_examples), \
               prec, recall, f1, results

    def predict(self, session, X, y=None):
        """Make predictions from the provided model."""
        # If y is given, the loss is also calculated
        # We deactivate dropout by setting it to 1
        dp = 0
        losses = []
        results = []
        data_X, data_X_feature = X
        if np.any(y):
            data = data_iterator(data_X, data_X_feature, y, batch_size=self.config.batch_size,
                                 shuffle=False)
        else:
            data = data_iterator(data_X, data_X_feature, batch_size=self.config.batch_size,
                                 shuffle=False)
        for step, (x, x_feature_tag, y) in enumerate(data):
            feed = self.create_feed_dict(input_batch=(x, x_feature_tag), dropout=dp, is_training=False)
            if np.any(y):
                feed[self.labels_placeholder] = y
                loss, preds = session.run(
                    [self.loss, self.predictions], feed_dict=feed)
                losses.append(loss)
            else:
                preds = session.run(self.predictions, feed_dict=feed)
            predicted_indices = np.round(preds)
            results.extend(predicted_indices)
        return np.mean(losses), results


def test_graph():
    config = Config()
    model = CRNNModel(config)


if __name__ == "__main__":
    test_graph()
#   repredict()

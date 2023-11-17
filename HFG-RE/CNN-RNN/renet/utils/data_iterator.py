from collections import defaultdict
import numpy as np
def padding(text, Max_token=175, Max_sentence=52):
    pad = np.zeros(shape=(Max_sentence, Max_token), dtype=int)
    for i, sentence in enumerate(text):
        for j, token in enumerate(sentence):
            if i < Max_sentence and j < Max_token:
                pad[i, j] = token
    return pad

def padding_feat_tag(text, Max_token=175, Max_sentence=52):
    pad = np.zeros(shape=(Max_sentence, Max_token), dtype=int)
    for i, sentence in enumerate(text):
        for j, token in enumerate(sentence):
            if i < Max_sentence and j < Max_token:
                pad[i, j] = np.array(token[0])
    return pad

def data_iterator(data_X, data_X_feature, data_y=None, batch_size=32, shuffle=True):
  # Optionally shuffle the data before training
  dim_x, dim_y = 52, 175
  if shuffle:
    indices = np.random.permutation(len(data_X))
  else:
    indices = np.arange(len(data_X))

  total_processed_examples = 0
  total_steps = int(np.ceil(len(data_X) / float(batch_size)))
  for step in xrange(total_steps):
    # Create the batch by selecting up to batch_size elements
    batch_start = step * batch_size
    
    indices_batch = indices[batch_start:batch_start + batch_size]
    
    X = np.empty((len(indices_batch), dim_x, dim_y), dtype = int)
    X_feature_tag = np.empty((len(indices_batch), dim_x, dim_y), dtype = int)
    
    y = np.empty((len(indices_batch), 1), dtype = int) if np.any(data_y) else None

        # Generate data
    for i, index in enumerate(indices_batch):
    # Store volume
        X[i, :, :] = padding(data_X[index])
        X_feature_tag[i, :, :] = padding_feat_tag(data_X_feature[index])
       
        if data_y is not None: y[i, 0] = data_y[index]
  
    ###
    yield X, X_feature_tag, y
    total_processed_examples += len(X)
  # Sanity check to make sure we iterated over all the dataset as intended
  assert total_processed_examples == len(data_X), 'Expected {} and processed {}'.format(len(data_X), total_processed_examples)

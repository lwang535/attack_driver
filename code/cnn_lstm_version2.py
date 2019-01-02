import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM, ActivityRegularization
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import metrics
from sklearn.metrics import auc, f1_score
from change_embedding import change_embedding
from model import kerasModel

def text_to_array(text, embeddings_index):
  '''
  这个函数在每条评论中取前30个词，并将其变成词向量，最后输出30*300的array
  '''
  empyt_emb = np.zeros(300) # empyt_emb is a list with 300 zeros.
  text = text[:-1].split()[:30] # get single question sentence and split it to many words, get first 30 words and discard last symbol
  embeds = [embeddings_index.get(x, empyt_emb) for x in text] 
  # dict.get(a,b) means that if a is not in dict's keys, return b; if a is in dict's keys, return a.
  embeds += [empyt_emb] * (30 - len(embeds))
  # If question content is less than 30 words, expend the array to 30*300 anyway.
  return np.array(embeds)

def get_raw_data(embeddings_index, train_file_name, train_nrows, test_file_name, test_nrows, valid_size = 0.3):
  train_df = pd.read_csv(train_file_name, nrows = train_nrows)
  train_df, val_df = train_test_split(train_df, test_size = valid_size)
  print('True')
  val_vects = np.array([text_to_array(X_text, embeddings_index) for X_text in tqdm(val_df["question_text"])])
  val_y = np.array(val_df["target"])

  test_df = pd.read_csv("../data/test.csv", nrows = test_nrows)
  test_vects = np.array([text_to_array(X_text, embeddings_index) for X_text in tqdm(test_df["question_text"])])
  return train_df, val_df, val_vects, val_y, test_df, test_vects

def batch_gen(embeddings_index, train_df, batch_size = 1024):
  '''
  这个函数返回一个generator的对象，用于后面fit_generator中，可以减少对内存的需要。
  generator的形式是(30*300*batch_size的词向量矩阵，batch_size的因变量向量)
  '''
  n_batches = math.ceil(len(train_df) / batch_size)
  while True: 
    train_df = train_df.sample(frac=1.)  # Shuffle the data.
    for i in range(n_batches):
      texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
      text_arr = np.array([text_to_array(text, embeddings_index) for text in texts])
      yield (text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size]))

def validation_predict(model, val_vects, val_y, display = 0):
  # predict prob on valid set.
  preds_val = model.predict(val_vects)

  # roc and auc.
  fpr, tpr, thresholds = metrics.roc_curve(val_y, preds_val, pos_label=1)
  plt.plot(fpr, tpr, marker = 'o')
  if display:
    plt.show()

  AUC = auc(fpr, tpr)
  print("AUC score is ", AUC)

  # use different threshold to get f1 score.
  for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = f1_score(val_y, (np.array([i[0] for i in preds_val]) > thresh).astype(np.int), average='macro')
    print('F1 score at threshold {0} is {1}:'.format(thresh, f1))

def test_predict(model, test_vects, test_df, thresh):
  preds = model.predict(test_vects)
  y_te = (np.array([i[0] for i in preds]) > thresh).astype(np.int)

  print(Counter(y_te))

  submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
  return submit_df
  

def runModel(
    embedname,
    train_file_name,
    train_nrows,
    test_file_name,
    test_nrows,
    valid_size,
    batch_size,
    steps_per_epoch,
    epochs,
    verbose,
    display,
    thresh
  ):
  # read embeddings, and get a dictionary with word-keys and array values.
  embeddings_index = change_embedding(embedname)
  print('Found %s word vectors.' % len(embeddings_index))
  
  # get train, valid, test set.
  train_df, val_df, val_vects, val_y, test_df, test_vects = get_raw_data(
    embeddings_index,
    train_file_name,
    train_nrows,
    test_file_name,
    test_nrows,
    valid_size
  )
  
  # model.
  model = kerasModel()
  model.fit_generator(
    batch_gen(embeddings_index, train_df, batch_size),
    steps_per_epoch,
    epochs,
    verbose,
    validation_data = (val_vects, val_y)
  )

  # predict proba on valid set.
  validation_predict(model, val_vects, val_y, display)

  # predict proba on test set.
  submit_df = test_predict(model, test_vects, test_df, thresh)

  submit_df.to_csv("submission.csv", index = False)

if __name__ == '__main__':
  runModel(
    embedname = 'glove',
    train_file_name = '../data/train.csv',
    train_nrows = 20000,
    test_file_name = '../data/test.csv',
    test_nrows = 10000,
    valid_size = 0.3,
    batch_size = 1024,
    steps_per_epoch = 30,
    epochs = 1,
    verbose = True,
    display = 0,
    thresh = 0.5
  )

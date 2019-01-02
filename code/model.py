import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, LSTM, ActivityRegularization


class kerasModel():
  def __init__(self):
    self.model = Sequential()

    self.model.add(Conv1D(128,kernel_size=3,activation='relu',input_shape=(30,300)))
    self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    self.model.add(ActivityRegularization(l1=0.01, l2=0.01))
    self.model.add(Dense(1,activation='sigmoid')) # 此处用softmax精度会超级低

    self.model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

  def fit_generator(self, train_df_generator, steps_per_epoch, epochs, verbose, validation_data):
    self.model.fit_generator(
      train_df_generator,
      steps_per_epoch = steps_per_epoch,
      epochs = epochs,
      verbose = verbose,
      validation_data = validation_data)

  def predict(self, X):
    return self.model.predict(X)

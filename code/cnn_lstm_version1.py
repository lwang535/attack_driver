import numpy as np
import pandas as pd
import os
os.listdir('./')
######################################## 读入embedding的词典 ######################################
# import embedding information from glove.8400.300d.txt
# The result is a dictionary with word-keys and array values.
from tqdm import tqdm
embeddings_index = {}
f = open('./embeddings/glove.840B.300d/glove.840B.300d.txt','r', encoding='UTF-8')
for line in tqdm(f):
    values = line.split(" ")
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

def text_to_array(text):
    '''
    这个函数在每条评论中取前30个词，并将其变成词向量，最后输出30*300的array
    '''
    empyt_emb = np.zeros(300) # empyt_emb is a list with 300 zeros.
    text = text[:-1].split()[:30] # get single question sentence and split it to many words, get first 30 words and discard last symbol
    embeds = [embeddings_index.get(x, empyt_emb) for x in text] 
    # dict.get(a,b) means that if a is not in dict's keys, return b; if a is in dict's keys, return a.
    embeds+= [empyt_emb] * (30 - len(embeds))
    # If question content is less than 30 words, expend the array to 30*300 anyway.
    return np.array(embeds)

########################################### 此处读入总的train.csv 并划分为训练集和验证集 #########################
from sklearn.model_selection import train_test_split
train_df = pd.read_csv("train.csv",nrows=30000)
#train_df = pd.read_csv("train.csv")
train_df, val_df = train_test_split(train_df, test_size=0.3)

val_vects = np.array([text_to_array(X_text) for X_text in tqdm(val_df["question_text"])])
val_y = np.array(val_df["target"])

batch_size = 90
import math
def batch_gen(train_df):
    '''
    这个函数返回一个generator的对象，用于后面fit_generator中，可以减少对内存的需要。
    generator的形式是(30*300*90的词向量矩阵，90的因变量向量)
    '''
    n_batches = math.ceil(len(train_df) / batch_size)
    while True: 
        train_df = train_df.sample(frac=1.)  # Shuffle the data.
        for i in range(n_batches):
            texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 1]
            text_arr = np.array([text_to_array(text) for text in texts])
            yield (text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size]))

############################################# 搭建，编译以及训练模型 ##########################################
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout,LSTM,ActivityRegularization

model = Sequential()

model.add(Conv1D(128,kernel_size=3,activation='relu',input_shape=(30,300)))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(ActivityRegularization(l1=0.01, l2=0.01))
model.add(Dense(1,activation='sigmoid')) # 此处用softmax精度会超级低

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit_generator(batch_gen(train_df),steps_per_epoch = 100,epochs=10,verbose=True,validation_data=(val_vects, val_y))


############################################# 根据模型进行预测 并给出submission file#############################
test_df = pd.read_csv("test.csv")
test_vects = np.array([text_to_array(X_text) for X_text in tqdm(test_df["question_text"])])

preds = model.predict(test_vects)
y_te = (np.array([i[0] for i in preds])>0.5).astype(np.int)
from collections import Counter  
print(Counter(y_te))
submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
submit_df.to_csv("submission.csv", index=False)

############################################ 用验证集验证本地的 模型效果 ###############################
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
preds_val = model.predict(val_vects)
fpr, tpr, thresholds = metrics.roc_curve(val_y, preds_val, pos_label=1)
plt.plot(fpr,tpr,marker = 'o')
plt.show()
from sklearn.metrics import auc 
AUC = auc(fpr, tpr)
print("AUC score is "+AUC)
from sklearn.metrics import f1_score
f1_score(val_y, (np.array([i[0] for i in preds_val])>0.5).astype(np.int), average='macro')  

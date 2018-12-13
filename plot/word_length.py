import numpy as np
import pandas as pd
dat = pd.read_csv("train.csv")

dat.shape
# (1306122, 3)
dat.describe
# ‘question_text’ 'target'
text=dat.loc[0:1000,'question_text']
target=dat.loc[0:1000,'target']

import nltk
nltk.word_tokenize(text[0])

total_length = [len(list(nltk.word_tokenize(i)) for i in text]
df = pd.DataFrame(text,target)
df.to_csv('word_length.csv')

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:03:10 2021


"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:18:27 2020

@author: ONS1KOR
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
model=Sequential()
import numpy as np
import pandas as pd
from numpy import array
df=pd.DataFrame()
text_data=[]
train=pd.read_excel('Forecast4.xlsx')
q=list(train['Value'])


out_seq = array([q[i] for i in range(len(q))])
q=out_seq.tolist()
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)
print(q)
q.append(1)
raw_seq=q[:48]
n_steps=3
X, y = split_sequence(raw_seq, n_steps)
print(X,y)
X = X.reshape((X.shape[0], X.shape[1], 1))
print(X.shape[0], X.shape[1])
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model

x_input=q[42:61]
print(x_input)
x_test,y_test=split_sequence(x_input, n_steps)
model.fit(X, y, epochs=5000, verbose=0)
print(x_test)
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1], 1))
yhat = model.predict(x_test, verbose=0)
print(yhat)

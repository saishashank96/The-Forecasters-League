# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 13:53:21 2021

@author: ONS1KOR
"""
from pmdarima import auto_arima
import pmdarima
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from numpy import array

df=pd.DataFrame()
text_data=[]
train=pd.read_excel(r'C:\Users\ons1kor\Desktop\Forecast\Forecast.xlsx')
q=list(train['Value'])
out_seq = array([q[i] for i in range(len(q))])
q=out_seq.tolist()
print(q)
q.append(1)
#val=list(p[['Software Bug']].values())
#print(val)
train,test=q[:48],q[48:49]
print(pmdarima.arima.nsdiffs(train,4,max_D=9))
print(train)
forplot=[]
# =============================================================================
stepwise_model=auto_arima(train,m=12,p=0,d=2,q=0,start_p=0,start_q=0,max_q=12,max_p=12, error_action='ignore', suppress_warnings=True)
model=stepwise_model.fit(train)

for i in range(len(test)):
    
    q1=model.predict(n_periods=1)
    train.append(q1)
    print(train)
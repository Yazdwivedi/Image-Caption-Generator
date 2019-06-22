import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from tqdm import tqdm
import re
from pickle import dump

from numpy import all, array, uint8

X1=pickle.load(open("X1.pkl","rb"))

X2=pickle.load(open("X2.pkl","rb"))

y=pickle.load(open("y.pkl","rb"))

tokenizer=pickle.load(open("tokenizer.pkl","rb"))

X1=np.array([np.array(x) for x in X1])
X2=np.array([np.array(x) for x in X2])

def clean(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]=float(data[i][j])
    return data        

from keras.models import Model,Sequential
from keras.layers import Input,Dense,Flatten,Embedding,Dropout,LSTM
from keras.layers.merge import add
 
b1=Input(shape=(38,))
b2=Embedding(len(tokenizer.word_index)+1,100,mask_zero=True)(b1)
b3=LSTM(64)(b2)

a1=Input(shape=X2.shape[1:])
a2=Dense(64,activation="relu")(a1)

c1=add([b3,a2])
c2=Dense(1000,activation="relu")(c1)
c3=Dense(len(tokenizer.word_index)+1,activation="softmax")(c2)

model=Model(inputs=[b1,a1],outputs=c3)
model.compile(optimizer="adam",loss="categorical_crossentropy")

model.fit([X1,X2],y,epochs=40,batch_size=128)


temp=np.array([np.array([1,2,3,4]),np.array([5,6,7,8])])
model.save("model.h5")

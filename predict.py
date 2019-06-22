import numpy as np
import pandas as pd
import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
import pickle
from numpy import argmax
import cv2
import os
from keras.models import load_model 
from keras.preprocessing.sequence import pad_sequences

vgg=VGG16()
vgg.layers.pop()
vgg=Model(inputs=vgg.inputs,outputs=vgg.layers[-1].output)

tokenizer=pickle.load(open("tokenizer.pkl","rb"))
model=load_model("model.h5")

val=tokenizer.texts_to_sequences(["stair"])[0]
val=argmax(val)

def get_word(tokenizer,integer):
    for word,index in tokenizer.word_index.items():
        if index==integer:
            return word
            
def get_caption(vgg,img):
    seq="beg"
    fcaption=[]
    fcaption.append(seq)
    pimg=vgg.predict(img)
    for i in range(20):
        s=tokenizer.texts_to_sequences([seq])[0]
        s=pad_sequences([s],maxlen=38,padding="post")
        l=[s,pimg]
        pred=model.predict(l)
        print(pred)
        word=get_word(tokenizer,argmax(pred))
        fcaption.append(word)
        seq=" ".join(fcaption)
        if word=="end":
            return seq
    return seq    
            
image_dict="C:\\Users\\SWADESH PLYWOODS\\Desktop\\Machine learning\\Image Caption part 2\\testing\\kiss.jpg"
img=cv2.resize(cv2.imread(image_dict),(224,224))
img.resize((1,224,224,3))            
get_caption(vgg,img)
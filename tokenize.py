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

train_dict=pickle.load(open("train_dict.pkl","rb"))
image_dict=pickle.load(open("img_dict.pkl","rb"))

def get_list(data):
    d=[]
    for y in data:
        for x in y.split():
            if x not in d:
                d.append(x)
    return d

d=get_list(words)      

def get_req_size(vals,size):
    new_dict=dict()
    i=0
    for key,item in vals.items():
        new_dict[key]=item
        i+=1
        if i>=size:
            break;
    return new_dict

def get_image_data(train,image):
    new_dict=dict()
    for key,item in train.items():
        new_dict[key]=image[key]
    return image_dict    

train_dict=get_req_size(train_dict,3000)
image_dict=get_image_data(train_dict,image_dict)    


def get_words(train_dict):
    names=[]
    for key,item in train_dict.items():
        names.append(item)
    return names
 
    
def get_maxlen(words):
    return max([len(x.split()) for x in words])

def get_tokenizer(words):
    tokenizer=Tokenizer()
    words=get_list(words)
    tokenizer.fit_on_texts(words)
    return tokenizer

def get_tokenized(train_dict,image_dict,tokenizer,maxlen):
    X1=[]
    X2=[]
    y=[]
    for key,line in tqdm(train_dict.items()):
        tokenize=tokenizer.texts_to_sequences([line])[0]
        for i in range(1,len(tokenize)):
            in_seq,out_seq=tokenize[:i],tokenize[i]
            in_seq=pad_sequences([in_seq],maxlen=maxlen,padding="post")
            out_seq=to_categorical(out_seq,num_classes=len(tokenizer.word_index)+1)
            X1.append(in_seq)
            X2.append(image_dict[key])
            y.append(out_seq)
    return np.array(X1),np.array(X2),np.array(y)        

        
words=get_words(train_dict)  
maxlen=get_maxlen(words)
tokenizer=get_tokenizer(words)


X1,X2,y=get_tokenized(train_dict,image_dict,tokenizer,maxlen)

X1.resize((34875,38))
X2.resize((34875,4096))
dump(tokenizer,open("tokenizer.pkl","wb"))
dump(X1,open("X1.pkl","wb"))
dump(X2,open("X2.pkl","wb"))
dump(y,open("y.pkl","wb"))

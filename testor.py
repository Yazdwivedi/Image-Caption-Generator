import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.applications.vgg16 import VGG16
from tqdm import tqdm
import os
import cv2
from pickle import dump
import re

model=VGG16()

model.layers.pop()

model=Model(inputs=model.inputs,outputs=model.layers[-1].output)

print(model.summary())

train_dir="C:\\Users\\SWADESH PLYWOODS\\Desktop\\Machine learning\\Image Caption part 2\\Flickr_8k.trainImages.txt"
test_dir="C:\\Users\\SWADESH PLYWOODS\\Desktop\\Machine learning\\Image Caption part 2\\Flickr_8k.testImages.txt"
token_dir="C:\\Users\\SWADESH PLYWOODS\\Desktop\\Machine learning\\Image Caption part 2\\Flickr8k.token.txt"
image_dir="C:\\Users\\SWADESH PLYWOODS\\Desktop\\Machine learning\\Image Caption part 2\\Flicker8k_Dataset"

def get_image_names(file):
    names=[]
    f=open(file,"r")
    f=f.read()
    for line in tqdm(f.split("\n")):
        names.append(line)
    return names    

def get_captions(file):
    names=dict()
    f=open(file,"r")
    fi=f.read()
    for line in tqdm(fi.split("\n")):
        try:
            whole=line.split("\t")
            cap=whole[1]
            wholeimg=whole[0].split("#")
            img=wholeimg[0]
            score=wholeimg[1]
            names[img]=cap
            if score==4:
                names[img]=cap
        except Exception as e:
            print(e)
    return names                

def get_final_data(name_dict,names):
    name=dict()
    for key,item in name_dict.items():
        if key in names:
            name[key]=item
    return name        

def save_data(names):
    file=open("final.txt","a")
    for key,item in names.items():
        string=key+"\t"+item
        file.write(string)
        file.write("\n")
        

def clean_dict(dic):
    cleaned=dict()
    for key,item in dic.items():
        temp=re.sub("[^a-zA-Z]"," ",item)
        temp=temp.split()
        temp=[x.lower() for x in temp]
        temp=" ".join(temp)    
        temp="<beg> "+temp+" <end>"
        cleaned[key]=temp
    return cleaned   


train_dict=get_captions(token_dir)
train_imgs=get_image_names(train_dir)
train_dict=get_final_data(train_dict,train_imgs)
train_dict=clean_dict(train_dict)
dump(train_dict,open("train_dict.pkl","wb"))
save_data(train_dict)


def get_image_data(train_dict,directory,model):
    img_dict=dict()
    for key in tqdm(train_dict.keys()):
        path_to_img=os.path.join(directory,key)
        img=cv2.resize(cv2.imread(path_to_img),(224,224))
        img=img.reshape(1,224,224,3)
        img=model.predict(img)
        img_dict[key]=img
    return img_dict

img_dict=get_image_data(train_dict,image_dir,model) 

dump(img_dict,open("img_dict.pkl","wb"))



  
        
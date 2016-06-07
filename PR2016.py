# -*- coding: utf-8 -*-
import numpy as np
import os,cPickle
from scipy.misc import imread, imresize, imsave

def load_data(img_size,path = '/Users/subercui/Downloads/pr2016'):
    train_dir=os.path.join(path, 'trainset%d.pkl'%img_size)
    valid_dir=os.path.join(path, 'validset%d.pkl'%img_size)
    if not os.path.exists(train_dir):
        make_dataset(img_size,path)        
    (X_train,y_train)=cPickle.load(open(train_dir,'r'))
    (X_valid,y_valid)=cPickle.load(open(valid_dir,'r'))
    return (X_train,y_train),(X_valid,y_valid)

def make_dataset(img_size,path = '/Users/subercui/Downloads/pr2016'):
    
    X_train=[]
    y_train=[]
    #iteratively load from 7 catorgary folders
    for i in range(0,7):
        folder=os.path.join(path, 'train','%02d'%i)
        assert os.path.exists(folder)
        for img_path in filesinroot(folder,'.jpg',0): 
            print img_path
            img = imread(os.path.join(folder,img_path))#1000*750,默认长边在前才是正方向
            if img.shape[0]<img.shape[1]:
                img=img.transpose(1,0,2)
            img=imresize(img, (img_size,img_size)).transpose(2,0,1)
            X_train.append(img)
            y_train.append(i)             
    X_train = np.array(X_train, dtype="uint8")
    y_train = np.array(y_train, dtype="uint8")
    
    assert y_train.shape[0]==X_train.shape[0]
    assert y_train.ndim==1
    assert X_train.ndim==4 and X_train.shape[1]==3
    y_train = np.reshape(y_train, (len(y_train), 1))
    
    train_dir=os.path.join(path, 'trainset%d.pkl'%img_size)
    cPickle.dump((X_train,y_train),open(train_dir,'w'),-1)
    
    del X_train,y_train
    
    X_valid=[]
    y_valid=[]
    #iteratively load from 7 catorgary folders
    for i in range(0,7):
        folder=os.path.join(path, 'valid','%02d'%i)
        assert os.path.exists(folder)
        for img_path in filesinroot(folder,'.jpg',0):
            print img_path
            img = imread(os.path.join(folder,img_path))#1000*750,默认长边在前才是正方向
            if img.shape[0]<img.shape[1]:
                img=img.transpose(1,0,2)
            img=imresize(img, (img_size,img_size)).transpose(2,0,1)
            X_valid.append(img)
            y_valid.append(i)             
    X_valid = np.array(X_valid, dtype="uint8")
    y_valid = np.array(y_valid, dtype="uint8")
    
    assert y_valid.shape[0]==X_valid.shape[0]
    assert y_valid.ndim==1
    assert X_valid.ndim==4 and X_valid.shape[1]==3
    y_valid = np.reshape(y_valid, (len(y_valid), 1))
    
    valid_dir=os.path.join(path, 'validset%d.pkl'%img_size)
    cPickle.dump((X_valid,y_valid),open(valid_dir,'w'),-1)
    
    
    
    
    
def filesinroot(dir,wildcard,recursion):#目录中的文件
    matchs=[]
    exts=wildcard.split()
    for root,subdirs,files in os.walk(dir):
        for name in files:
            for ext in exts:
                if(name.endswith(ext)):
                    matchs.append(name)
                    break
        if(not recursion):
            break
    return matchs
    
if __name__=='__main__':
    (X_train,y_train),(X_valid,y_valid)=load_data(250)
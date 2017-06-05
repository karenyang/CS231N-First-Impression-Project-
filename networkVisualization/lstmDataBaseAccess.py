'''
code mostly copied from CORRECT_LSTM_notebook
'''

# -*- coding: utf-8 -*- 
from __future__ import print_function, division

import pickle
import sys

import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import matplotlib.pyplot as plt


from PIL import Image
import time
import copy
import shutil

from torchvision import  transforms


def get_img_audio_label(dataset_dir,audio_dataset_dir,label_dataset_dir):
    """Returns a list of np.array(img_paths), np.array(audio_paths),
        np.array(labels), np.array(raw_movienames)
    Args:
    dataset for video, for audio_feats from pyAudioAnalysis, labels
    """
 
    print("processing dataset: "+ dataset_dir)
    img_paths = [] 
    audio_paths=[]
    raw_movienames = []
    labels = []

    annotaion_filename = label_dataset_dir + "/annotation_training.pkl"
    
    with open(annotaion_filename, 'rb') as f:
        label_dicts = pickle.load(f, encoding='latin1') 

    for movie in os.listdir(dataset_dir):
        fileEnding ='_50uniform' #TODO: figure out how to make more general
        if fileEnding not in movie: continue #skip non-movie files
        raw_moviename = movie.replace(fileEnding,'.mp4')      
        big_five = [label_dicts['extraversion'][raw_moviename], 
                    label_dicts['neuroticism'][raw_moviename],
                    label_dicts['agreeableness'][raw_moviename],
                    label_dicts['conscientiousness'][raw_moviename],
                    label_dicts['openness'][raw_moviename] ]
                    #label_dicts['interview'][raw_moviename]]      
        movie_path = os.path.join(dataset_dir, movie)
        mv_partitions = []
        p = 0
        all_imgs = os.listdir(movie_path)
        assert(len(all_imgs) >= num_partition)
        opened = True
        for i in range(num_partition):
            path = os.path.join(movie_path, all_imgs[i])
            try:
                open(path)
            except:
                print('image failed to open',path)
                opened = False
                
            mv_partitions.append(path)
        assert(len(mv_partitions)==num_partition)
        
        
        audiofeat_path = os.path.join(audio_dataset_dir,raw_moviename+'.wav.csv')
        try:
            open(audiofeat_path)
        except:
            print('audio failed to open',path)
            opened = False
        if opened :
            img_paths.append(mv_partitions)
            audio_paths.append(audiofeat_path)
            raw_movienames.append(raw_moviename)
            labels.append(big_five)
            
    
    return np.array(img_paths),np.array(audio_paths),\
                np.array(labels), np.array(raw_movienames)

def default_img_loader(img_paths,transform):
    ten_img_tensor = []
    for path in img_paths:
        img = Image.open(path).convert('RGB')
        if transform is not None:
            img = transform(img)
        ten_img_tensor.append(img)
        
    return torch.cat(ten_img_tensor)
        

def default_audio_loader(path):
	return np.loadtxt(path,delimiter=',')

class VisualAudio(data.Dataset):
    def __init__(self,split,img_paths,audio_paths, movie_names,labels,transform=None,
                 img_loader=default_img_loader,audio_loader=default_audio_loader):
        self.split = split 
        self.img_paths = img_paths
        self.audio_paths = audio_paths
        self.movie_names = movie_names
        self.labels = labels
        self.transform = transform
        self.img_loader=img_loader
        self.audio_loader= audio_loader
        
    def __getitem__(self, index):
        img_paths, audio_paths,target = self.img_paths[index], \
                                        self.audio_paths[index], self.labels[index]
        ten_img_tensor = self.img_loader(img_paths,self.transform)
        ten_audio = self.audio_loader(audio_paths)
        #return 30x224x224 , 10x68, 10 x 5
        
        assert(ten_img_tensor.size() == (30,256,256))
        
        return ten_img_tensor, ten_audio[:10,:], target

    def __len__(self):
        return len(self.img_paths)

train_dataset_dir = '/home/noa_glaser/dataBig/train-frames/train'
val_dataset_dir = '/home/noa_glaser/dataBig/train-frames/val'
audio_dataset_dir = '/home/noa_glaser/dataBig/train-audio'
label_dataset_dir = '/home/noa_glaser/dataBig/'
exp_name = 'avepool_dropout_L2loss_ResNet34_LSTM_experiment' # roughly 3K videos
num_classes = 5 
num_partition = 10
batch_size = 8


data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(256),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


train_img_paths,train_audio_paths, train_labels, train_movienames \
        = get_img_audio_label(train_dataset_dir,audio_dataset_dir,label_dataset_dir) 

    
dsets = {}
dsets['train'] = VisualAudio('train',train_img_paths,train_audio_paths,\
                    train_movienames ,train_labels,transform=data_transforms['train'] )
dset_loaders = {'train': torch.utils.data.DataLoader(dsets['train'], batch_size=batch_size,
                        shuffle=True, num_workers=1)}

def returnSampleDataBatch():  
    input_image,input_audio, target = next(iter(dset_loaders['train']))
    # train_unflattened_sample = train_imgsamples.view(-1,3,256,256)
    return input_image,input_audio, target 
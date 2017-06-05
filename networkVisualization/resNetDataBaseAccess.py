'''
Code copied from resnet34-only script
'''

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

train_dataset_dir = '/home/noa_glaser/dataBig/train-frames/train'
val_dataset_dir = '/home/noa_glaser/dataBig/train-frames/val'
audio_dataset_dir = '/home/noa_glaser/dataBig/train-audio'
label_dataset_dir = '/home/noa_glaser/dataBig/'
exp_name = 'ResNet34_ONLY_experiment_L2_dropout' # roughly 3K videos
num_classes = 5 
num_partition = 10
batch_size = 32

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),
        transforms.RandomCrop(256),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


def _get_imgname_and_moviename_and_labels(dataset_dir):
    """Returns a list of 
filenames and inferred class names.
    Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.
    Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
    """
   
    print("processing dataset: "+ dataset_dir)
    img_names = [] 
    movies= []
    labels = []

    annotaion_filename  = label_dataset_dir + 'annotation_training.pkl'
    
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
                    label_dicts['openness'][raw_moviename]
                    #label_dicts['interview'][raw_moviename]
                    ]
        
        movie_path = os.path.join(dataset_dir, movie)
        for img in  os.listdir(movie_path):
            
            path = os.path.join(movie_path, img)
            if path == os.path.join(dataset_dir,'bt-ev53zZWE.004_5fps/frame51.jpg'): # this is a bad 
                continue
            img_names.append(path)
            movies.append(movie)
            labels.append(big_five)
        

    return np.array(img_names), np.array(labels), np.array(movies)

def default_loader(path):
	return Image.open(path).convert('RGB')

class Visual_ONLY(data.Dataset):
    def __init__(self,root,split,imgs,movies,labels,transform=None, target_transform=None,
                 loader=default_loader):
        
        #classes, class_to_idx = find_classes(root)
        #imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.split = split
        self.root = root
        self.imgs = imgs
        self.movies = movies
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.mean = np.array(self.loader(self.imgs[0]))

    def __getitem__(self, index):
        path, target = self.imgs[index], self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# dset_sizes = {x: len(dsets[x]) for x in ['train']}
# get the data file names
training_filenames, train_labels, train_movies = _get_imgname_and_moviename_and_labels( train_dataset_dir)  
validation_filenames, val_labels, val_movies = _get_imgname_and_moviename_and_labels(val_dataset_dir)  
# This is the function the visualizer will be using 
# initialize the dataset , modifying to get rid of the validation images, don't need those 
# TODO : wrap in some initializer later 
dsets = {}
dsets['train'] = Visual_ONLY(train_dataset_dir,'train',training_filenames, \
                             train_movies,train_labels,transform=data_transforms['train'] )
dset_loaders = {'train': torch.utils.data.DataLoader(dsets['train'], batch_size=32,
                                               shuffle=True, num_workers=4)}

'''
The function that we will be using
'''
def returnSampleDataBatch():  
    train_imgsamples,train_labelsample = next(iter(dset_loaders['train']))
    # train_unflattened_sample = train_imgsamples.view(-1,3,256,256)
    return train_imgsamples,train_labelsample 
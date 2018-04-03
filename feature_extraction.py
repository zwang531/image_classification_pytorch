
# coding: utf-8

# # feature extraction

# In[1]:


import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import PIL

from __future__ import print_function, division

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import os
import copy
#avoid urlopen error certificate verify failed
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

class Caltech256(Dataset):
    def __init__(self, root_dir, transform=None, train = True):
        self.images_per_class = 32 if train else 8
        self.start_image = (~train) * 32
        self.end_image = self.start_image + self.images_per_class
        self.root_dir = root_dir
        self.transform = transform
        self.cats = os.listdir(root_dir)
        self.files = {}
        self.train = train
        for cat in self.cats:
            if "clutter" in cat:
                continue

            currdir = os.path.join(root_dir, cat)
            images = os.listdir(currdir)
            images = list(filter(lambda s: s.endswith("jpg"), images))
            assert self.images_per_class <= len(images), "Not enough images in class {c}".format(c = currdir)
                
            for i in range(self.start_image, self.end_image):
                self.files[os.path.join(currdir, images[i])] = int("".join(images[i][0:3]))
                 
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_name, label = list(self.files.items())[idx]
        image = PIL.Image.open(img_name).convert("RGB") # A few images are grayscale
        label = torch.Tensor([label])
        

        if self.transform:
            image = self.transform(image)
        sample = (image, label)
        return sample
    

example_transform = transforms.Compose(
    [
        transforms.Scale((224,224)),
        transforms.ToTensor(),
    ]
)

#load caltech256
caltech256_train = Caltech256("/datasets/Caltech256/256_ObjectCategories/", example_transform, train=True)
caltech256_test = Caltech256("/datasets/Caltech256/256_ObjectCategories/", example_transform, train=False)

# -*- coding: utf-8 -*-


# License: BSD
# Author: Sasank Chilamkurthy

######################################################################


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    trainacc=np.zeros((num_epochs,1))
    testacc=np.zeros((num_epochs,1))
    trainloss=np.zeros((num_epochs,1))
    testloss=np.zeros((num_epochs,1))

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            if phase=='train':
                datas=train_data
                batchsize=32
            else:
                datas=test_data
                batchsize=8
            # calculate datasize    
            dataset_sizes=len(datas)*batchsize
            for data in datas:
                # get the inputs
                inputs, labels = data
                #shift from 1-256 to 0-255
                labels = labels-1
            
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = torch.squeeze(Variable(labels.cuda()).long())
                else:
                    inputs, labels = Variable(inputs), torch.squeeze(Variable(labels).long())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
               
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes
            epoch_acc = running_corrects / dataset_sizes
            if phase == 'train':
                trainloss[epoch]=epoch_loss
                trainacc[epoch]=epoch_acc
            else:
                testloss[epoch]=epoch_loss
                testacc[epoch]=epoch_acc
                    
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,trainloss,trainacc,testloss,testacc



######################################################################
# ConvNet as fixed feature extractor

model_conv = torchvision.models.vgg16(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

#use gpu or not
use_gpu = torch.cuda.is_available()

#if use gpu 
if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()


#reload the data and use a smaller batch size
train_data = DataLoader(
    dataset = caltech256_train,
    batch_size = 32,
    shuffle = True,
    num_workers = 4
)

test_data = DataLoader(
    dataset = caltech256_test,
    batch_size = 8,
    shuffle = True,
    num_workers = 4
)

#extract features
featuremod = list(model_conv.features.children())


# In[2]:


#get a sample image from dataloader
for data in train_data:
    inputs, labels = data
    if use_gpu:
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)
    sampleimg=inputs[0:1,:,:,:]
    break
#3 conv blocks
conv3mod = featuremod[0:17]
model_3conv=copy.deepcopy(model_conv)
model_3conv.features=torch.nn.Sequential(*conv3mod).cuda()
mod = list([nn.Linear(256*28*28, 1024),nn.Linear(1024, 256)])
new_classifier = torch.nn.Sequential(*mod).cuda()
model_3conv.classifier = new_classifier

model_3conv(sampleimg)


# In[3]:


#4 conv blocks
conv4mod = featuremod[0:24]
model_4conv=copy.deepcopy(model_conv)
model_4conv.features=torch.nn.Sequential(*conv4mod).cuda()
mod = list([nn.Linear(512*14*14, 1024),nn.Linear(1024, 256)])
new_classifier = torch.nn.Sequential(*mod).cuda()
model_4conv.classifier = new_classifier
model_4conv(sampleimg)


# In[4]:


#3 conv blocks
conv3mod = featuremod[0:17]
model_3conv=copy.deepcopy(model_conv)
model_3conv.features=torch.nn.Sequential(*conv3mod).cuda()
mod3 = list([nn.Linear(256*28*28, 1024),nn.Linear(1024, 256)])
new_classifier = torch.nn.Sequential(*mod3).cuda()
model_3conv.classifier = new_classifier


# feature extraction only parameters of classifier are being optimized 
optimizer_3conv = optim.SGD(model_3conv.classifier.parameters(), lr=0.001, momentum=0.9,weight_decay=0.1)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_3scheduler = lr_scheduler.StepLR(optimizer_3conv, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate

num_epochs=30
model_3conv,trainloss_3,trainacc_3,testloss_3,testacc_3= train_model(model_3conv, criterion, optimizer_3conv,exp_lr_3scheduler, num_epochs)


# In[5]:


#4 conv blocks
conv4mod = featuremod[0:24]
model_4conv=copy.deepcopy(model_conv)
model_4conv.features=torch.nn.Sequential(*conv4mod).cuda()
mod4 = list([nn.Linear(512*14*14, 1024),nn.Linear(1024, 256)])
new_classifier = torch.nn.Sequential(*mod4).cuda()
model_4conv.classifier = new_classifier

optimizer_4conv = optim.SGD(model_4conv.classifier.parameters(), lr=0.001, momentum=0.9,weight_decay=0.1)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_4scheduler = lr_scheduler.StepLR(optimizer_4conv, step_size=7, gamma=0.1)

num_epochs=10
model_4conv,trainloss_4,trainacc_4,testloss_4,testacc_4 = train_model(model_4conv, criterion, optimizer_4conv,exp_lr_4scheduler, num_epochs)



# coding: utf-8

# # transfer learning

# In[ ]:


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

# -*- coding: utf-8 -*-


# License: BSD
# Author: Sasank Chilamkurthy

######################################################################


# In[ ]:


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



#plot firstlayer weights
def plot_weights(tensor, num_cols=8, name='noname.png'):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        tensor[i]=(tensor[i]-np.min(tensor[i]))/(np.max(tensor[i])-np.min(tensor[i]))
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    plt.savefig(name, dpi=600)

def plot_activation(tensor, num_cols=8, name='noname.png'):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    num_kernels = tensor.shape[1]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    #plt.title('activation')
    for i in range(tensor.shape[1]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[0,i,:,:])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    #plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    plt.savefig(name, dpi = 600)


######################################################################
# ConvNet as fixed feature extractor

model_conv = torchvision.models.vgg16(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False

#use gpu or not
use_gpu = torch.cuda.is_available()

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs=model_conv.classifier[-1].in_features
mod = list(model_conv.classifier.children())
mod.pop()
mod.append(nn.Linear(num_ftrs, 257))
#mod.append(nn.Linear(num_ftrs, 256))
new_classifier = torch.nn.Sequential(*mod)
model_conv.classifier = new_classifier

#if use gpu 
if use_gpu:
    model_conv = model_conv.cuda()

criterion = nn.CrossEntropyLoss()

#extract all layers till the final conv layers
mod = list(model_conv.features.children())
mod = mod[:-2]
model_allconv=torch.nn.Sequential(*mod)


# In[ ]:


model_conv.classifier[-1].parameters


# In[ ]:


# transfer learning only parameters of final layer are being optimized 
optimizer_conv = optim.SGD(model_conv.classifier[-1].parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

######################################################################
# Train and evaluate

num_epochs=10
model_conv,trainloss,trainacc,testloss,testacc = train_model(model_conv, criterion, optimizer_conv,exp_lr_scheduler, num_epochs)

######################################################################



#plot acc and loss vs epochs
x=np.arange(num_epochs)
plt.figure()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(x,trainacc,label='trainacc')
plt.plot(x,testacc,label='testacc')
plt.title('accuracy-epoch')  
plt.legend()
plt.figure()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(x,trainloss,label='trainloss')
plt.plot(x,testloss,label='testloss')
plt.title('loss-epoch')
plt.legend()
plt.show()


# In[ ]:


#plot acc and loss vs epochs
x=np.arange(num_epochs)
plt.figure()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(x,trainacc,label='trainacc')
plt.plot(x,testacc,label='testacc')
plt.title('accuracy-epoch')  
plt.legend()
plt.savefig('2a.png',dpi = 600)
plt.figure()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(x,trainloss,label='trainloss')
plt.plot(x,testloss,label='testloss')
plt.title('loss-epoch')
plt.legend()
plt.savefig('2b.png',dpi = 600)
plt.show()


# # weights visualization

# In[ ]:


#get a sample image from dataloader
torch.manual_seed(43)
count = 1
for data in train_data:
    inputs, labels = data
    if use_gpu:
        inputs = Variable(inputs.cuda())
    else:
        inputs = Variable(inputs)
    sampleimg=inputs[0:1,:,:,:]
    break
plt.imshow(sampleimg[0,:,:,:].permute(1,2,0).data.cpu().numpy())
plt.title('original image')

plt.savefig('4a.png',dpi=600)

firstlayer=model_conv.features[0](sampleimg)
plot_activation(firstlayer.data.cpu().numpy(), num_cols=8)



finallayer=model_allconv(sampleimg)

plot_activation(finallayer.data.cpu().numpy(), num_cols=24)


#print first layer weights
tensor = model_conv.features[0].weight.data.cpu().numpy()
plot_weights(tensor)




# In[ ]:


firstlayer[0,0]


# ### Analyze weights

# In[34]:


tensor = firstlayer.data.cpu().numpy()
i = 12
tensor2 = model_conv.features[0].weight.data.cpu().numpy()

ts=(tensor2[i]-np.min(tensor2[i]))/(np.max(tensor2[i])-np.min(tensor2[i]))

fig = plt.figure(figsize=(2,1))
#plt.title('activation')

ax1 = plt.subplot(121)
ax1.imshow(tensor[0,i,:,:])
ax1.axis('off')
# ax1.set_xticklabels([])
# ax1.set_yticklabels([])

ax2 = plt.subplot(122)
ax2.imshow(ts)
ax2.axis('off')
# ax2.set_xticklabels([])
# ax2.set_yticklabels([])


#plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('5a',dpi=600)
plt.show()


# In[59]:


tensor = firstlayer.data.cpu().numpy()
i = 6
tensor2 = model_conv.features[0].weight.data.cpu().numpy()[:,:,:,1]

ts=(tensor2[i]-np.min(tensor2[i]))/(np.max(tensor2[i])-np.min(tensor2[i]))

fig = plt.figure(figsize=(2,1))
#plt.title('activation')

ax1 = plt.subplot(121)
ax1.imshow(tensor[0,i,:,:])
ax1.axis('off')
# ax1.set_xticklabels([])
# ax1.set_yticklabels([])

ax2 = plt.subplot(122)
ax2.imshow(ts)
ax2.axis('off')
# ax2.set_xticklabels([])
# ax2.set_yticklabels([])


#plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig('5b',dpi=600)
plt.show()


# In[42]:


model_conv.features[0].weight.data.cpu().numpy()[:,:,:,0]


# In[53]:


ts.shape


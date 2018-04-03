
# coding: utf-8

# In[1]:


from torchvision import transforms, datasets as ds
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
                             ])
trainset = ds.CIFAR10(root='/datasets/CIFAR-10', train=True,download=False, transform=transform) 
num_train = len(trainset)
indices = range(num_train)
print(len(indices))
np.random.shuffle(indices)

train_idx=indices[5000:]
hold_idx=indices[:5000]

hold_set=[]
train_set=[]
for i in range(len(hold_idx)):
    hold_set.append(trainset[hold_idx[i]])
for i in range(len(train_idx)):
    train_set.append(trainset[train_idx[i]])
trainloader = DataLoader(dataset=train_set,
                         batch_size=100,
                         shuffle=True,
                         num_workers=4)
holdloader = DataLoader(dataset=hold_set,
                         batch_size=100,
                         shuffle=False,
                         num_workers=4)

testset = ds.CIFAR10(root='datasets/CIFAR-10',train=False,download=False,transform=transform)
print(len(testset))
testloader = DataLoader(dataset=testset, 
                        batch_size=100,  
                        shuffle=False, 
                        num_workers=4)  
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.constant(self.conv1.bias, 0.1)
        self.conv1_bn = nn.BatchNorm2d(64)
        
        self.pool  = nn.MaxPool2d(2,2)
        self.avgpool=nn.AvgPool2d(2,2)
        
        self.conv2 = nn.Conv2d(64, 256, 3)
        nn.init.xavier_uniform(self.conv2.weight)
        nn.init.constant(self.conv2.bias, 0.1)
        self.conv2_bn = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256,512,3)
        nn.init.xavier_uniform(self.conv3.weight)
        nn.init.constant(self.conv3.bias, 0.1)
        self.conv3_bn = nn.BatchNorm2d(512)
        
        self.conv4 = nn.Conv2d(512,1024,3)
        nn.init.xavier_uniform(self.conv4.weight)
        nn.init.constant(self.conv4.bias, 0.1)
        self.conv4_bn = nn.BatchNorm2d(1024)
        
        self.conv5 = nn.Conv2d(1024,2048,3)
        nn.init.xavier_uniform(self.conv5.weight)
        nn.init.constant(self.conv5.bias, 0.1)
        self.conv5_bn = nn.BatchNorm2d(2048)
        
        
        self.fc1 = nn.Linear(2048*4*4, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2   = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.fc3   = nn.Linear(84, 10)
        #self.fc4   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.avgpool(F.relu(self.conv5(x)))
        #x = self.avgpool(F.relu(self.conv2_bn(self.conv2(x))))
        #x = self.avgpool(F.relu(self.conv3_bn(self.conv3(x))))
        #x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 2048*4*4)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)
        return x


net=Net()
net.cuda()

import torch.optim as optim            
criterion = nn.CrossEntropyLoss()    
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 
optimizer=optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))
criterion.cuda()


from torch.autograd import Variable
epoch=1
train_loss=[]
hold_loss=[]
k=0
holdrun_loss=0.0
trainaccu=[]
holdaccu=[]
testaccu=[]
while(1): # loop over the dataset multiple times
    running_loss = 0.0
    running_accu=0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        
        # wrap them in Variable
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()        
        optimizer.step()
        pred = torch.max(outputs, 1)[1]
        accu_train = sum(pred == labels).data[0]
        # print statistics
        running_accu += accu_train
        running_loss += loss.data[0]
        print(running_accu)
    train_loss.append(running_loss/45000)
    trainaccu.append(running_accu/i/100.0)
    print(running_accu/i/100.0)
    
    accu_test=0.0
    for i, data in enumerate(testloader):
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        
        outputs = net(inputs)
        pred = torch.max(outputs, 1)[1]
        accu = sum(pred == labels).data[0]
        accu_test += accu
    testaccu.append(accu_test/i/100)
    print(accu_test/i/100)
    
    accu_hold=0.0
    running_loss=0.0
    for i, data in enumerate(holdloader):
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.data[0]
        pred = torch.max(outputs, 1)[1]
        accu = sum(pred == labels).data[0]
        accu_hold +=accu
    running_loss=running_loss/i/100
    holdaccu.append(accu_hold/i/100)
    print(accu_hold/i/100)
    if(running_loss>holdrun_loss):
        k=k+1
    else:
        k=0
    holdrun_loss=running_loss
    hold_loss.append(holdrun_loss/5000)
    if(k>2):
        break
    epoch=epoch+1

import matplotlib.pyplot as plt
plt.figure
l1,=plt.plot(range(epoch),trainaccu,label='train')
l2,=plt.plot(range(epoch),holdaccu,label='holdout')
l3,=plt.plot(range(epoch),testaccu,label='test')
plt.legend(handles = [l1, l2, l3,], labels = ['train', 'holdout','test'], loc = 'best')
plt.savefig("xavier.png",dpi=120)
plt.show()


# In[2]:


import torch
print('CUDA available :',torch.cuda.is_available())


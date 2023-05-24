#!/usr/bin/env python
# coding: utf-8

# # Autoencoder In String Theory

# A deep autoencoder is a feedforward NN, mainly used to dimensionally
# reduce or compress some complex high dimensional data. It is built by two components: the 
# encoder and the decoder. The purpose of the encoder is to identify and reduce the 
# redundancies (and noise) of input data defined by a large number of parameters, lowering 
# step-wise in various (hidden) layers the dimensionality of the parameter space. If the encoder is 
# deep enough, i.e.\ if it has a large number of layers, and the data is adequate, its 
# capability to encode the input data into a small number of parameters tends to improve. 
# The last layer of the encoder is known as the (central) *latent layer* or latent-space 
# representation, and it contains a "compressed code" of the input data, represented in 
# a small number of parameters. The decoder operates inversely to the encoder, reconstructing 
# the data from the latent layer back to its original higher dimensional form. One can define the 
# accuracy of an autoencoder as the level of likeness between the output resulting
# from the decoder and the corresponding original information in the input layer.
# 
# Given some input data, one must choose an autoencoder *configuration* that
# maximizes the accuracy of the algorithm. The properties that describe an autoencoder
# configuration are:
# 
# * **Topology**: overall structure that defines the way the neurons of the NN
#   are connected among different layers. The topology can be symmetric or asymmetric with 
#   respect to the latent or bottleneck layer, fully or partly connected, and can include convolutional 
#   layers or other types of substructures. We avoid convolutional or other complex layers
#   for simplicity.
# 
# * **Architecture**: number of layers and number of neurons per layer (layer size). In the case of
#   an autoencoder, it includes the sizes of all hidden layers of the encoder and decoder,
#   the input and output layers, as well as the size of the latent layer.
# 
# * **Initial weight distribution**: the values of the trainable parameters or weights that 
#   characterize the neurons must be initialized at random values using a method that 
#   may be useful to arrive at the best accuracy; it is customary to take a Gaussian or 
#   uniform distribution, but other options (such as Xavier or He initializations) 
#   are possible.
# 
# * **Activation function**: together with a bias, it defines the output of a neuron 
#   given some input information; it adds some non-linearity to the learning process in
#   order to improve it. Some examples that we shall use in this work include Leaky-ReLU, 
#   Softplus, ELU, CELU, ReLU, SELU, Hardtanh, Softmax and LogSigmoid 
#   for details of activation functions).
#   In principle, every layer can have a different activation function, but we apply homogeneously 
#   the same activation function to all layers for simplicity.
# 
# * **Loss function**: evaluation during the training stage that determines the magnitude of
#   the inaccuracy that the NN has achieved before updating the weights of the
#   network.
#   Some examples of loss functions used in this paper are Cross Entropy (CE), SmoothL1, 
#   MSE, Huber, BCEWL, L1 and Hinge Embedding.
# 
# * **Optimizer**: optimization algorithm used to minimize the loss; some examples applied in this
#   work are Adam, AdamW, Adamax, RMSProp, Adagrad, Adadelta, SGD and ASGD.
# 
# * **Number of epochs**: number of times that the algorithm is run to improve the 
#   learning skills of the algorithm, trying to minimize the error.
# 
# * **Batch size**: for each epoch, it is the number of samples in which the training input
#   data is split in order to have several training subsets. Typically, large batch sizes
#   lead to better statistical characterizations; however, one must choose the size that
#   best helps to maximize the accuracy of the algorithm.
# 
# * **Shuffling**: to optimize the learning process, whether or not the elements contained 
#   in each batch per epoch are randomly shuffled. 
# 
# * **Dropout**: if applied, it defines the number of dropout layers and the fraction of neurons that are 
#   randomly dropped out; this is typically used with the goal of reducing overfitting.
# 

# ## Dependences

# In[16]:


import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import data as data
import routines as routines
import utils as utils


# ## Setup

# ### Neural Network

# The NN implemented is fully connected and symmetrical. The dimensions array has the ordered number 
# of neurons per layer. All the layers are set with *leaky_relu* as the activation function. The Autoencoder is
# segmented in two sections because the encoding and decoding elements are required individually in further
# sections.

# In[17]:


class Net(nn.Module):

    def __init__(self, dimensions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(dimensions[0], dimensions[1])
        self.fc2 = nn.Linear(dimensions[1], dimensions[2])
        self.fc3 = nn.Linear(dimensions[2], dimensions[3])
        self.fc4 = nn.Linear(dimensions[3], dimensions[4])
        self.fc5 = nn.Linear(dimensions[4], dimensions[5])

        self.fc6 = nn.Linear(dimensions[5], dimensions[6])
        self.fc7 = nn.Linear(dimensions[6], dimensions[7])
        self.fc8 = nn.Linear(dimensions[7], dimensions[8])
        self.fc9 = nn.Linear(dimensions[8], dimensions[9])
        self.fc10 = nn.Linear(dimensions[9], dimensions[10])

    def encode(self, Layer):
        Layer = F.leaky_relu(self.fc1(Layer))
        Layer = F.leaky_relu(self.fc2(Layer))
        Layer = F.leaky_relu(self.fc3(Layer))
        Layer = F.leaky_relu(self.fc4(Layer))
        Layer = F.leaky_relu(self.fc5(Layer))
        return Layer

    def decode(self, Layer):
        Layer = F.leaky_relu(self.fc6(Layer))
        Layer = F.leaky_relu(self.fc7(Layer))
        Layer = F.leaky_relu(self.fc8(Layer))
        Layer = F.leaky_relu(self.fc9(Layer))
        Layer = F.leaky_relu(self.fc10(Layer))
        return Layer

    def forward(self, Layer):
        return self.decode(self.encode(Layer))


# ### Parameters

# In this section several input variables can be set.

# In[18]:


# Name input dataset
datasetname = './data/600K_Z8-Z12.csv'

# ohe and features lenghts
lenghts_data = data.lenghts_features(datasetname) 
l_ohe = sum(lenghts_data)

# latent space dimension
latent = 3

# Dimensions of layers 
dimensions = [l_ohe, 2*l_ohe, 200, 26, 13, latent, 13, 26, 200, 2*l_ohe, l_ohe]

# Number of epochs for training
epochs = 1010
# Save every num of epochs
save_each_epoch = 100


# Parameters for dataset train
# Ration of the dataset to be trained
train_set = 0.6
# Seed to be set for the reproducibility of the model
seed = 1
# A high value for the batchsize is recommended, although it may be changed to fit the PC capabilities
batchsize = 32
# Number of processed used on CPU to mount on GPU
workers = 8

# label train log
label = "1010_e_leaky"


# ### Instantiation

# In[19]:


# Parameters parsing
parameters = {"datasetname":datasetname,
             "seed":1,
             "epochs": epochs,
             "train_set": train_set,
             "batchsize": batchsize,
             "workers": workers,
             "label":label,
             "lenghts_data":lenghts_data,
             "latent": latent,
             "save_each_epoch":save_each_epoch}

# Model instantiation
model = Net(dimensions)
# Definition of loss function
criterion = nn.CrossEntropyLoss()
# Definition of optimizer & learning rate
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Scheduler set
scheduler = lr_scheduler.StepLR(optimizer, step_size= 4000, gamma=0.1)


# ## Training

# In[ ]:


routines.train(model, criterion, optimizer, scheduler, **parameters)



# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Enrique_Escalante-Notario
# Instituto de Fisica, UNAM
# email: <enriquescalante@gmail.com>
# Distributed under terms of the GPLv3 license.
# data.py
# --------------------------------------------------------

import numpy as np
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import loss as ls
import data as dt



# save models 
def save(label, best_model, best_optimizer, best_epoch):
    path = "./savedModels/model-"+label+".pt"
    torch.save({'model_dict': best_model,
                'optim_dict': best_optimizer,
                'epoch': best_epoch}, path)
    

def train(model_untrained, criterion, optimizer, scheduler, **kwargs):
    
    epochs = kwargs["epochs"]
    
    # Load and ohe dataset 
    print('Preparing data',end='\n')
    dataset = dt.CustomDataset(kwargs['datasetname'])
        
    # Split dataset for category
    train_length=int(kwargs['train_set']*len(dataset))
    val_length=len(dataset)-train_length
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, (train_length, val_length))
    
    
    # Dataloder train phase
    dataloader_train=torch.utils.data.DataLoader(train_dataset,
            batch_size=kwargs['batchsize'], shuffle=True, num_workers=kwargs['workers'])
    dataloader_val=torch.utils.data.DataLoader(val_dataset,
            batch_size=kwargs['batchsize'], shuffle=True, num_workers=kwargs['workers'])
       
    # Load data in GPU
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(kwargs['seed'])
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model_untrained.to(device)
    
    if use_cuda:
        print('Using GPU')

    # Open log train
    loggertrain = open("./reports/log_loss_accuracy-"+kwargs['label'], "w")
    
    print('Start training',end='\n')
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_epoch = 0
    best_optimizer = optimizer.state_dict()

    for epoch in range(1,epochs+1):
        print('-' * 50)
        print('Epoch {}/{}'.format(epoch, epochs))
        # Each epoch has a training and validation phase
        for phase in ['train','valid']:
            if phase == 'train':
                if epoch != 1:
                   scheduler.step()
                model.train(True)
                dataloader = dataloader_train
            else:
                model.train(False)
                dataloader = dataloader_val

            running_loss = 0.0
            running_corrects = 0

            for (x, y) in dataloader:   
                data = x.to(device)
              
                output = model(data)
                train_loss = ls.CustomLossFunction(data, output, criterion, kwargs['lenghts_data'])
                optimizer.zero_grad()
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    train_loss.backward()
                    optimizer.step()

                # calculate accuracy
                ini = 0
                for ind in kwargs['lenghts_data']:
                    curr_target = output[:,ini:ini + ind]
                    curr_data = data[:,ini:ini + ind]
                    max_ind_tar = torch.max(curr_target,1)[1]
                    max_ind_dat = torch.max(curr_data,1)[1]
                    running_corrects += torch.sum(max_ind_tar == max_ind_dat)
                    ini = ind + ini 
                    
                # statistics 
                running_loss += train_loss.item()
                        
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / len(dataloader.dataset)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == "train":
                print('{}, {:.4f}, {:.4f}'.format(epoch, epoch_loss, epoch_acc), end=', ',file=loggertrain)
            if phase == "valid":
                print('{:.4f}, {:.4f}'.format(epoch_loss, epoch_acc), file=loggertrain)

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                best_optimizer = optimizer.state_dict()
                best_epoch = epoch
            if phase == 'valid' and epoch % kwargs['save_each_epoch'] == 0:
                save(str(epoch)+"epoch_"+kwargs['label'], model.state_dict(), optimizer.state_dict(), epoch)
                

    # Save the best model
    save("best-"+kwargs['label'], best_model_wts, best_optimizer, best_epoch)
   
    loggertrain.close()
    
    
# Latent space model mapping
# input: model_trained, dataloander_train, dim_dim, output_file
def process(model, dataloader, latent, output_file):
    model.eval()
    processed = []
    count = 0

    for (x, y) in dataloader:
        processed.append(model.encode(x).detach().numpy())
        count += len(x)

    out = np.empty([count, latent])
    index = 0
    for batch in processed:
        out[index:index + len(batch)] = batch
        index += len(batch)
    np.savetxt(output_file, out)  
  
    
# Accuracy by feature
def success(model, dataloader, label_states, lenghts_data, epoch_trained):
    model.eval()
    path = "./success/success_"+label_states
    loggerreconstruction = open(path, "w")

    for (x,y) in dataloader:
        output = model(x)
        evalIn = np.copy(x.numpy())
        evalOut = np.copy(output.detach().numpy())
        for j in range(0, len(evalIn)):
            correct_counter = 0
            partial_counter = np.zeros(len(lenghts_data),dtype=np.int8)
            ini = 0
            k = 0
            for ind in lenghts_data:
                curr_target = evalIn[j][ini:ini + ind]
                curr_data = evalOut[j][ini:ini + ind]
                target_index = curr_target.tolist().index(np.max(curr_target))
                data_index = curr_data.tolist().index(np.max(curr_data))
                if target_index == data_index:
                    partial_counter[k] = partial_counter[k] + 1
                    correct_counter += 1
                ini = ind + ini
                k += 1
            print('{},{}'.format(correct_counter,",".join(map(str, partial_counter.tolist()))), file=loggerreconstruction)
    loggerreconstruction.close()

    
# construction space latent

def reconstruction(model, label_states, **kwargs):

    # Load and ohe dataset 
    print('Preparing data',end='\n')
    dataset = dt.CustomDataset(kwargs['datasetname'])
    # Split dataset for geometry                                           # (Must be improve)
    indices_z8 = [idx for idx, item in enumerate(dataset) if item[1] == 0]
    dataset_z8 = Subset(dataset, indices_z8)
    indices_z12 = [idx for idx, item in enumerate(dataset) if item[1] == 1]
    dataset_z12 = Subset(dataset, indices_z12)
    
    # Dataloders
    dataloader_z8=torch.utils.data.DataLoader(dataset_z8,
            batch_size=kwargs['batchsize'], shuffle=True, num_workers=kwargs['workers'])
    dataloader_z12=torch.utils.data.DataLoader(dataset_z12,
            batch_size=kwargs['batchsize'], shuffle=True, num_workers=kwargs['workers'])
    dataloader=torch.utils.data.DataLoader(dataset,
            batch_size=kwargs['batchsize'], shuffle=True, num_workers=kwargs['workers'])
  
    # Load states and instantion model 
    print('Load model')
    path = "./savedModels/"+label_states+".pt"
    dict_model = torch.load(path, map_location=torch.device('cpu'))
    model_trained = model.to('cpu')   
    model_trained.load_state_dict(dict_model['model_dict'])
    epoch_trained = dict_model['epoch']

    # latent construction
    print('Construction latent space')
    output_z8 = "./latentSpaces/latent_z8_"+label_states
    output_z12 = "./latentSpaces/latent_z12_"+label_states
    process(model_trained, dataloader_z8, kwargs['latent'], output_z8)
    process(model_trained, dataloader_z12,kwargs['latent'], output_z12)
    
    # success construction
    print('Construction of success each feature')
    success(model_trained, dataloader, label_states, kwargs["lenghts_data"], epoch_trained)
    
    
    
# space latente new models
def reconstruction_other_models(model, label_states, datasetname_other,label_other, **kwargs):

    # Load and ohe dataset 
    print('Preparing data',end='\n')
    dataset_other = dt.CustomDatasetExtra(kwargs['datasetname'], datasetname_other)
    dataloader_other = torch.utils.data.DataLoader(dataset_other,
        batch_size=kwargs['batchsize'], shuffle=True, num_workers=kwargs['workers'])
    
    # Load states and instantion model 
    print('Load model')
    path = "./savedModels/"+label_states+".pt"
    dict_model = torch.load(path, map_location=torch.device('cpu'))
    model_trained = model.to('cpu')   
    model_trained.load_state_dict(dict_model['model_dict'])
    epoch_trained = dict_model['epoch']
        
    output = "./latentSpaces/latent_"+label_other+"-"+label_states
    process(model_trained, dataloader_other, kwargs['latent'], output)
    
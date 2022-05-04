#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:56:49 2022

@author: hellraiser
"""

import time
import numpy as np
import torch

class RunModel:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, n_epochs=3):
        super(RunModel, self).__init__()
        

        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print('CUDA is not available.  Training on CPU ...')
        else:
            print('CUDA is available!  Training on GPU ...')
    
        since = time.time()
        valid_loss_min = np.Inf # track change in validation loss
        # Early stopping
        last_loss = 100
        patience = 2
        triggertimes = 0
        # Accuracy
        eval_losses=[]
        eval_accu=[]
    
        for epoch in range(1, n_epochs+1):
    
            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            correct=0.0
            total=0.0
    
            ###################
            # train the model #
            ###################
            model.train()
            for data, target in train_loader:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item()*data.size(0)
    
            ######################    
            # validate the model #
            ######################
            model.eval()
            for data, target in valid_loader:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss 
                valid_loss += loss.item()*data.size(0)
                # scheduler 
                scheduler.step()
    
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
    
            # calculate average losses
            train_loss = train_loss/len(train_loader.sampler)
            valid_loss = valid_loss/len(valid_loader.sampler)
    
            accu=100.*correct/total
    
            eval_accu.append(accu)
            eval_losses.append(train_loss)
            print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))
    
            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))
    
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(model.state_dict(), str(model).split('(')[0]+'.pt')
                valid_loss_min = valid_loss
    
            # Early stopping
            if valid_loss > valid_loss_min:
                triggertimes += 1
                print('Trigger Times:', triggertimes)
    
                if triggertimes >= patience:
                    print('Early stopping!\nStart to test process.')
    
            else:
                print('trigger times: 0')
                triggertimes = 0
    
        print('time_elapsed: ', time.time() - since)
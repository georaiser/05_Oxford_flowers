#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 18:56:49 2022

@author: hellraiser
"""

import time
import numpy as np
import torch
from tqdm import tqdm

class RunModel:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, n_epochs):
        super(RunModel, self).__init__()
        
        self.model=model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.n_epochs = n_epochs        
        
    def run(self):
        since = time.time()
        train_on_gpu = torch.cuda.is_available()
    
        valid_loss_min = np.Inf # track change in validation loss
        # Early stopping
        patience = 5
        triggertimes = 0

        losses = {'train':[], 'val':[]}
        accuracies = {'train':[], 'val':[]}
    
        for epoch in range(1, self.n_epochs+1):
    
            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            correct=0.0
            total=0.0
    
            ###################
            # train the model #
            ###################
            self.model.train()
            
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for data, target in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")            

                    # move tensors to GPU if CUDA is available
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()
                    # clear the gradients of all optimized variables
                    self.optimizer.zero_grad()
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = self.model(data)
                    # calculate the batch loss
                    loss = self.criterion(output, target)
                    # backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # perform a single optimization step (parameter update)
                    self.optimizer.step()
                    # update training loss
                    train_loss += loss.item()*data.size(0)
                    
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                    
                    accuracy=100.*correct/total
                    tepoch.set_postfix(loss=train_loss/len(self.train_loader.sampler), accuracy=accuracy)
                    time.sleep(0.1)
                 
                # Metrics
                # calculate average losses
                train_loss = train_loss/len(self.train_loader.sampler)
                accuracy=100.*correct/total
                accuracies['train'].append(accuracy)
                losses['train'].append(train_loss)

            ######################    
            # validate the model #
            ######################
            self.model.eval()
            for data, target in self.valid_loader:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                # calculate the batch loss
                loss = self.criterion(output, target)
                # update average validation loss 
                valid_loss += loss.item()*data.size(0)
                # scheduler 
                self.scheduler.step()
    
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                

            # Metrics    
            # calculate average losses
            valid_loss = valid_loss/len(self.valid_loader.sampler)  
            accuracy=100.*correct/total
            accuracies['val'].append(accuracy)
            losses['val'].append(valid_loss)
            
            print('Epoch: %.0f | Train Loss: %.3f | Accuracy: %.3f'%(epoch, train_loss, accuracy))
            print('Epoch: %.0f | Valid Loss: %.3f | Accuracy: %.3f'%(epoch, valid_loss, accuracy))

    
            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                torch.save(self.model.state_dict(), str(self.model).split('(')[0]+'.pt')
                valid_loss_min = valid_loss
    
            # Early stopping
            if valid_loss > valid_loss_min:
                triggertimes += 1
                #print('Trigger Times:', triggertimes)
    
                if triggertimes >= patience:
                    print('Early stopping!')
    
            else:
                #print('trigger times: 0')
                triggertimes = 0
        
        
    
        print('time_elapsed: ', time.time() - since)
        
        return accuracies, losses
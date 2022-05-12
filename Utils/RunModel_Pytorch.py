#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  7 20:26:20 2022

@author: hellraiser
"""
# libraries
import numpy as np
from tqdm.auto import tqdm

# pytorch
import torch
import torch.nn.functional as F

class RunModel():
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, scheduler, model_save, n_epochs):
        super(RunModel, self).__init__()

        self.model=model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_save = model_save   
        self.n_epochs = n_epochs
        
    def train_model(self): 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        self.model.to(device)
        
        # Early stopping
        patience = 10
        triggertimes = 0
        valid_loss_min = np.Inf 
        
        history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

        for epoch in range(self.n_epochs): 
            #############
            # model.train
            #############
            self.model.train()
            # zero
            training_loss = 0.0
            sum_correct = 0 
            sum_examples = 0           
            loop = tqdm(self.train_loader, leave = False)
            for inputs, targets in loop:
                # to device
                inputs, targets = inputs.to(device), targets.to(device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                # optimizer
                self.optimizer.step()
                # partial statistics - predictions
                training_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(F.softmax(outputs, dim=1), dim=1)[1], targets).view(-1)
                sum_correct += torch.sum(correct).item()
                sum_examples += correct.shape[0]
                # loop tqdm
                loop.set_description("Epoch [{}/{}]".format(epoch+1,self.n_epochs))
                loop.set_postfix(train_loss=loss.data.item() * inputs.size(0), train_accuracy=sum_correct / sum_examples, Lr=self.optimizer.param_groups[0]['lr'])
            # total statistics    
            training_loss = training_loss/len(self.train_loader.dataset)
            training_accuracy = sum_correct / sum_examples
            
            ############
            # model.eval
            ############
            self.model.eval()
            # zero
            valid_loss = 0.0
            sum_correct = 0 
            sum_examples = 0            
            loop = tqdm(self.valid_loader, leave = False)
            
            with torch.no_grad():      
                for inputs, targets in loop:
                    # to device
                    inputs, targets = inputs.to(device), targets.to(device)
                    # forward + backward + optimize
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets) 
                    # partial statistics - predictions
                    valid_loss += loss.data.item() * inputs.size(0)                            
                    correct = torch.eq(torch.max(F.softmax(outputs, dim=1), dim=1)[1], targets).view(-1)
                    sum_correct += torch.sum(correct).item()
                    sum_examples += correct.shape[0]  
                    # loop tqdm
                    loop.set_description("Epoch [{}/{}]".format(epoch+1,self.n_epochs))
                    loop.set_postfix(val_loss=loss.data.item() * inputs.size(0), val_accuracy=sum_correct / sum_examples, Lr=self.optimizer.param_groups[0]['lr'])
                
            # total statistics 
            valid_loss = valid_loss/len(self.valid_loader.dataset)
            valid_accuracy = sum_correct / sum_examples
            
            # scheduler
            self.scheduler.step(valid_loss)
            
            ##################           
            history['train_loss'].append(training_loss)
            history['val_loss'].append(valid_loss)
            history['train_acc'].append(training_accuracy)
            history['val_acc'].append(valid_accuracy)
            
            print('Epoch: {}, Training_Loss: {:.4f}, Validation_Loss: {:.4f}, Training_accuracy = {:.4f}, Validation_accuracy = {:.4f}, lr = {}'.format(
                epoch+1, training_loss, valid_loss, training_accuracy, valid_accuracy, self.optimizer.param_groups[0]['lr']))
            
            # save model if validation loss has decreased
            if valid_loss < valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
                # save full model   ...---> partial with '.state_dict()'
                torch.save(self.model, self.model_save)
                valid_loss_min = valid_loss
            
            # Early stopping
            if valid_loss > valid_loss_min:
                triggertimes += 1
                if triggertimes >= patience:
                    print('Early stopping!')
                    print('patience times: ', patience)
                    break 
            else:
                #print('trigger times: 0')
                triggertimes = 0

        return history
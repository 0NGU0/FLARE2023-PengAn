#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.metric import dice_coeff
import logging
import os

def write_metric_values(net, output, epoch, train_metric, val_metric):
    file = open(output+'epoch-%0*d.txt'%(2,epoch),'w') 
    if net.n_classes > 1:
        logging.info('training cross-entropy: {}'.format(train_metric[-1]))
        logging.info('validation cross-entropy: {}'.format(val_metric[-1]))
        file.write('train cross-entropy = %f\n'%(train_metric[-1]))
        file.write('val cross-entropy = %f'%(val_metric[-1]))
    else:
        logging.info('training dice: {}'.format(train_metric[-1]))
        logging.info('validation dice: {}'.format(val_metric[-1]))
        file.write('train dice = %f\n'%(train_metric[-1]))
        file.write('val dice = %f'%(val_metric[-1]))
    file.close()

def launch_training(epochs, net, train_loader, val_loader, n_train, device, net_id, criterion, optimizer, output):
    ''' generic training launcher '''
    if net.n_classes > 1:
        train_cross_entropy, val_cross_entropy = [], []

    for epoch in range(epochs):
        ## net.training = True
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs, masks = batch[0], batch[1]
                imgs = imgs.to(device=device, dtype=torch.float32)

                mask_type = torch.long
                masks = masks.to(device=device, dtype=mask_type)
                preds = net(imgs)
                if net.n_classes > 1:
                    if net_id in [1]:
                        loss_sum = 0
                        for i in range(len(preds)):
                            loss1 = criterion(preds[i], torch.squeeze(masks,1))
                            loss_sum += loss1
                        loss = loss_sum/len(preds)       
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
        if net.n_classes > 1:
            train_cross_entropy.append(eval_net(net, train_loader, device))
            val_cross_entropy.append(eval_net(net, val_loader, device))
            write_metric_values(net, output, epoch, train_cross_entropy, val_cross_entropy)
            if epoch>0:
                torch.save(net.state_dict(), output+'epoch{}.pth'.format(epoch))
                logging.info(f'checkpoint {epoch + 1} saved !')
            else:
                torch.save(net.state_dict(), output+'epoch{}.pth'.format(epoch))

    if net.n_classes > 1: 
        return train_cross_entropy, val_cross_entropy
            
def dice_history(epochs, train_dices, val_dices, output):
    ''' display and save dices after training '''
    plt.plot(range(epochs), train_dices)
    plt.plot(range(epochs), val_dices)
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('dice')
    plt.xlim([0,epochs-1]) 
    plt.ylim([0,1]) 
    plt.grid()
    plt.savefig(output+'dices.png')
    plt.close()
    np.save(output+'train_dices.npy', np.array(train_dices))
    np.save(output+'val_dices.npy', np.array(val_dices))
    
def cross_entropy_history(epochs, train_cross_entropy, val_cross_entropy, output):
    ''' display and save dices after training '''
    plt.plot(range(epochs), train_cross_entropy)
    plt.plot(range(epochs), val_cross_entropy)
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('cross-entropy')
    plt.xlim([0,epochs-1]) 
    # plt.ylim([0,1]) 
    plt.grid()
    plt.savefig(output+'cross-entropy.png')
    plt.close()
    np.save(output+'train_cross_entropy.npy', np.array(train_cross_entropy))
    np.save(output+'val_dcross_entropy.npy', np.array(val_cross_entropy))
    
def eval_net(net, loader, device):
    ''' evaluation with dice coefficient '''
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)
    tot = 0
    for batch in loader:
        imgs, masks = batch[0], batch[1]
        imgs = imgs.to(device=device, dtype=torch.float32)
        masks = masks.to(device=device, dtype=mask_type)
        with torch.no_grad():
            preds = net(imgs)
        if net.n_classes > 1:
            tot += nn.functional.cross_entropy(preds,torch.squeeze(masks,1))
        else:
            preds = torch.sigmoid(preds)            
            preds = (preds > 0.5).float()
            tot += dice_coeff(preds, masks).item()
    return tot / n_val

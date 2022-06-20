import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import DFRscore
from .data import TrainDataset, gat_collate_fn

import numpy as np
import pickle
import os
import sys
import random

import time
import copy

import argparse
from datetime import datetime


MAX_STEP=None

def HingeMSELoss(y_pred, y_true):
    global MAX_STEP
    ZERO = torch.tensor(0).float().to(y_pred.device)
    NEG_MIN = torch.tensor(MAX_STEP+1).float().to(y_pred.device)
    y_true = y_true.float()
    return  torch.mean(torch.where(y_true==NEG_MIN, (torch.where(y_pred>NEG_MIN, ZERO, y_pred-NEG_MIN))**2, (y_pred-y_true)**2))

def train(model,loss_fn,optimizer,train_data_loader):
    train_loss_list = []
    for i_batch,batch in enumerate(train_data_loader):
        x = batch['feature'].float().cuda()
        A = batch['adj'].float().cuda()
        y = batch['label'].cuda()             # label is in int type
        y_pred = model(x,A) 
        loss = loss_fn(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.data.cpu().numpy())
    return train_loss_list

def validate(model,loss_fn,val_data_loader):
    val_loss_list = []
    for i_batch,batch in enumerate(val_data_loader):
        x = batch['feature'].float().cuda()
        A = batch['adj'].float().cuda()
        y = batch['label'].cuda()
        y_pred = model(x,A)
        loss = loss_fn(y_pred,y)
        val_loss_list.append(loss.data.cpu().numpy())
    return val_loss_list

def test(model,loss_fn,test_data_loader):
    test_loss_list = []
    for i_batch,batch in enumerate(test_data_loader):
        x = batch['feature'].float().cuda()
        A = batch['adj'].float().cuda()
        y = batch['label'].cuda()
        y_pred = model(x,A) 
        loss = loss_fn(y_pred,y)
        test_loss_list.append(loss.data.cpu().numpy())
    return test_loss_list


def train_DFRscore(args):
    # 0. initial setting
    data_dir = args.data_dir
    save_dir = args.save_dir
    log = args.logger
    log('\n2. Model Training Phase')
    new_data_dir=data_dir
    now = datetime.now()
    since_from = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    since = time.time()
    # 1. Set training parameters
    torch.set_num_threads(int(args.num_threads))

    loss_fn = HingeMSELoss
    global MAX_STEP
    MAX_STEP = args.max_step

    predictor = DFRscore(
            conv_dim=args.conv_dim,
            fc_dim=args.fc_dim,
            n_GAT_layer=args.n_conv_layer,
            n_fc_layer=args.n_fc_layer,
            num_heads=args.num_heads,
            len_features=args.len_features,
            max_num_atoms=args.max_num_atoms,
            max_step=args.max_step,
            dropout=args.dropout)
    predictor.cuda()

    optimizer = torch.optim.Adam(predictor.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=args.factor,
                    patience=args.patience,
                    threshold=args.threshold,
                    min_lr=args.min_lr
                    )
    lr = args.lr

    log('  ----- Train Config Information -----')
    log(f'  save_dir: {args.save_dir}')
    log(f'  data dir: {args.data_dir}')
    log.log_arguments(args)
    log()
    log('  ----- Training Log -----')
    
    best_loss = 100000
    # 2. Training with validation
    train_data_loader = DataLoader(
                    TrainDataset(data_dir=f'{new_data_dir}/generated_data', key_dir=f'{new_data_dir}/data_keys',mode='train'),
                    batch_size=args.batch_size,
                    shuffle = True,
                    collate_fn=gat_collate_fn,
                    num_workers=int(args.num_threads)
                    )
    val_data_loader = DataLoader(
                    TrainDataset(data_dir=f'{new_data_dir}/generated_data', key_dir=f'{new_data_dir}/data_keys',mode='val'),
                    batch_size=args.batch_size,
                    shuffle = False,
                    collate_fn=gat_collate_fn,
                    num_workers=int(args.num_threads)
                    )
    #test_data_loader = DataLoader(
    #                TrainDataset(data_dir=f'{new_data_dir}/generated_data', key_dir=f'{new_data_dir}/data_keys',mode='test'),
    #                batch_size=args.batch_size,
    #                shuffle = False,
    #                collate_fn=gat_collate_fn
    #                )

    train_loss_history = []
    val_loss_history = []
    for i in range(args.num_epoch):
        epoch_start = time.time()
        # 2-1. Train phase
        predictor.train()
        train_epoch_loss_list = train(predictor,loss_fn,optimizer,train_data_loader)
        train_epoch_loss= np.mean(train_epoch_loss_list)
        train_loss_history.append(train_epoch_loss)
        if (i+1)%5 == 0:
            torch.save(predictor.state_dict(),f'{save_dir}/DFR_model_{str(i+1)}.pt')
    
        # 2-2. Validation phase
        predictor.eval()
        val_epoch_loss_list = validate(predictor,loss_fn,val_data_loader)
        val_epoch_loss = np.mean(val_epoch_loss_list)
        val_loss_history.append(val_epoch_loss)
    
        if best_loss > val_epoch_loss:
            best_loss = val_epoch_loss
            best_epoch = i+1
            best_model = copy.deepcopy(predictor.state_dict())
        epoch_end = time.time()

        # 2-3. Logging
        log(f'  {i+1}th epoch,',
            f'   training loss: {train_epoch_loss}',
            f'   val loss: {val_epoch_loss}',
            f'   epoch time: {epoch_end-epoch_start:.2f}')
        scheduler.step(val_epoch_loss)
        if optimizer.param_groups[0]["lr"] < lr:
            lr = float(optimizer.param_groups[0]["lr"])
            log(f'   scheduler has reduced lr, current is: {lr}')


    # 3. Finish and save the result
    torch.save(best_model,f'{save_dir}/Best_model_{str(best_epoch)}.pt')
    now = datetime.now()
    finished_at = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    time_elapsed = int(time.time()-since)
    log()
    log(f'  ----- Training Finised -----',
        f'  finished at : {finished_at}',
        '  time passed: [%dh:%dm:%ds]' %(time_elapsed//3600, (time_elapsed%3600)//60, time_elapsed%60),
        f'  Best epoch: {best_epoch}',
        f'  Best loss: {best_loss}',
        f'  Decayed_lr: {optimizer.param_groups[0]["lr"]}')
    with open(f'{save_dir}/loss_history.pkl', 'wb') as fw:
        pickle.dump({'train': train_loss_history, 'val':val_loss_history}, fw)

    # 4. Test phase
    #predictor.eval()
    #test_loss_list = test(predictor,loss_fn,test_data_loader)
    #test_loss= np.mean(test_loss_list)
    ## Logging
    #log()
    #log(f'  ----- Test result -----',f'  test loss: {test_loss}')

if __name__=='__main__':
    ZERO = torch.tensor(0).float()
    NEG_MIN = torch.tensor(5)
    y_true= torch.cat([torch.ones(5),torch.ones(5)*2,torch.ones(5)*3,torch.ones(5)*4,torch.ones(5)*5])
    y_pred= (torch.cat([torch.arange(5),torch.arange(5),torch.arange(5),torch.arange(5),torch.arange(5)])+1).float()
    print('y_true:', y_true)
    print('y_pred:', y_pred)
    print(torch.where(y_true==NEG_MIN, (torch.where(y_pred>NEG_MIN, ZERO, y_pred-NEG_MIN))**2, (y_pred-y_true)**2))
    print(torch.mean(torch.where(y_true==NEG_MIN, (torch.where(y_pred>NEG_MIN, ZERO, y_pred-NEG_MIN))**2, (y_pred-y_true)**2)))
    y_pred= (torch.ones(25)*6).float()
    print('y_pred:', y_pred)
    print(torch.where(y_true==NEG_MIN, (torch.where(y_pred>NEG_MIN, ZERO, y_pred-NEG_MIN))**2, (y_pred-y_true)**2))
    print(torch.mean(torch.where(y_true==NEG_MIN, (torch.where(y_pred>NEG_MIN, ZERO, y_pred-NEG_MIN))**2, (y_pred-y_true)**2)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scripts.modelScripts.model import SVS
from scripts.modelScripts.data import TrainDataset
import scripts.utils as utils

import numpy as np
import pickle
import os
import sys
import random

import time
import copy

import argparse
from datetime import datetime

def train(model,loss_fn,optimizer,train_data_loader):
    train_loss_list = []
    for i_batch,batch in enumerate(train_data_loader):
        x = batch['feature'].float().cuda()
        A = batch['adj'].float().cuda()
        y = batch['label'].long().cuda()
        y_pred = model(x,A) 
        #loss = loss_fn(y_pred,y)-lamb*torch.mean(torch.log(F.softmax(y_pred,dim=1)))
        loss = loss_fn(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss_list.append(loss.data.cpu().numpy())
    return train_loss_list

def validate(model,loss_fn,val_data_loader):
    val_loss_list = []
    for i_batch,batch in enumerate(val_data_loader):
        x = batch['feature'].float().cuda()
        A = batch['adj'].float().cuda()
        y = batch['label'].long().cuda()
        y_pred = model(x,A) 
        #loss = loss_fn(y_pred,y)-lamb*torch.mean(torch.log(F.softmax(y_pred,dim=1)))
        loss = loss_fn(y_pred,y)
        val_loss_list.append(loss.data.cpu().numpy())
    return val_loss_list

def test(model,loss_fn,test_data_loader):
    test_loss_list = []
    for i_batch,batch in enumerate(test_data_loader):
        x = batch['feature'].float().cuda()
        A = batch['adj'].float().cuda()
        y = batch['label'].long().cuda()
        y_pred = model(x,A) 
        #loss = loss_fn(y_pred,y)-lamb*torch.mean(torch.log(F.softmax(y_pred,dim=1)))
        loss = loss_fn(y_pred,y)
        test_loss_list.append(loss.data.cpu().numpy())
    return test_loss_list


def train_SVS(args):
    # 0. initial setting
    data_dir = args.data_dir
    #save_dir = os.path.join(args.save_dir,f'/model_{args.data_preprocessing}_{args.n_conv_layer}_{args.conv_dim}_{args.fc_dim}_{args.lr}')
    save_dir = args.save_dir
    log = args.logger
    log('2. Model Training Phase')
    # For myself
    '''
    import shutil
    dir_name=data_dir.split('/')[-1]
    save_name = save_dir.split('/')[-2]
    if os.path.isdir(f'/scratch/hwkim/{save_name}/{dir_name}'):
        log('Data Exists!')
    else:
        shutil.copytree(data_dir, f'/scratch/hwkim/{save_name}/{dir_name}')
        '''
    #new_data_dir=f'/scratch/hwkim/{save_name}/{dir_name}'
    new_data_dir=data_dir
    now = datetime.now()
    since_from = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    since = time.time()
    # 1. Set training parameters
    torch.set_num_threads(int(args.num_threads))
    os.environ['CUDA_VISIBLE_DEVICES'] = utils.get_cuda_visible_devices(1)

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    predictor = SVS(
            conv_dim=args.conv_dim,
            fc_dim=args.fc_dim,
            n_GAT_layer=args.n_conv_layer,
            n_fc_layer=args.n_fc_layer,
            num_heads=args.num_heads,
            len_features=args.len_features,
            max_num_atoms=args.max_num_atoms,
            num_class=args.max_step+1,
            dropout=args.dropout)
    optimizer = torch.optim.Adam(predictor.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.gamma)
    lr = args.lr
    predictor.cuda()
    log('  ----- Train Config Information -----')
    #log(f' data dir: {data_dir}')
    log('  save_dir: {save_dir}')
    log.log_arguments(args)
    log()
    log('  ----- Training Log -----')
    
    best_loss = 100000
    # 2. Training with validation
    train_data_loader = DataLoader(
                    TrainDataset(data_dir=f'{new_data_dir}/generated_data', key_dir=f'{new_data_dir}/data_keys',mode='train'),
                    batch_size=args.batch_size,
                    shuffle = True,
                    num_workers=2
                    )
    val_data_loader = DataLoader(
                    TrainDataset(data_dir=f'{new_data_dir}/generated_data', key_dir=f'{new_data_dir}/data_keys',mode='val'),
                    batch_size=args.batch_size,
                    shuffle = False
                    )
    test_data_loader = DataLoader(
                    TrainDataset(data_dir=f'{new_data_dir}/generated_data', key_dir=f'{new_data_dir}/data_keys',mode='test'),
                    batch_size=args.batch_size,
                    shuffle = False
                    )


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
            torch.save(predictor.state_dict(),f'{save_dir}/GAT_model_{str(i+1)}.pt')
    
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
        if i > args.decay_epoch:
            scheduler.step()
            lr *= args.gamma        # to report

    # 3. Finish and save the result
    torch.save(best_model,f'{save_dir}/GAT_best_model_{str(best_epoch)}.pt')
    now = datetime.now()
    finished_at = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    time_elapsed = int(time.time()-since)
    log()
    log(f'  ----- Training Finised -----',
        f'  finished at : {finished_at}',
        '  time passed: [%dh:%dm:%ds]' %(time_elapsed//3600, (time_elapsed%3600)//60, time_elapsed%60),
        f'  Best epoch: {best_epoch}',
        f'  Best loss: {best_loss}',
        f'  Decayed_lr: {lr}')
    with open(f'{save_dir}/loss_history.pkl', 'wb') as fw:
        pickle.dump({'train': train_loss_history, 'val':val_loss_history}, fw)

    # 4. Test phase
    predictor.eval()
    test_loss_list += test(predictor,loss_fn,test_data_loader)
    test_loss= np.mean(test_loss_list)
    # Logging
    log()
    log(f'  ----- Test result -----','  test loss: {test_loss}')

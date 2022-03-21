#!/bin/bash

#PBS -N Train_SVS
#PBS -l nodes=gnode1:ppn=4:gpu
#PBS -l walltime=7:00:00:00 

##### Run ##### 
date

source activate SVS

cd ~/SVS

data_dir=save/PubChem/retro_result/
save_name=base
data_preprocessing=data_base

# Training parameters
num_data=250000
lr=0.0004
num_epoch=200

decay_epoch=0
n_conv_layer=5
conv_dim=256
fc_dim=128
len_features=36

python train_model.py --data_dir $data_dir \
    --save_name $save_name \
    --data_preprocessing $data_preprocessing \
    --len_features $len_features \
    --lr $lr --decay_epoch $decay_epoch \
    --n_conv_layer $n_conv_layer \
    --conv_dim $conv_dim \
    --fc_dim $fc_dim 
    
date

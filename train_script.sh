#!/bin/bash

#PBS -N Train_SVS_test
#PBS -l nodes=gnode2:ppn=4:gpu
#PBS -l walltime=7:00:00:00 

##### Run ##### 
date

source activate SVS

cd ~/SVS

data_dir=save/PubChem/Prev_data/
model_save_name=previous_data_model
data_preprocessing=test_240000

# Training parameters
num_data=240000
lr=0.0004
num_epoch=1000
decay_epoch=0

n_conv_layer=4
n_fc_layer=4
conv_dim=256
fc_dim=128
len_features=36

python train_model.py --data_dir $data_dir \
    --save_name $model_save_name \
    --data_preprocessing $data_preprocessing \
    --num_data $num_data \
    --num_epoch $num_epoch \
    --lr $lr --decay_epoch $decay_epoch \
    --n_conv_layer $n_conv_layer \
    --n_fc_layer $n_fc_layer \
    --conv_dim $conv_dim \
    --fc_dim $fc_dim  \
    --len_features $len_features \
    
date

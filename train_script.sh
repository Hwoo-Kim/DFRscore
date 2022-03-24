#!/bin/bash

#PBS -N SVS_nConv4_lr0_001_batch128
#PBS -l nodes=gnode2:ppn=4:gpu
#PBS -l walltime=7:00:00:00 

##### Run ##### 
date

source activate SVS

cd ~/SVS

data_dir=save/PubChem/Prev_data/
model_save_name=nConv4_lr0_001_batch128
data_preprocessing=basic_processing

# Training parameters
num_data=240000
num_epoch=500
lr=0.001
threshold=0.001
batch_size=128

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
    --lr $lr  --threshold $threshold \
    --batch_size $batch_size \
    --n_conv_layer $n_conv_layer \
    --n_fc_layer $n_fc_layer \
    --conv_dim $conv_dim \
    --fc_dim $fc_dim  \
    --len_features $len_features \
    
date

#!/bin/bash

#PBS -N Train_SVS
#PBS -l nodes=gnode1:ppn=4:gpu
#PBS -l walltime=7:00:00:00 

##### Run ##### 
date

conda activate SVS

cd ~/SVS

data_dir=save/PubChem/Prev_data/
model_save_name=previous_data_model
data_preprocessing=data_processed

# Training parameters
num_data=240000
lr=0.0004
num_epoch=300

decay_epoch=0
n_conv_layer=5
n_fc_layer=4
conv_dim=256
fc_dim=128
len_features=36

python train_model.py --data_dir $data_dir \
    --save_name $model_save_name \
    --data_preprocessing $data_preprocessing \
    --lr $lr --decay_epoch $decay_epoch \
    --n_conv_layer $n_conv_layer \
    --n_fc_layer $n_fc_layer \
    --conv_dim $conv_dim \
    --fc_dim $fc_dim  \
    --num_data $num_data
    --len_features $len_features \
    
date

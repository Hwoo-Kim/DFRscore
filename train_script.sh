#!/bin/bash

#PBS -N MPNN_one_hundred
#PBS -l nodes=gnode10:ppn=4:gpus=1:gpu2
#PBS -l walltime=7:00:00:00 
#PBS -o out.txt

##### Run ##### 
date

source activate SVS_A

cd ~/SVS

data_dir=save/PubChem/mcule_selected/
model_save_name=MPNN_test
#data_preprocessing=no_additional_seed1024
data_preprocessing=with_ring_inform_0.1

# Training parameters
# Not yet.
num_data=25000
num_epoch=200
lr=0.0005
batch_size=32

# To be changed.
n_conv_layer=6
n_fc_layer=3
conv_dim=256
fc_dim=128
len_features=30
#len_features=36

python train_model.py --data_dir $data_dir \
    --save_name $model_save_name \
    --data_preprocessing $data_preprocessing \
    --num_data $num_data \
    --num_epoch $num_epoch \
    --lr $lr \
    --batch_size $batch_size \
    --n_conv_layer $n_conv_layer \
    --n_fc_layer $n_fc_layer \
    --conv_dim $conv_dim \
    --fc_dim $fc_dim  \
    --len_features $len_features \
    
date

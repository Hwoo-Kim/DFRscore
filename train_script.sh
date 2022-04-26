#!/bin/bash

#PBS -N nConv6_nfc3_convdim256_fcdim128
#PBS -l nodes=gnode7:ppn=4:gpus=1:gpu1
#PBS -l walltime=7:00:00:00 

##### Run ##### 
date

source activate SVS

cd ~/SVS

data_dir=save/PubChem/mcule_selected/
model_save_name=nConv6_nfc3_convdim256_fcdim128
#data_preprocessing=no_additional_seed1024
data_preprocessing=with_ring_inform

# Training parameters
# Not yet.
num_data=250000
num_epoch=250
lr=0.0004
batch_size=128

# To be changed.
n_conv_layer=6
n_fc_layer=3
conv_dim=256
fc_dim=128
#len_features=30
len_features=36

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

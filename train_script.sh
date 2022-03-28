#!/bin/bash

#PBS -N nConv6_nfc2_lr0_0004_conv256_fc128_drop0_2_totdegree
#PBS -l nodes=gnode2:ppn=4:gpu
#PBS -l walltime=5:00:00 

##### Run ##### 
date

source activate SVS

cd ~/SVS

data_dir=save/PubChem4M/retro_result/
model_save_name=nConv6_nfc2_lr0_0004_conv256_fc128_drop0_2_totdegree
data_preprocessing=totdegree_seed1024

# Training parameters
num_data=250000
num_epoch=300
lr=0.0004
batch_size=128

n_conv_layer=6
n_fc_layer=2
conv_dim=256
fc_dim=128
#len_features=34
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

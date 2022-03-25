#!/bin/bash

#PBS -N train_SVS_nConv6_lr0_0004_conv256_fc128_8HEADS_batch256
#PBS -l nodes=gnode8:ppn=4:gpu
#PBS -l walltime=7:00:00:00 

##### Run ##### 
date

source activate SVS

cd ~/SVS

data_dir=save/PubChem4M/retro_result/
model_save_name=SVS_nConv6_lr0_0004_conv256_fc128_8HEADS_batch256
data_preprocessing=random_seed_1024

# Training parameters
num_data=240000
num_epoch=400
lr=0.0004
batch_size=256

n_conv_layer=6
n_fc_layer=4
conv_dim=256
fc_dim=128
len_features=36

python train_model.py --data_dir $data_dir \
    --save_name $model_save_name \
    --data_preprocessing $data_preprocessing \
    --num_data $num_data \
    --num_epoch $num_epoch \
    --num_heads $num_heads \
    --lr $lr  --threshold $threshold \
    --batch_size $batch_size \
    --n_conv_layer $n_conv_layer \
    --n_fc_layer $n_fc_layer \
    --conv_dim $conv_dim \
    --fc_dim $fc_dim  \
    --len_features $len_features \
    
date

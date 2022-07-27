#!/bin/bash

#PBS -N DFRscore_train
#PBS -l nodes=gnode4:ppn=4:gpus=1:gpu1
#PBS -l walltime=7:00:00:00 
#PBS -o out.txt

##### Run ##### 
date

source ~/.bashrc
conda activate DFRscore

cd ~/DFRscore

data_dir=save/PubChem/retro_result/
model_save_name=DFRscore
data_preprocessing=basic_process

# Training parameters
num_data=250000
num_epoch=200
lr=0.0004
batch_size=128

feature_size=49
n_conv_layer=6
n_fc_layer=2
conv_dim=256
fc_dim=128

use_scratch=false

MAIN_CMD="python train_model.py
--data_dir $data_dir
--save_name $model_save_name
--data_preprocessing $data_preprocessing
--num_data $num_data
--num_epoch $num_epoch
--lr $lr
--batch_size $batch_size
--feature_size $feature_size
--n_conv_layer $n_conv_layer
--n_fc_layer $n_fc_layer
--conv_dim $conv_dim
--fc_dim $fc_dim"

if $use_scratch; then
    MAIN_CMD=$MAIN_CMD" --use_scratch"
fi

$MAIN_CMD

date

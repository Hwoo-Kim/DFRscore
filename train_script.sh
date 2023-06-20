#!/bin/bash

#PBS -N DFRscore_6_2_256_128
#PBS -l nodes=gnode3:ppn=4:gpus=1:gpu1
#PBS -l walltime=7:00:00:00 
#PBS -o out.txt

##### Run ##### 
date

shopt -s expand_aliases
source ~/.bashrc
mamba activate DFRscore

cd ~/DFRscore

data_dir=save/PubChem/retro_result/
model_save_name=DFRscore_6_2_256_128
data_preprocessing=basic_process
#data_preprocessing=no_ring

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

random_seed=1024
use_scratch=true
write_log_in_notion=false
database_id=""

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
--fc_dim $fc_dim
--random_seed $random_seed"

if $use_scratch; then
    MAIN_CMD=$MAIN_CMD" --use_scratch"
fi
if $write_log_in_notion; then
    MAIN_CMD=$MAIN_CMD" --database_id $database_id"
fi


$MAIN_CMD

date

#!/bin/bash

#PBS -N DFR_train
#PBS -l nodes=gnode6:ppn=4:gpus=1:gpu1
#PBS -l walltime=7:00:00:00 
#PBS -o out.txt

##### Run ##### 
date

#source activate DFRscore_A
source activate DFRscore

cd ~/DFRscore

data_dir=save/PubChem/retro_result/
model_save_name=DFRscore
data_preprocessing=basic_process

# Training parameters
num_data=250000
num_epoch=200
lr=0.0004
batch_size=128

n_conv_layer=5
n_fc_layer=4
conv_dim=256
fc_dim=128

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

date

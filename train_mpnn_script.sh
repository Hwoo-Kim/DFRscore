#!/bin/bash

#PBS -N MPNN_
#PBS -l nodes=gnode10:ppn=4:gpus=1:gpu2
#PBS -l walltime=7:00:00:00 
#PBS -o out.txt

##### Run ##### 
date

#. ~/.bashrc
source activate SVS_A

cd ~/SVS

data_dir=save/PubChem/mcule_selected/
model_save_name=MPNN_full
#data_preprocessing=no_additional_seed1024
data_preprocessing=with_ring_inform

# Training parameters
# Not yet.
num_data=250000
num_epoch=100
lr=0.0005
batch_size=32

# To be changed.
edge_dim=5
node_dim=30
hidden_dim=80
message_dim=80
num_layers=3
dropout=0.2

python train_mpnn_model.py --data_dir $data_dir \
    --save_name $model_save_name \
    --data_preprocessing $data_preprocessing \
    --num_data $num_data \
    --num_epoch $num_epoch \
    --lr $lr \
    --batch_size $batch_size \
    --edge_dim $edge_dim \
    --node_dim $node_dim \
    --hidden_dim $hidden_dim \
    --message_dim $message_dim \
    --num_layers $num_layers \
    --dropout $dropout \
    
date

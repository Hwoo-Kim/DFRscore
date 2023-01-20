#!/bin/bash

#PBS -N DFR_Exp04_GGM
#PBS -l nodes=cnode13:ppn=16
#PBS -l walltime=7:00:00:00 

shopt -s expand_aliases
source ~/.bashrc
mamba activate DFRscore

cd ~/DFRscore/Experiments/exp4/

num_cores=16
model_path=/home/hwkim/DFRscore/save/PubChem/DFRscore/Best_model_163.pt
#data=FDA
data=GDBChEMBL

python Exp04.py \
    --test_data $data \
    --num_cores $num_cores \
    --model_path $model_path \
    --use_cuda


#!/bin/bash

#PBS -N DFR_Exp03
#PBS -l nodes=gnode4:ppn=16:gpus=1:gpu1
#PBS -l walltime=7:00:00:00 

source ~/.bashrc
conda activate DFRscore

cd ~/DFRscore/Experiments/exp3/

num_cores=16
#model_path=/home/hwkim/DFRscore/save/PubChem/DFRscore/Best_model_118.pt
model_path=/home/hwkim/DFRscore/save/PubChem/DFRscore_new_feature/Best_model_163.pt

data=GGM
python Exp03.py \
    --num_cores $num_cores \
    --model_path $model_path \
    --test_data $data

data=GDBChEMBL
python Exp03.py \
    --num_cores $num_cores \
    --model_path $model_path \
    --test_data $data

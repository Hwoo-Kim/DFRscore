#!/bin/bash

#PBS -N DFR_Exp04_FDA
#PBS -l nodes=gnode8:ppn=16:gpus=1:gpu1
#PBS -l walltime=7:00:00:00 

source ~/.bashrc
conda activate DFRscore

cd ~/DFRscore/Experiments/exp4/

num_cores=16
model_path=/home/hwkim/DFRscore/save/PubChem/DFRscore/Best_model_163.pt
data=FDA

python Exp04.py \
    --test_data $data \
    --num_cores $num_cores \
    --model_path $model_path \
    --use_cuda


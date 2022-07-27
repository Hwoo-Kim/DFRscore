#!/bin/bash

#PBS -N DFR_time_check
#PBS -l nodes=gnode4:ppn=4:gpus=1:gpu1
#PBS -l walltime=7:00:00:00 
#PBS -o out.txt
#PBS -e error.txt

source ~/.bashrc
condat activate DFRscore
cd ~/DFRscore/Experiments/exp2

num_cores=4
model_path=/home/hwkim/DFRscore/save/PubChem/DFRscore/Best_model_118.pt
test_path=/home/hwkim/DFRscore/data/retro_target_data

##### Main #####
date

python time_check.py \
    --num_cores $num_cores \
    --model_path $model_path \
    --test_path $test_path \
    --use_cuda

python time_check.py \
    --num_cores $num_cores \
    --model_path $model_path \
    --test_path $test_path

date

#!/bin/bash

#PBS -N DFR_time_check
#PBS -l nodes=gnode7:ppn=4:gpus=1:gpu2
#PBS -l walltime=7:00:00:00 
#PBS -o out.txt
#PBS -e error.txt

shopt -s expand_aliases
source ~/.bashrc
mamba activate DFRscore
cd ~/DFRscore/Experiments/exp2

export CUDA_VISIBLE_DEVICES=0

num_cores=4
model_path=/home/hwkim/DFRscore/save/PubChem/DFRscore/Best_model_163.pt
test_path=/home/hwkim/DFRscore/data/retro_target_data

##### Main #####
date

#python time_check.py \
#    --num_cores $num_cores \
#    --model_path $model_path \
#    --test_path $test_path \
#    --seed 0
#
#python time_check.py \
#    --num_cores $num_cores \
#    --model_path $model_path \
#    --test_path $test_path \
#    --seed 1
#
#python time_check.py \
#    --num_cores $num_cores \
#    --model_path $model_path \
#    --test_path $test_path \
#    --seed 2

python time_check.py \
    --num_cores $num_cores \
    --model_path $model_path \
    --test_path $test_path \
    --seed 3

#!/bin/bash

#PBS -N DFR_Exp03_GGM&GDBChEMBL
#PBS -l nodes=gnode4:ppn=16:gpus=1:gpu1
#PBS -l walltime=7:00:00:00 

source activate DFRscore
conda activate DFRscore

cd ~/DFRscore/Experiments/exp1/

num_cores=16
class_size=2000
model_path=/home/hwkim/DFRscore/save/PubChem/DFRscore/Best_model_118.pt

data=ZINC
python Exp01.py \
    --test_data $data \
    --num_cores $num_cores \
    --class_size $class_size \
    --model_path $model_path

data=ChEMBL
python Exp01.py \
    --test_data $data \
    --num_cores $num_cores \
    --class_size $class_size \
    --model_path $model_path

data=MOSES
python Exp01.py \
    --test_data $data \
    --num_cores $num_cores \
    --class_size $class_size \
    --model_path $model_path

data=GGM
python Exp01.py \
    --test_data $data \
    --num_cores $num_cores \
    --class_size $class_size \
    --model_path $model_path \
    --only_DFR

data=GDBChEMBL
python Exp01.py \
    --test_data $data \
    --num_cores $num_cores \
    --class_size $class_size \
    --model_path $model_path
    --only_DFR

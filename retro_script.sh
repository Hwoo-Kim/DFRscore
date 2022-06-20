#!/bin/bash

#PBS -N GGM_250k_11000
#PBS -l nodes=cnode17:ppn=16
#PBS -l walltime=7:00:00:00
#PBS -o /home/hwkim/DFRscore/save/out.out
#PBS -e /home/hwkim/DFRscore/save/error.txt

##### Run ##### 

date

source activate DFRscore
conda activate DFRscore

cd ~/DFRscore

retro_target=data/retro_target_data/GGM.smi
#retro_target=data/SCscore_case_study/false_pos.smi
#retro_target=data/retro_target_data/test.smi

python retro_analysis.py \
    --template data/template/retro_template.pkl \
    --reactant data/reactant_bag/mcule.smi \
    --retro_target $retro_target \
    --depth 4 \
    --start_index 250000 \
    --num_cores 16 \
    --num_molecules 11000 \
    --exclude_in_R_bag True \
    --path False \
    --max_time 3600

date

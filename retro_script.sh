#!/bin/bash

#PBS -N Retro_GGM_100k_text
#PBS -l nodes=gnode10:ppn=32
#PBS -l walltime=7:00:00:00
#PBS -o /home/hwkim/SVS/save/out.out
#PBS -e /home/hwkim/SVS/save/error.txt

##### Run ##### 

date

source activate SVS
conda activate SVS

cd ~/SVS

retro_target=data/retro_target_data/GGM_test.smi
#retro_target=data/retro_target_data/test.smi

python retro_analysis.py \
    --template data/template/retro_template.pkl \
    --reactant data/reactant_bag/mcule.smi \
    --retro_target $retro_target \
    --depth 4 \
    --start_index 0 \
    --num_cores 32 \
    --num_molecules 100000 \
    --exclude_in_R_bag True \
    --path False
#    --max_time 300

date

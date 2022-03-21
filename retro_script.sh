#!/bin/bash

#PBS -N GGM_test
#PBS -l nodes=cnode11:ppn=16
#PBS -l walltime=7:00:00:00
#PBS -o /home/hwkim/SVS/save/out.out
#PBS -e /home/hwkim/SVS/save/error.txt

##### Run ##### 

date

source activate SVS

cd ~/SVS

#export retro_target=data/retro_target_data/GGM.smi
export retro_target=data/retro_target_data/test.smi

python retro_analysis.py \
    --template data/template/retro_template.pkl \
    --reactant data/reactant_bag/ZINC_reactants.smi \
    --retro_target $retro_target \
    --depth 4 \
    --num_molecules 1 \
    --num_cores 1

date

#!/bin/bash

#PBS -N Retro_path_test_PubChem
#PBS -l nodes=gnode2:ppn=4
#PBS -l walltime=7:00:00:00
#PBS -o /home/hwkim/SVS/save/out.out
#PBS -e /home/hwkim/SVS/save/error.txt

##### Run ##### 

date

source activate SVS

cd ~/SVS

retro_target=data/retro_target_data/PubChem.smi
#retro_target=data/retro_target_data/DrugBank.smi

python retro_analysis.py \
    --template data/template/retro_template.pkl \
    --reactant data/reactant_bag/emolecules.smi \
    --retro_target $retro_target \
    --depth 4 \
    --num_cores 20 \
    --num_molecules 160 \
    --path True \
    --max_time 3000
#    --num_molecules 160

date

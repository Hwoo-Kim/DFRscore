#!/bin/bash

#PBS -N Retro_PubChem_for_timecheck
#PBS -l nodes=cnode4:ppn=20
#PBS -l walltime=7:00:00:00
#PBS -o /home/hwkim/SVS/save/out.out
#PBS -e /home/hwkim/SVS/save/error.txt

##### Run ##### 

date

source activate SVS
conda activate SVS

cd ~/SVS

retro_target=data/retro_target_data/PubChem.smi

python retro_analysis.py \
    --template data/template/retro_template.pkl \
    --reactant data/reactant_bag/PubChem.smi \
    --retro_target $retro_target \
    --depth 4 \
    --start_index 0 \
    --num_cores 20 \
    --num_molecules 100000 \
    --exclude_in_R_bag True

#    --max_time 300 \
#    --num_molecules 100 \
#    --path True 

date

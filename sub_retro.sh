#!/bin/bash

#PBS -N GDBMedChem
#PBS -l nodes=cnode17:ppn=16
#PBS -l walltime=7:00:00:00

##### Run ##### 

date

conda activate SVS

cd ~/SVS

python retro_analysis.py \
    --template data/template/retro_template.pkl \
    --reactant data/reactant_bag/canonicalized.smi \
    --retro_target data/retro_target_data/ChEMBL/filtered_ChEMBL.smi \
    --depth 4 \
    --num_molecules 1000 \
    --num_cores 20

date

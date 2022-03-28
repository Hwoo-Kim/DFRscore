#!/bin/bash

#PBS -N Retro_DrugBank
#PBS -l nodes=cnode15:ppn=16
#PBS -l walltime=7:00:00:00
#PBS -o /home/hwkim/SVS/save/out.out
#PBS -e /home/hwkim/SVS/save/error.txt

##### Run ##### 

date

source activate SVS

cd ~/SVS

#retro_target=data/retro_target_data/PubChem_11k.smi
retro_target=data/retro_target_data/DrugBank.smi

python retro_analysis.py \
    --template data/template/retro_template.pkl \
    --reactant data/reactant_bag/ZINC_reactants.smi \
    --retro_target $retro_target \
    --depth 4 \
    --num_molecules 7877 \
    --num_cores 16

date

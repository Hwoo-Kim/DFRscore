#!/bin/bash

#PBS -N Retro_data_check
#PBS -l nodes=cnode16:ppn=16
#PBS -l walltime=7:00:00:00
#PBS -o /home/hwkim/DFRscore/save/out.out
#PBS -e /home/hwkim/DFRscore/save/error.txt

##### Run ##### 

date

source activate DFRscore
conda activate DFRscore

cd ~/DFRscore

#retro_target=data/retro_target_data/MOSES.smi
#retro_target=data/SCscore_case_study/false_pos.smi
#retro_target=data/retro_target_data/test.smi
retro_target=data/retro_target_data/UDG_PUBCHEM.smi
exclude_in_R_bag=true
path=false

MAIN_CMD="python retro_analysis.py
--template data/template/retro_template.pkl
--reactant data/reactant_bag/mcule.smi
--retro_target $retro_target
--depth 4
--start_index 0
--num_cores 16
--num_molecules 250000
--max_time 3600"

if $exclude_in_R_bag; then
    MAIN_CMD=$MAIN_CMD" --exclude_in_R_bag"
if $path; then
    MAIN_CMD=$MAIN_CMD" --path"

$MAIN_CMD

date

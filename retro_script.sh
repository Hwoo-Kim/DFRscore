#!/bin/bash

#PBS -N Retro_further_chembl
#PBS -l nodes=cnode16:ppn=16
#PBS -l walltime=7:00:00:00
#PBS -o /home/hwkim/DFRscore/save/out.out
#PBS -e /home/hwkim/DFRscore/save/error.txt

##### Run ##### 

date

source ~/.bashrc
micromamba activate DFRscore

cd ~/DFRscore

#retro_target=data/retro_target_data/PubChem.smi
#retro_target=data/retro_target_data/ZINC.smi
#retro_target=data/retro_target_data/ChEMBL.smi
#retro_target=data/retro_target_data/MOSES.smi
#retro_target=data/retro_target_data/test.smi
#retro_target=data/retro_target_data/UDG_PUBCHEM.smi

#retro_target=data/scscore_case_study/false_pos.smi
#retro_target=data/scscore_case_study/result.smi
#retro_target=data/scscore_case_study/false_neg.smi

#retro_target=Experiments/exp1/further_analysis/further_zinc.smi
retro_target=Experiments/exp1/further_analysis/further_chembl.smi

exclude_in_R_bag=true
path=true

MAIN_CMD="python retro_analysis.py
--template data/template/retro_template.pkl
--reactant data/reactant_bag/mcule.smi
--retro_target $retro_target
--depth 4
--start_index 0
--num_cores 16
--num_molecules 1000
--max_time 3600"

if $exclude_in_R_bag; then
    MAIN_CMD=$MAIN_CMD" --exclude_in_R_bag"
fi
if $path; then
    MAIN_CMD=$MAIN_CMD" --path"
fi

$MAIN_CMD

date

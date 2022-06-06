#!/bin/bash

source activate SVS
conda activate SVS

cd ~/SVS/Experiments/exp1/

num_cores=16

data=ZINC
each_class_size=2000
python Exp01.py $data $num_cores $each_class_size

#data=ChEMBL
#each_class_size=2000
#python Exp01.py $data $num_cores $each_class_size
#
#data=MOSES
#each_class_size=1850
#python Exp01.py $data $num_cores $each_class_size
#

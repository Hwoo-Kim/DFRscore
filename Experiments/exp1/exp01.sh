#!/bin/bash

source activate DFRscore
conda activate DFRscore

cd ~/DFRscore/Experiments/exp1/

num_cores=4

data=ZINC
each_class_size=2000
python Exp01.py $data $num_cores $each_class_size

data=ChEMBL
each_class_size=2000
python Exp01.py $data $num_cores $each_class_size

data=MOSES
each_class_size=2000
python Exp01.py $data $num_cores $each_class_size

data=PubChem
each_class_size=2000
python Exp01.py $data $num_cores $each_class_size

#!/bin/bash

source activate DFRscore
conda activate DFRscore

cd ~/DFRscore/Experiments/exp1/

python make_violin.py ZINC
python make_violin.py ChEMBL
python make_violin.py MOSES

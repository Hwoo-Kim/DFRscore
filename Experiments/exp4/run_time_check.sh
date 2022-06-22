#!/bin/bash

#source activate DFRscore

cd ~/DFRscore/Experiments/exp4

num_cores=4
use_cuda=True
date
python time_check.py $num_cores $use_cuda
date

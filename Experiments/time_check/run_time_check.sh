#!/bin/bash

#source activate DFRscore

cd ~/DFRscore/Experiments/time_check

num_cores=4
use_cuda=False
date
python time_check.py $num_cores $use_cuda
date

#!/bin/zsh

#source activate DFRscore

cd ~/works/DFRscore/Experiments/time_check

num_cores=4
use_cuda=False
#python time_check.py $num_cores $use_cuda >> out.log
python time_check.py $num_cores $use_cuda

#!/bin/bash

source ~/.bashrc
conda activate DFRscore

cd ~/DFRscore/Experiments/exp4/

#python dist_plot.py FDA
python dist_plot.py GDBChEMBL

#!/bin/bash

num_cores=4
python mcule_filter.py ./mcule_bb_canon.smi $num_cores <<EOF
Y
EOF

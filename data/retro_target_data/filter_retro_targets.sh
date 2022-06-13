#!/bin/bash

num_cores=16

#1 PubChem
python molecules_filter.py before_filtering/PubChem_smiles $num_cores 10000000 <<EOF
Y
EOF

#2 ZINC
python molecules_filter.py before_filtering/ZINC.smi $num_cores 2000000 <<EOF
Y
EOF

#3 ChEMBL
python molecules_filter.py before_filtering/ChEMBL.smi $num_cores <<EOF
Y
EOF

#4 MOSES
python molecules_filter.py before_filtering/moses_data.smi $num_cores <<EOF
Y
EOF

#5 GGM
python molecules_filter.py before_filtering/GGM.smi $num_cores <<EOF
Y
EOF

#6 GDBChEMBL
python molecules_filter.py before_filtering/GDBChEMBL.smi $num_cores 1000000 <<EOF
Y
EOF

#7 ARAE
python molecules_filter.py before_filtering/ARAE.smi $num_cores <<EOF
Y
EOF

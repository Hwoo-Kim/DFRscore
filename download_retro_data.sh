#!/bin/bash

REACTANT_DIR=data/reactant_bag
RETRO_TARGET_DIR=data/retro_target_data

# Download reactant bag data
#wget  -O $REACTANT_DIR/ZINC_reactants.smi
cp /home/share/DATA/hwkim_SVS/reactant_bag.tar.gz $REACTANT_DIR
tar zxf /home/share/DATA/hwkim_SVS/reactant_bag.tar.gz -C $REACTANT_DIR
echo "Reactant bag downloaded in:" $REACTANT_DIR

# Download retro target data
#wget  -O $RETRO_TARGET_DIR/.smi
cp /home/share/DATA/hwkim_SVS/retro_target_data.tar.gz $RETRO_TARGET_DIR
tar zxf /home/share/DATA/hwkim_SVS/retro_target_data.tar.gz -C $RETRO_TARGET_DIR
echo "Retro target data downloaded in:" $RETRO_TARGET_DIR

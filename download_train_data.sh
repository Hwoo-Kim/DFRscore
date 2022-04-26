#!/bin/bash

TRAIN_DATA_DIR=save/PubChem/mcule_selected
if [ ! -d $TRAIN_DATA_DIR]; then
    mkdir $TRAIN_DATA_DIR
fi

# Download train data
#wget  -O $TRAIN_DATA_DIR/
cp /home/share/DATA/hwkim_SVS/train_data.tar.gz $TRAIN_DATA_DIR
tar -zxf /home/share/DATA/hwkim_SVS/train_data.tar.gz -C $TRAIN_DATA_DIR
echo "Train data downloaded in:" $TRAIN_DATA_DIR

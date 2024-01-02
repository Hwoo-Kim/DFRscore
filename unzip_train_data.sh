#!/bin/bash

RETRO_DATA_DIR=save/PubChem/retro_result

# unzip training data
tar -zxvf $RETRO_DATA_DIR/retro_result.tar.gz -C $RETRO_DATA_DIR
echo "Train data has been unzipped at:" $RETRO_DATA_DIR

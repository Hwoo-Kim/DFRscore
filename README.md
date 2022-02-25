# Synthesizability for Virtual Screening
Scoring synthesizability of drug candidates using GAT model

# Table of Contents

## Install Dependencies
SVS needs conda environment. After installing [conda](https://www.anaconda.com/),   
you can manually install the required pakages as follows:
- rdkit=2020.09.1
- scipy
- numpy
- scikit-learn
- pytorch

Or simply you can install the required packages by running
```
./dependencies
```
This will configure a new conda environment named 'SVS'.

## Download Data
1. If you ONLY want to train the SVS model without retro-analysis, you can download the data for training with:   
```
./download_train_data.sh
```
And go [here](#retro-analysis).

2. Or, if you want to get the training data by running our retro-anaylsis tool followed by model training, you can download the ingredients with:
```
./download_retro_and_train_data.sh
```
And go [here](#train).

## Retro-analysis
After downloading the data by ```./download_training_data.sh```, you can run retro-analysis tool by:
```
python ./retro_analysis.py --data 
```
And go [here](#train).

## Train and Test
After getting data for training by ```./download_retro_data.sh``` or manually running ```./retro_analysis.py```,   
### Train
You can train a new model by:
```
python ./train_model.py
```
### Test
You can test your model by:
```
python ./test_model.py
```

## Using trained model as a tool for VS
Getting a score directly from chemical SMILES is described in ```SVS/getSVS.py```.   
For a fast test, you can simply run ```python SVS/getSVS.py --model_dir <path_to_model_dir> --smi <SMILES>```

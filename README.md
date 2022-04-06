# Synthesizability for Virtual Screening
Scoring synthesizability of drug candidates using GAT model

# Table of Contents
- [Install Dependencies](#install-dependencies)
- [Download Data](#download-data)
- [Retro-analysis](#retro-analysis)
- [Train and Test](#train-and-test)
  - [Train](#train)
  - [Test](#test)
- [Using trained model as a scoring metric for VS](#using-trained-model-as-a-scoring-metric-for-vs)

## Install Dependencies
SVS needs conda environment. After installing [conda](https://www.anaconda.com/),   
you can manually install the required pakages as follows:
- rdkit=2020.09.1
- matplotlib
- scipy
- numpy
- scikit-learn
- pytorch>=1.9.0

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
And go to [Train and Test](#train-and-test).

2. Or, if you want to get the training data by running our retro-anaylsis tool followed by model training, you can download the ingredients with:
```
./download_retro_data.sh
```
And go to [Retro-analysis](#retro-analysis).

## Retro-analysis
After downloading the data by ```./download_retro_data.sh```, you can run retro-analysis tool by:
```
python retro_analysis.py \
    --template data/template/retro_template.pkl \
    --reactant data/reactant_bag/filtered_ZINC.smi \
    --retro_target <PATH TO TARGET FILE> \
    --depth 4 \
    --num_molecules <NUMBER OF MOLECULES TO BE ANALYZED> \
    --num_cores <NUMBER OF CPU CORES>
```
And go to [Train and Test](#train-and-test).

## Train and Test
After getting data for training by ```./download_train_data.sh``` or manually running ```./retro_analysis.py```,   

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

## Using trained model as a scoring metric for VS
Useful functions are described in ```SVS``` class in ```scripts/modelScripts/model.py``` file.

For a fast test, you can simply run ```python getSVS.py --model_dir <path_to_model_dir> --smi <SMILES>```

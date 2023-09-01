# DFRscore
Code for the paper: "[DFRscore: Deep Learning-Based Scoring of Synthetic Complexity with Drug-Focused Retrosynthetic Analysis for High-Throughput Virtual Screening](https://doi.org/10.1021/acs.jcim.3c01134)" by Hyeongwoo Kim, Kyunghoon Lee, Chansu Kim, Jaechang Lim, and Woo Youn Kim (currently in acceptance)

# Table of Contents
- [Install Dependencies](#install-dependencies)
- [Download Data](#download-data)
- [Retrosynthetic analysis](#retrosynthetic-analysis)
- [Training DFRscore](#training-dfrscore)
- [Using DFRscore to Evaluate Synthetic Accessibility](#using-dfrscore-to-evaluate-synthetic-accessibility)

## Install Dependencies
DFRscore needs conda environment. After installing [conda](https://www.anaconda.com/),   
you can manually install the required pakages as follows:
- rdkit=2020.09.1
- matplotlib
- scipy
- numpy
- scikit-learn
- pytorch>=1.11.0

Or simply you can install the required packages by running
```bash
. ./dependencies
```
This will configure a new conda environment named 'DFRscore'.

## Download Data
### Currently not working because the data links are not yet prepared... They would be available with Zenodo as soon as possible.
1. If you ONLY want to train the DFRscore model without retrosynthetic analysis, you can download the data for training with:   
```bash
./download_train_data.sh
```
And go to [Training DFRscore](#training-dfrscore).

2. Or, if you want to get the training data by running our retrosynthetic anaylsis tool (FRA) followed by model training, you can download the ingredients with:
```bash
./download_retro_data.sh
```
And go to [Retrosynthetic Analysis](#retrosynthetic-analysis).

## Retrosynthetic Analysis
After downloading the data by ```./download_retro_data.sh```, you can run retro-analysis tool by:
```bash
python retro_analysis.py \
--template data/template/retro_template.pkl \
--reactant data/reactant_bag/filtered_ZINC.smi \
--retro_target <PATH-TO-TARGET-FILE> \
--depth 4 \
--num_molecules <NUMBER-OF-MOLECULES-TO-BE-ANALYZED> \
--num_cores <NUMBER-OF-CPU-CORES>
```
And go to [Training DFRscore](#training-dfrscore).

## Training DFRscore
After getting data for training by ```./download_train_data.sh``` or manually running ```./retro_analysis.py```, you can train a new model by:
```bash
python ./train_model.py
```

## Using DFRscore to Evaluate Synthetic Accessibility
Useful functions are described in ```DFRscore``` class in ```scripts/modelScripts/model.py``` file.

For a fast test, you can simply run ```python getDFRscore.py --model_path <path_to_model_dir> --smiles <SMILES>```

For example, ```python get_DFRscore.py --model_path ./save/PubChem/DFRscore/Best_model_163.pt --smiles "CC(=O)Nc1ccc(cc1)OCC"```

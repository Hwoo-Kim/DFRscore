# DFRscore
Code for the paper: "[DFRscore: Deep Learning-Based Scoring of Synthetic Complexity with Drug-Focused Retrosynthetic Analysis for High-Throughput Virtual Screening](https://doi.org/10.1021/acs.jcim.3c01134)" by Hyeongwoo Kim, Kyunghoon Lee, Chansu Kim, Jaechang Lim, and Woo Youn Kim.

# Table of Contents
- [Install Dependencies](#install-dependencies)
- [Preparing Training Data](#preparing-training-data)
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

## Preparing Training Data
1. If you ONLY want to train the DFRscore model without retrosynthetic analysis, you can get the data for training with:   
```bash
./unzip_train_data.sh
```
And go to [Training DFRscore](#training-dfrscore).

2. Or, if you want to get the training data by running our retrosynthetic anaylsis tool (FRA) followed by model training, you should download the ingredients and put them
at ```data/retro_target_data```. Every line of the file should inclue a single molecule of SMILES representation. You can filter them with ```data/retro_target_data/molecules_filter.py```.

And go to [Retrosynthetic Analysis](#retrosynthetic-analysis).

## Retrosynthetic Analysis
After preparing the target molecule data for retrosynthetic analysis, you can run FRA by:
```bash
python retro_analysis.py \
--template data/template/retro_template.pkl \
--reactant data/reactant_bag/filtered_mcule.smi \
--retro_target <PATH-TO-TARGET-FILE> \
--depth 4 \
--num_molecules <NUMBER-OF-MOLECULES-TO-BE-ANALYZED> \
--num_cores <NUMBER-OF-CPU-CORES>
```
And go to [Training DFRscore](#training-dfrscore).

## Training DFRscore
After getting data for training by ```./unzip_train_data.sh``` or manually running ```./retro_analysis.py```, you can train a new model by:
```bash
python ./train_model.py
```

## Using DFRscore to Evaluate Synthetic Accessibility
Useful functions are described in ```DFRscore``` class in ```scripts/modelScripts/model.py``` file.

For a fast test, you can simply run ```python getDFRscore.py --model_path <path_to_model_dir> --smiles <SMILES>```

For example, ```python get_DFRscore.py --model_path ./save/PubChem/DFRscore/Best_model_163.pt --smiles "CC(=O)Nc1ccc(cc1)OCC"```

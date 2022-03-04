#!/bin/bash
#PBS -N PubC_0.0004_formal_charge_4_128
#PBS -l nodes=1:ppn=4:gnode4:gpu
#PBS -l walltime=7:00:00:00 
date

##### Run ##### 
conda activate HWKim
cd /home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/
#cd /home/wykgroup/hwkim/Synthesizability_for_VS/model_train/

data_dir1=data_for_train/PubChem_200k
data_dir2=data_for_train/Pub_200k_ZINC_100k 

save_name1=PubChem_200k_debugging
save_name2=Pub_200k_ZINC_100k_ring

training_data_path=formal_charge11
data_preprocessing=False # True for data generation and training, False for only training
len_features=36

# Training parameters
lr=0.0004
decay_epoch=0
n_conv_layer=4
conv_dim=256
fc_dim=128

python train.py --data_dir $data_dir1 --save_name $save_name1 --data_preprocessing $data_preprocessing --len_features $len_features --lr $lr --decay_epoch $decay_epoch --n_conv_layer $n_conv_layer --conv_dim $conv_dim --fc_dim $fc_dim --training_data_path $training_data_path
date

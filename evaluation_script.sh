#!/bin/bash

#PBS -N EXP01_GAT
#PBS -l nodes=1:ppn=4:gnode2:gpu
#PBS -l walltime=7:00:00:00 
date

##### Run ##### 
cd /home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/
conda activate HWKim

#export data_dir0=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/model_test/PubChem_test_set/
data_dir0=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/model_test/Pub_ZINC_test_set/
data_dir1=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/exp1/ChEMBL/
data_dir2=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/exp1/MOSES/
data_dir3=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/exp1/ZINC/

data_dir4=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/exp3/ARAE/
data_dir5=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/exp3/GGM/
data_dir6=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/exp3/drug_bank/

# For EXP02
data_dir7=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/exp2/

#save_dir0=PubChem_test_set
save_dir0=Pub_ZINC_test_set
save_dir1=ChEMBL
save_dir2=MOSES
save_dir3=ZINC

save_dir4=ARAE
save_dir5=GGM
save_dir6=drug_bank

model_path1=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/result_train/PubChem_200k_GAT/GAT_30features/GAT_best_model_290.pt
model_path2=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/result_train/Pub_200k_ZINC_100k_GAT/GAT_30features/GAT_best_model_281.pt

model_name1=PubChem_200k_GAT
model_name2=Pub_200k_ZINC_100k_GAT

# DI for data generation and inference, I for only inference 
#mode=DI
max_step=4
shuffle_before_sampling=False
max_num_atoms=64
len_features=30
n_dim=256
cuda_device=1
num_threads=4
each_class_size01=2000
each_class_size03=100000

# For EXP01
#python model_evaluation.py --data_dir $data_dir1 --save_dir $save_dir1 --model_path $model_path1 --model_name $model_name1 \
#    --max_step $max_step --shuffle_before_sampling $shuffle_before_sampling  \
#    --model_type $model_type --max_num_atoms $max_num_atoms --len_features $len_features --n_dim $n_dim \
#    --cuda_device $cuda_device --num_threads $num_threads --each_class_size $each_class_size01 --exp EXP01
#python model_evaluation.py --data_dir $data_dir2 --save_dir $save_dir2 --model_path $model_path1 --model_name $model_name1 \
#    --max_step $max_step --shuffle_before_sampling $shuffle_before_sampling  \
#    --model_type $model_type --max_num_atoms $max_num_atoms --len_features $len_features --n_dim $n_dim \
#    --cuda_device $cuda_device --num_threads $num_threads --each_class_size $each_class_size01 --exp EXP01
#python model_evaluation.py --data_dir $data_dir3 --save_dir $save_dir3 --model_path $model_path1 --model_name $model_name1 \
#    --max_step $max_step --shuffle_before_sampling $shuffle_before_sampling  \
#    --model_type $model_type --max_num_atoms $max_num_atoms --len_features $len_features --n_dim $n_dim \
#    --cuda_device $cuda_device --num_threads $num_threads --each_class_size $each_class_size01 --exp EXP01

#python model_evaluation.py --data_dir $data_dir1 --save_dir $save_dir1 --model_path $model_path2 --model_name $model_name2 \
#    --max_step $max_step --shuffle_before_sampling $shuffle_before_sampling  \
#    --model_type $model_type --max_num_atoms $max_num_atoms --len_features $len_features --n_dim $n_dim \
#    --cuda_device $cuda_device --num_threads $num_threads --each_class_size $each_class_size01 --exp EXP01
#python model_evaluation.py --data_dir $data_dir2 --save_dir $save_dir2 --model_path $model_path2 --model_name $model_name2 \
#    --max_step $max_step --shuffle_before_sampling $shuffle_before_sampling  \
#    --model_type $model_type --max_num_atoms $max_num_atoms --len_features $len_features --n_dim $n_dim \
#    --cuda_device $cuda_device --num_threads $num_threads --each_class_size $each_class_size01 --exp EXP01
#python model_evaluation.py --data_dir $data_dir3 --save_dir $save_dir3 --model_path $model_path2 --model_name $model_name2 \
#    --max_step $max_step --shuffle_before_sampling $shuffle_before_sampling  \
#    --model_type $model_type --max_num_atoms $max_num_atoms --len_features $len_features --n_dim $n_dim \
#    --cuda_device $cuda_device --num_threads $num_threads --each_class_size $each_class_size01 --exp EXP01

# For EXP02
python model_evaluation.py --data_dir $data_dir3 --save_dir $save_dir3 --model_path $model_path2 --model_name $model_name2 \
    --max_step $max_step --shuffle_before_sampling $shuffle_before_sampling  \
    --model_type $model_type --max_num_atoms $max_num_atoms --len_features $len_features --n_dim $n_dim \
    --cuda_device $cuda_device --num_threads $num_threads --each_class_size $each_class_size01 --exp EXP01

# For EXP03
#python model_evaluation.py --data_dir $data_dir4 --save_dir $save_dir4 --model_path $model_path2 --model_name $model_name2 \
#    --max_step $max_step --shuffle_before_sampling $shuffle_before_sampling  \
#    --model_type $model_type --max_num_atoms $max_num_atoms --len_features $len_features --n_dim $n_dim \
#    --cuda_device $cuda_device --num_threads $num_threads --each_class_size $each_class_size03 --exp EXP03
#python model_evaluation.py --data_dir $data_dir5 --save_dir $save_dir5 --model_path $model_path2 --model_name $model_name2 \
#    --max_step $max_step --shuffle_before_sampling $shuffle_before_sampling  \
#    --model_type $model_type --max_num_atoms $max_num_atoms --len_features $len_features --n_dim $n_dim \
#    --cuda_device $cuda_device --num_threads $num_threads --each_class_size $each_class_size03 --exp EXP03
#python model_evaluation.py --data_dir $data_dir6 --save_dir $save_dir6 --model_path $model_path2 --model_name $model_name2 \
#    --max_step $max_step --shuffle_before_sampling $shuffle_before_sampling  \
#    --model_type $model_type --max_num_atoms $max_num_atoms --len_features $len_features --n_dim $n_dim \
#    --cuda_device $cuda_device --num_threads $num_threads --each_class_size $each_class_size03 --exp EXP03

#!/bin/bash

#PBS -N EXP01_GAT
#PBS -l nodes=1:ppn=4:gnode2:gpu
#PBS -l walltime=7:00:00:00 
date

##### Run ##### 
cd /home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/
conda activate HWKim

export data_dir=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/exp1/ChEMBL/
export save_dir0=result_inference/Pub_ZINC_test_set

export save_dir4=result_inference/ARAE
export save_dir5=result_inference/GGM
export save_dir6=result_inference/drug_bank

export model_path1=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/result_train/PubChem_200k_GAT/GAT_30features/GAT_best_model_290.pt
export model_path2=/home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/result_train/Pub_200k_ZINC_100k_GAT/GAT_30features/GAT_best_model_281.pt

export model_name1=PubChem_200k_GAT
export model_name2=Pub_200k_ZINC_100k_GAT

# DI for data generation and inference, I for only inference 
#export mode=DI
export max_step=4
export shuffle_before_sampling=False
export max_num_atoms=64
export len_features=30
export n_dim=256
export cuda_device=1
export num_threads=4
export each_class_size01=2000
export each_class_size03=100000

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

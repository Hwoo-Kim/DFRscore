import time
import os
import sys

atom_feature='tot_degree'
#atom_feature='formal_charge'
num_conv_layers = [4,5,6]
fc_dims = [128,256]
datas= ['PubC', 'ZINC']
lrs = [1e-4,2e-4,4e-4]
i=0

for data in datas:
    if data != 'PubC':continue

    for lr in lrs:
        if lr == 4e-4: continue
        if lr == 1e-4:
            decay_epoch=200
        elif lr == 2e-4:
            decay_epoch=100
        elif lr == 4e-4:
            decay_epoch=0
    
        for num_conv_layer in num_conv_layers:
            #if num_conv_layer != 4: continue
    
            for fc_dim in fc_dims:
                #if fc_dim != 128: continue
                exp =  f'HWKim_{data}_'
                exp += f'{lr}_{atom_feature}_{num_conv_layer}_{fc_dim}'
                node = f'horus{18-(i//2)}'
                #node = f'horus{3+i}'
                
                lines = f"""#!/bin/bash
#PBS -N {exp}
#PBS -l nodes=1:ppn=7:{node}:gpus=1
#PBS -l walltime=7:00:00:00 
date

##### Run ##### 
source ~/hwkim/.bashrc
#cd /home/hwkim/simple_reaction_prediction/template_application/GitHub/model_train/
cd /home/wykgroup/hwkim/Synthesizability_for_VS/model_train/

data_dir1=data_for_train/PubChem_200k
data_dir2=data_for_train/Pub_200k_ZINC_100k 

save_name1=PubChem_200k_ring
save_name2=Pub_200k_ZINC_100k_ring

training_data_path={atom_feature}
data_preprocessing=False # True for data generation and training, False for only training
len_features=36
# Training parameters
lr={lr}
decay_epoch={decay_epoch}
n_conv_layer={num_conv_layer}
conv_dim=256
fc_dim={fc_dim}

#sleep {600*(i%2)}
"""
                if data=='PubC':
                    lines+=f"""python train.py --data_dir $data_dir1 --save_name $save_name1 \
--data_preprocessing $data_preprocessing --len_features $len_features --lr $lr --decay_epoch $decay_epoch \
--n_conv_layer $n_conv_layer --conv_dim $conv_dim --fc_dim $fc_dim \
--training_data_path $training_data_path
date"""
                elif data=='ZINC':
                    lines+=f"""python train.py --data_dir $data_dir2 --save_name $save_name2 \
--data_preprocessing $data_preprocessing --len_features $len_features --lr $lr --decay_epoch $decay_epoch \
--n_conv_layer $n_conv_layer --conv_dim $conv_dim --fc_dim $fc_dim \
--training_data_path $training_data_path
date"""

                with open("auto_train_script.sh", 'w') as w:
                    w.writelines(lines)
                i+=1
                os.system("qsub auto_train_script.sh")
        
                time.sleep(15)

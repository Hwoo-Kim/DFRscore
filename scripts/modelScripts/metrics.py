import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
import time
import argparse
from datetime import datetime
from scripts.utils import working_dir_setting
from sklearn.metrics import confusion_matrix

class ConfMatrix:
    def __init__(self, true_list, pred_list, labels=None, sample_weight=None, normalize=None):
        self.conf_matrix = confusion_matrix(
                y_true=true_list,
                y_pred=pred_list,
                labels=labels,
                sample_weight=sample_weight,    # same size with y_true and y_pred
                normalize=normalize     # {'true', 'pred', 'all'}
                )

    def get_numb_data(self):
        return np.sum(self.conf_matrix)

    def get_numb_pos(self, neg_label):
        return np.sum(self.conf_matrix[:neg_label,:]) + np.sum(self.conf_matrix[neg_label+1:,:])

    def get_numb_neg(self, neg_label):
        return np.sum(self.conf_matrix[neg_label,:])

    def get_numb_correct(self):
        return np.sum(self.conf_matrix.diagonal())

    def get_numb_false(self):
        return np.sum(self.conf_matrix) - np.sum(self.conf_matrix.diagonal())

    def get_class_precision(self, col_idx):
        column = self.conf_matrix[:,col_idx]
        return float(self.conf_matrix[col_idx,col_idx]/np.sum(column))

    def get_class_recall(self, row_idx):
        row = self.conf_matrix[row_idx,:]
        return float(self.conf_matrix[row_idx,row_idx]/np.sum(row))

class BinaryConfusionMatrix(ConfMatrix):
    def __init__(self, true_list, pred_list, neg_label):
        super().__init__(true_list = true_list, pred_list = pred_list)
        self.neg_label = int(neg_label)
        self.pos_label = int(1-neg_label)

    def get_accuracy(self):                         # for computing critical error. not explicit use.
        return float(self.get_numb_correct()/self.get_numb_data())

    def get_precision(self):             # for EXP01 and EXP03.
        return float(self.get_class_precision(col_idx=self.pos_label))

    def get_recall(self):
        return float(self.get_class_recall(row_idx=self.pos_label))

    def get_f1_score(self):
        prec, recall = self.get_precision(self.pos_label), self.get_recall(self.pos_label)
        return float(2/((1/prec)+(1/recall)))

    def get_critical_error(self):                   # for EXP01 and EXP03.
        return float(1-self.get_accuracy())

    def get_initial_pos_ratio(self):     # for EXP03.
        return float(self.get_numb_pos(self.neg_label)/self.get_numb_data())

    def get_ratio_change(self):          # for EXP03.
        initial_ratio = f'{int(self.get_numb_pos(self.neg_label))}:{int(self.get_numb_neg(self.neg_label))}'
        after_filtering_ratio = f'{int(self.conf_matrix[self.pos_label][self.pos_label])}:{int(self.conf_matrix[self.neg_label][self.pos_label])}'
        return ' >> '.join([initial_ratio, after_filtering_ratio])

    def get_filtering_ratio(self):       # for EXP03.
        neg_filtered_ratio = float(self.conf_matrix[self.neg_label,self.neg_label]/self.get_numb_neg(self.neg_label))
        pos_filtered_ratio = float(self.conf_matrix[self.pos_label,self.neg_label]/self.get_numb_pos(self.neg_label))
        tot_filtered_ratio = float(np.sum(self.conf_matrix[:,self.neg_label])/self.get_numb_data())
        return {'pos':pos_filtered_ratio, 'neg':neg_filtered_ratio, 'tot':tot_filtered_ratio}

class UnbalMultiConfusionMatrix(ConfMatrix):        # for EXP01, EXP03. MCCs.
    def __init__(self, true_list, pred_list, numb_classes):
        super().__init__(true_list = true_list, pred_list = pred_list)
        self.numb_classes = numb_classes

    def get_accuracy(self):     # = Micro-avg prec, Micro-avg recall, Micro-avg F1 score.
        return float(self.get_numb_correct()/self.get_numb_data())

    def get_macro_avg_precision(self):
        precs = np.array([self.get_class_precision(class_idx) for class_idx in range (self.numb_classes)])
        return float(np.sum(precs)/self.numb_classes)

    def get_macro_avg_recall(self):
        recalls = np.array([self.get_class_recall(class_idx) for class_idx in range (self.numb_classes)])
        return float(np.sum(recalls)/self.numb_classes)

    def get_macro_avg_f1_score(self):
        prec, recall = self.get_macro_avg_precision(), self.get_macro_avg_recall()
        return float(2/((1/prec)+(1/recall)))

def model_eval(data_dir, save_dir, max_step, threshold, model_path, exp):
    '''
    Arguments:
      data_dir: total_prob_list.pt. true_label
        True label starts from 1, and the negative label is 0.
        Predicted label also starts from 1, and the negative label is 0.
        Ex) in the case of max_step=4,
           negative class label: 0  /  positive class labels: 1,2,3,4.
      max_step: the maximum number of the steps used in retrosyn analysis.
    '''
    # 1. Reading model output data
    now = datetime.now()
    since_from = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    since = time.time()
    save_dir = working_dir_setting(save_dir, 'model_evaluation')
    result_file_name = f'{save_dir}/model_eval_result.txt'
    #only_result_inform= f'{save_dir}/only_important.txt'
    csv_file = f'{save_dir}/eval_metrics.csv'
    mcc_array_file = f'{save_dir}/mcc_confusion_matrix.txt'
    bin_array_file = f'{save_dir}/max_bin_confusion_matrix.txt'
    with open(f'{data_dir}/total_prob_list.pt','rb') as fr:
        data_list = torch.load(fr)
    true_label_list = []
    pred_label_list = []
    prob_list = []
    each_class_sizes = [0] * (max_step+1)
    for data in data_list:
        #smi_list.append(data[0])
        true_label_list.append(int(data[1]))            # int
        pred_label_list.append(int(data[2]))            # int
        prob_list.append(np.array(data[3]))             # data[3] --> torch obj.
        each_class_sizes[int(data[1])] +=1
    
    initial_num_data = len(true_label_list)
    true_label_list = np.array(true_label_list)
    pred_label_list = np.array(pred_label_list)
    neg_label = 0
    evaluation_setting_inform = [
                '----- Input Config Information -----\n',
                f'Since from: {since_from}\nmax_step: {max_step}\n',
                f'model_path: {model_path}\ndata_dir: {data_dir}\nsave_dir: {save_dir}\n',
                f'threshold(accumulate_bin_classfication): {threshold}\n{exp}\n',
                '\n----- Evaluation Result -----\n'
                ]
    for step in range(max_step+1):
        if step == 0:
            evaluation_setting_inform.append(f'Neg: {each_class_sizes[step]}\n')
        else:
            evaluation_setting_inform.append(f'Pos{step}: {each_class_sizes[step]}\n')

    for line in evaluation_setting_inform:
        print(line, end='')
    with open(result_file_name,'w') as fw:
        fw.writelines(evaluation_setting_inform)
    #with open(only_result_inform,'w') as fw:
        #fw.writelines(evaluation_setting_inform)

    # 1-1. Confusion matrix for MCC.
    MCC_conf_matrix = UnbalMultiConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list, numb_classes=max_step+1)

    # 1-2. Make Max-binary classification labels.
    pos_labels = [j+1 for j in list(range(max_step))]
    max_bin_true_label_list = []
    for label in true_label_list:
        if label in pos_labels:
            max_bin_true_label_list.append(1)
        else:
            max_bin_true_label_list.append(0)
    max_bin_pred_label_list = []
    for label in pred_label_list:
        if label in pos_labels:
            max_bin_pred_label_list.append(1)
        else:
            max_bin_pred_label_list.append(0)
    Max_bin_conf_matrix = BinaryConfusionMatrix(true_list=max_bin_true_label_list, pred_list=max_bin_pred_label_list, neg_label = neg_label)

    # 1-3. Make Accumulated-binary classification labels.
    # all the labels are from the raw probability output.
    pos_labels = [j+1 for j in list(range(max_step))]
    acc_bin_true_label_list = []
    for label in true_label_list:
        if label in pos_labels:
            acc_bin_true_label_list.append(1)
        else:
            acc_bin_true_label_list.append(0)
    acc_bin_pred_label_list = []
    for dist in prob_list:
        if dist[0] > threshold:         # distribution[0] = prob for neg.
            acc_bin_pred_label_list.append(0)
        else:
            acc_bin_pred_label_list.append(1)
    Acc_bin_conf_matrix = BinaryConfusionMatrix(true_list=acc_bin_true_label_list, pred_list=acc_bin_pred_label_list, neg_label=neg_label)


    # 2. Evaluation metrics calculation
    if exp == 'EXP01':
        # 2-1. EXP01
        mcc_acc, macro_avg_precision, macro_avg_f1_score = \
                MCC_conf_matrix.get_accuracy(), MCC_conf_matrix.get_macro_avg_precision(), MCC_conf_matrix.get_macro_avg_f1_score()
        max_bin_acc, max_bin_prec, max_bin_critical_err = \
                Max_bin_conf_matrix.get_accuracy(), Max_bin_conf_matrix.get_precision(), Max_bin_conf_matrix.get_critical_error()
        acc_bin_acc, acc_bin_prec, acc_bin_critical_err = \
                Acc_bin_conf_matrix.get_accuracy(), Acc_bin_conf_matrix.get_precision(), Acc_bin_conf_matrix.get_critical_error()
        result_dict = {
                'mcc_acc': mcc_acc, 'macro_avg_precision': macro_avg_precision, 'macro_avg_f1_score': macro_avg_f1_score,
                'max_bin_acc': max_bin_acc, 'max_bin_prec': max_bin_prec, 'max_bin_critical_err': max_bin_critical_err,
                'acc_bin_acc': acc_bin_acc, 'acc_bin_prec': acc_bin_prec, 'acc_bin_critical_err': acc_bin_critical_err 
                }
        with open(result_file_name,'a') as fw:
            fw.write('\n***** Multi Class Classification *****\n')
            fw.write(f'MCC_accuracy: {mcc_acc:.3f}\nMacro_avg_precision: {macro_avg_precision:.3f}\nMacro_avg_f1_score: {macro_avg_f1_score:.3f}\n')
            fw.write('\n***** Max Binary Classification *****\n')
            fw.write(f'Bin_acc: {max_bin_acc:.3f}\nBin_prec: {max_bin_prec:.3f}\nCritical_error: {max_bin_critical_err:.3f}\n')
            fw.write('\n***** Accumulated Binary Classification *****\n')
            fw.write(f'Bin_acc: {acc_bin_acc:.3f}\nBin_prec: {acc_bin_prec:.3f}\nCritical_error: {acc_bin_critical_err:.3f}\n')
        print('\n***** Multi Class Classification *****')
        print(f'MCC_accuracy: {mcc_acc:.3f}\nMacro_avg_precision: {macro_avg_precision:.3f}\nMacro_avg_f1_score: {macro_avg_f1_score:.3f}')
        print('\n***** Max Binary Classification *****')
        print(f'Bin_acc: {max_bin_acc:.3f}\nBin_prec: {max_bin_prec:.3f}\nCritical_error: {max_bin_critical_err:.3f}')
        print('\n***** Accumulated Binary Classification *****')
        print(f'Bin_acc: {acc_bin_acc:.3f}\nBin_prec: {acc_bin_prec:.3f}\nCritical_error: {acc_bin_critical_err:.3f}')
        
        first_row, second_row = [], []
        for key, val in result_dict.items():
            first_row.append(key)
            second_row.append(val)
        with open(csv_file, 'w', newline='') as fw:
            wr=csv.writer(fw)
            wr.writerow(first_row)
            wr.writerow(second_row)
        np.savetxt(mcc_array_file, MCC_conf_matrix.conf_matrix, fmt='%.0f')
        np.savetxt(bin_array_file, Max_bin_conf_matrix.conf_matrix, fmt='%.0f')

    elif exp == 'EXP03':
        # 2-2. EXP03
        mcc_acc, macro_avg_precision, macro_avg_f1_score = \
                MCC_conf_matrix.get_accuracy(), MCC_conf_matrix.get_macro_avg_precision(), MCC_conf_matrix.get_macro_avg_f1_score()
        init_pos_ratio, ratio_change, filtering_ratio, = \
                Max_bin_conf_matrix.get_initial_pos_ratio(), Max_bin_conf_matrix.get_ratio_change(), Max_bin_conf_matrix.get_filtering_ratio()
                #Acc_bin_conf_matrix.get_initial_pos_ratio(), Acc_bin_conf_matrix.get_ratio_change(), Acc_bin_conf_matrix.get_filtering_ratio()
        bin_precision, critical_err = Max_bin_conf_matrix.get_precision(), Max_bin_conf_matrix.get_critical_error()
        #bin_precision, critical_err = Acc_bin_conf_matrix.get_precision(), Acc_bin_conf_matrix.get_critical_error()
        result_dict = {
                'mcc_acc': mcc_acc, 'macro_avg_precision': macro_avg_precision, 'macro_avg_f1_score': macro_avg_f1_score,
                'init_pos_ratio': init_pos_ratio, 'ratio_change (pos:neg)': ratio_change, 'critical_err': critical_err,
                'filtered_out_ratio_pos': filtering_ratio['pos'],'filtered_out_ratio_neg': filtering_ratio['neg'],'filtered_out_ratio_tot': filtering_ratio['tot']
                }
        with open(result_file_name,'a') as fw:
            fw.write('\n***** Multi Class Classification *****\n')
            fw.write(f'MCC_accuracy: {mcc_acc:.3f}\nMacro_avg_precision: {macro_avg_precision:.3f}\nMacro_avg_f1_score: {macro_avg_f1_score:.3f}\n')
            fw.write(f'Critical_error: {critical_err:.3f}\n')
            fw.write('\n***** Max Binary Classification *****\n')
            fw.write(f'Bin_precision: {bin_precision:.3f}\nInit_pos_ratio: {init_pos_ratio:.3f}\n')
            fw.write(f'ratio_change (pos:neg): {ratio_change}\nfiltered_out_ratio_pos: {filtering_ratio["pos"]:.3f}\n')
            fw.write(f'filtered_out_ratio_neg: {filtering_ratio["neg"]:.3f}\nfiltered_out_ratio_tot: {filtering_ratio["tot"]:.3f}\n')
        print('\n***** Multi Class Classification *****')
        print(f'MCC_accuracy: {mcc_acc:.3f}\nMacro_avg_precision: {macro_avg_precision:.3f}\nMacro_avg_f1_score: {macro_avg_f1_score:.3f}')
        print(f'Critical_error: {critical_err:.3f}')
        print('\n***** Max Binary Classification *****')
        print(f'Bin_precision: {bin_precision:.3f}\nInit_pos_ratio: {init_pos_ratio:.3f}')
        print(f'ratio_change (pos:neg): {ratio_change}\nfiltered_out_ratio_pos: {filtering_ratio["pos"]:.3f}')
        print(f'filtered_out_ratio_neg: {filtering_ratio["neg"]:.3f}\nfiltered_out_ratio_tot: {filtering_ratio["tot"]:.3f}')

        first_row, second_row = [], []
        for key, val in result_dict.items():
            first_row.append(key)
            second_row.append(val)
        with open(csv_file, 'w', newline='') as fw:
            wr=csv.writer(fw)
            wr.writerow(first_row)
            wr.writerow(second_row)
        np.savetxt(mcc_array_file, MCC_conf_matrix.conf_matrix, fmt='%.0f')
        np.savetxt(bin_array_file, Max_bin_conf_matrix.conf_matrix, fmt='%.0f')

    return True

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',help='data directory',type=str)
    parser.add_argument('--threshold',help='data directory',type=str)
    args = parser.parse_args()
    data_directory = args.data_dir
    model_eval(data_directory, max_step=4, threshold=float(args.threshold))

    '''
    a = BinaryConfusionMatrix(
            true_list = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
            pred_list = [0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1],
            neg_label = 0
            )
    
    print(a.conf_matrix)
    print(a.get_initial_pos_ratio())
    print(a.get_ratio_change())
    print(a.get_filtering_ratio())
    '''

import sys, os
import pickle
from multiprocessing import Pool
sys.path.append(f'{os.path.abspath(os.path.dirname(__file__))}')
sys.path.append(f'{os.path.dirname(os.path.abspath(os.path.dirname(__file__)))}')
from metrics import BinaryConfusionMatrix, UnbalMultiConfusionMatrix, get_AUROC
from getScores import getSCScore, getSAScore, rescale_score

import numpy as np
import csv


def runExp01(predictor, max_step, save_dir, test_file_path, logger, each_class_sizes=None):
    '''
    Arguments:
      predictor: SVS object already restored by trained model.
      num_class: the number of classes. equals to max_step+1.
      test_smi_list: list of list. [neg, pos1, pos2, ...]
        length = num_class
        Negative samples comes from first (idx=0), positive samples start from the next(idx>=1).
      save_dir: the directory where evaluation result will be saved.
    '''
    # 0. reading test files
    if each_class_sizes is None:
        each_class_sizes = [None for i in range(max_step+1)]
    test_smi_list, true_smis, false_smis = [], [], []
    for i in range(max_step+1):
        if i ==0:
            with open(os.path.join(test_file_path, f'neg{max_step}.smi'), 'r') as fr:
                if each_class_sizes[i]:
                    data = fr.read().splitlines()[:each_class_sizes[i]]
                    each_class_sizes[i] = len(data)
                else:
                    data = fr.read().splitlines()
                    each_class_sizes[i] = len(data)
                #locals()[f'neg{max_step}'] = data
                false_smis += data
        else:
            with open(os.path.join(test_file_path, f'pos{i}.smi'), 'r') as fr:
                if each_class_sizes[i]:
                    data = fr.read().splitlines()[:each_class_sizes[i]]
                    each_class_sizes[i] = len(data)
                else:
                    data = fr.read().splitlines()
                    each_class_sizes[i] = len(data)
                #locals()[f'pos{i}'] = data
                true_smis += data
        test_smi_list += data

    # 1. result save path setting.
    csv_file = os.path.join(save_dir, 'eval_metrics.csv')
    mcc_array_file= os.path.join(save_dir, 'mcc_confusion_matrix.txt')
    bin_array_file = os.path.join(save_dir, 'max_bin_confusion_matrix.txt')

    # 2. BCC! - comparison with SA and SC
    for step in range(max_step+1):
        if step == 0:
            logger(f'  Neg: {each_class_sizes[step]}')
        else:
            logger(f'  Pos{step}: {each_class_sizes[step]}')

    # 2-1. get scores SA score, SC score, and SVS
    # SA score and SC score were already rescaled into [0, 1].
    logger('\n----- BCC Evaluation -----')
    logger('  Calculating scores...')
    logger('  calculating SA score...', end='\t')
    #p = Pool(16)
    #SAScores = p.map_async(getSAScore, test_smi_list)
    #SAScores.wait()
    #p.close()
    #p.join()
    #SAScores=SAScores.get()
    SAScores = np.array([0 for i in range(len(test_smi_list))]).astype(float)
    logger('  Done.')
    logger('  calculating SC score...', end='\t')
    SCScores = getSCScore(test_smi_list)
    logger('  Done.')
    logger('  calculating SVS...', end='\t')
    SVSs = predictor.smiListToScores(test_smi_list)
    SVS_probs = predictor.smiListToScores(test_smi_list, get_probs=True)
    logger('  Done.')

    # 2-2. Setting for BCC test.
    true_label_list = []
    for idx, l in enumerate(each_class_sizes):
        if idx ==0:
            true_label_list += [0 for i in range(l)]
        else:
            true_label_list += [1 for i in range(l)]
    true_label_list = np.array(true_label_list)

    # 2-2-1. Extract Max Probability from raw probability result.
    pred_label_list1 = list((np.argmax(SVS_probs, axis=1)!=0).astype(int))

    # 2-2-2. Extract Accumulated Probability from raw probability result.
    pred_label_list2 = list((SVS_probs[:,0]<0.5).astype(int))

    # 2-2-3. Using rescaled score from SVSs
    pred_label_list3 = list((rescale_score(SVSs, m=1, M=max_step+1, reverse=True)>0.5).astype(int))

    # 2-3. Confusion matrix and evaluate metrics
    Max_bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list1, neg_label = 0)
    Acc_bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list2, neg_label = 0)
    Scalar_bin = BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list3, neg_label = 0)
    logger('1. Max Probability')
    max_bin_acc, max_bin_prec, max_bin_critical_err = \
            Max_bin_conf_matrix.get_accuracy(), Max_bin_conf_matrix.get_precision(), Max_bin_conf_matrix.get_critical_error()
    logger('  Bin_acc, Bin_prec, Bin_critical_err, auroc =')
    logger(f'  {max_bin_acc}, {max_bin_prec}, {max_bin_critical_err}', end=', ')
    logger(get_AUROC(true_label_list, pred_label_list1))

    logger('2. Accumulated Probability')
    acc_bin_acc, acc_bin_prec, acc_bin_critical_err = \
            Acc_bin_conf_matrix.get_accuracy(), Acc_bin_conf_matrix.get_precision(), Acc_bin_conf_matrix.get_critical_error()
    logger('  Bin_acc, Bin_prec, Bin_critical_err, auroc =')
    logger(f'  {acc_bin_acc}, {acc_bin_prec}, {acc_bin_critical_err}', end=', ')
    logger(get_AUROC(true_label_list, pred_label_list2))

    logger('3. Scalar score')
    scalar_bin_acc, scalar_bin_prec, scalar_bin_critical_err = \
            Scalar_bin.get_accuracy(), Scalar_bin.get_precision(), Scalar_bin.get_critical_error()
    logger('  Bin_acc, Bin_prec, Bin_critical_err, auroc =')
    logger(f'  {scalar_bin_acc}, {scalar_bin_prec}, {scalar_bin_critical_err}', end=', ' )
    logger(get_AUROC(true_label_list, pred_label_list3))

    # 2-4. SA score and SC score
    logger('4. SA score')
    sas_auroc = get_AUROC(true_label_list, SAScores)
    logger(f'  auroc = {sas_auroc}')

    logger('5. SC score')
    scs_auroc = get_AUROC(true_label_list, SCScores)
    logger(f'  auroc = {scs_auroc}')

    # 3. Setting for MCC test.
    logger('\n----- MCC Evaluation -----')
    pred_label_list = list(np.argmax(SVS_probs, axis=1).astype(int))
    MCC_conf_matrix = UnbalMultiConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list, numb_classes=max_step+1)
    mcc_acc, macro_avg_precision, macro_avg_f1_score = \
        MCC_conf_matrix.get_accuracy(), MCC_conf_matrix.get_macro_avg_precision(), MCC_conf_matrix.get_macro_avg_f1_score()
    logger('  mcc_acc, macro_avg_precision, macro_avg_f1_score = ')
    logger(f'  {mcc_acc}, {macro_avg_precision}, {macro_avg_f1_score}')

    result_dict = {
            'max_bin_acc': max_bin_acc, 'max_bin_prec': max_bin_prec, 'max_bin_critical_err': max_bin_critical_err,
            'acc_bin_acc': acc_bin_acc, 'acc_bin_prec': acc_bin_prec, 'acc_bin_critical_err': acc_bin_critical_err,
            'scalar_bin_acc': scalar_bin_acc, 'scalar_bin_prec': scalar_bin_prec, 'scalar_bin_critical_err': scalar_bin_critical_err,
            'mcc_acc': mcc_acc, 'macro_avg_precision': macro_avg_precision, 'macro_avg_f1_score': macro_avg_f1_score,
            'sa_auroc': sas_auroc, 'sc_auroc':scs_auroc
            }

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

def runExp03(predictor, max_step, save_dir, test_file_path, logger, each_class_sizes=None):
    '''
    Arguments:
      predictor: SVS object already restored by trained model.
      num_class: the number of classes. equals to max_step+1.
      test_smi_list: list of list. [neg, pos1, pos2, ...]
        length = num_class
        Negative samples comes from first (idx=0), positive samples start from the next(idx>=1).
      save_dir: the directory where evaluation result will be saved.
    '''
    # 0. reading test files
    if each_class_sizes is None:
        each_class_sizes = [None for i in range(max_step+1)]
    test_smi_list, true_smis, false_smis = [], [], []
    for i in range(max_step+1):
        if i ==0:
            with open(os.path.join(test_file_path, f'neg{max_step}.smi'), 'r') as fr:
                if each_class_sizes[i]:
                    data = fr.read().splitlines()[:each_class_sizes[i]]
                    each_class_sizes[i] = len(data)
                else:
                    data = fr.read().splitlines()
                    each_class_sizes[i] = len(data)
                #locals()[f'neg{max_step}'] = data
                false_smis += data
        else:
            with open(os.path.join(test_file_path, f'pos{i}.smi'), 'r') as fr:
                if each_class_sizes[i]:
                    data = fr.read().splitlines()[:each_class_sizes[i]]
                    each_class_sizes[i] = len(data)
                else:
                    data = fr.read().splitlines()
                    each_class_sizes[i] = len(data)
                #locals()[f'pos{i}'] = data
                true_smis += data
        test_smi_list += data

    # 1. result save path setting.
    csv_file = os.path.join(save_dir, 'eval_metrics.csv')
    mcc_array_file= os.path.join(save_dir, 'mcc_confusion_matrix.txt')
    bin_array_file = os.path.join(save_dir, 'max_bin_confusion_matrix.txt')

    # 2. BCC! - comparison with SA and SC
    for step in range(max_step+1):
        if step == 0:
            logger(f'  Neg: {each_class_sizes[step]}')
        else:
            logger(f'  Pos{step}: {each_class_sizes[step]}')

    # 2-1. get scores SA score, SC score, and SVS
    # SA score and SC score were already rescaled into [0, 1].
    logger('\n----- BCC Evaluation -----')
    logger('  Calculating scores...')
    logger('  calculating SA score...', end='\t')
    p = Pool(16)
    SAScores = p.map_async(getSAScore, test_smi_list)
    SAScores.wait()
    p.close()
    p.join()
    SAScores=SAScores.get()
    #SAScores = np.array([0 for i in range(len(test_smi_list))]).astype(float)
    logger('  Done.')
    logger('  calculating SC score...', end='\t')
    SCScores = getSCScore(test_smi_list)
    logger('  Done.')
    logger('  calculating SVS...', end='\t')
    SVSs = predictor.smiListToScores(test_smi_list)
    SVS_probs = predictor.smiListToScores(test_smi_list, get_probs=True)
    logger('  Done.')

    # 2-2. Setting for BCC test.
    true_label_list = []
    for idx, l in enumerate(each_class_sizes):
        if idx ==0:
            true_label_list += [0 for i in range(l)]
        else:
            true_label_list += [1 for i in range(l)]
    true_label_list = np.array(true_label_list)

    # 2-2-1. Extract Max Probability from raw probability result.
    pred_label_list1 = list((np.argmax(SVS_probs, axis=1)!=0).astype(int))

    # 2-2-2. Extract Accumulated Probability from raw probability result.
    pred_label_list2 = list((SVS_probs[:,0]<0.5).astype(int))

    # 2-2-3. Using rescaled score from SVSs
    pred_label_list3 = list((rescale_score(SVSs, m=1, M=max_step+1, reverse=True)>0.5).astype(int))

    # 2-3. Confusion matrix and evaluate metrics
    Max_bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list1, neg_label = 0)
    Acc_bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list2, neg_label = 0)
    Scalar_bin = BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list3, neg_label = 0)
    logger('1. Max Probability')
    max_bin_acc, max_bin_prec, max_bin_critical_err = \
            Max_bin_conf_matrix.get_accuracy(), Max_bin_conf_matrix.get_precision(), Max_bin_conf_matrix.get_critical_error()
    logger('  Bin_acc, Bin_prec, Bin_critical_err, auroc =')
    logger(f'  {max_bin_acc}, {max_bin_prec}, {max_bin_critical_err}', end=', ')
    logger(get_AUROC(true_label_list, pred_label_list1))

    logger('2. Accumulated Probability')
    acc_bin_acc, acc_bin_prec, acc_bin_critical_err = \
            Acc_bin_conf_matrix.get_accuracy(), Acc_bin_conf_matrix.get_precision(), Acc_bin_conf_matrix.get_critical_error()
    logger('  Bin_acc, Bin_prec, Bin_critical_err, auroc =')
    logger(f'  {acc_bin_acc}, {acc_bin_prec}, {acc_bin_critical_err}', end=', ')
    logger(get_AUROC(true_label_list, pred_label_list2))

    logger('3. Scalar score')
    scalar_bin_acc, scalar_bin_prec, scalar_bin_critical_err = \
            Scalar_bin.get_accuracy(), Scalar_bin.get_precision(), Scalar_bin.get_critical_error()
    logger('  Bin_acc, Bin_prec, Bin_critical_err, auroc =')
    logger(f'  {scalar_bin_acc}, {scalar_bin_prec}, {scalar_bin_critical_err}', end=', ' )
    logger(get_AUROC(true_label_list, pred_label_list3))

    # 2-4. SA score and SC score
    logger('4. SA score')
    sas_auroc = get_AUROC(true_label_list, SAScores)
    logger(f'  auroc = {sas_auroc}')

    logger('5. SC score')
    scs_auroc = get_AUROC(true_label_list, SCScores)
    logger(f'  auroc = {scs_auroc}')

    # 3. Setting for MCC test.
    logger('\n----- MCC Evaluation -----')
    pred_label_list = list(np.argmax(SVS_probs, axis=1).astype(int))
    MCC_conf_matrix = UnbalMultiConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list, numb_classes=max_step+1)
    mcc_acc, macro_avg_precision, macro_avg_f1_score = \
        MCC_conf_matrix.get_accuracy(), MCC_conf_matrix.get_macro_avg_precision(), MCC_conf_matrix.get_macro_avg_f1_score()
    logger('  mcc_acc, macro_avg_precision, macro_avg_f1_score = ')
    logger(f'  {mcc_acc}, {macro_avg_precision}, {macro_avg_f1_score}')

    result_dict = {
            'max_bin_acc': max_bin_acc, 'max_bin_prec': max_bin_prec, 'max_bin_critical_err': max_bin_critical_err,
            'acc_bin_acc': acc_bin_acc, 'acc_bin_prec': acc_bin_prec, 'acc_bin_critical_err': acc_bin_critical_err,
            'scalar_bin_acc': scalar_bin_acc, 'scalar_bin_prec': scalar_bin_prec, 'scalar_bin_critical_err': scalar_bin_critical_err,
            'mcc_acc': mcc_acc, 'macro_avg_precision': macro_avg_precision, 'macro_avg_f1_score': macro_avg_f1_score,
            'sa_auroc': sas_auroc, 'sc_auroc':scs_auroc
            }

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

    if exp == 'EXP03':
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

import sys, os
import pickle
from multiprocessing import Pool
from .metrics import BinaryConfusionMatrix, UnbalMultiConfusionMatrix, get_AUROC
sys.path.append(f'{os.path.dirname(os.path.abspath(os.path.dirname(__file__)))}')
from getScores import getSCScore, getSAScore, rescale_score

import numpy as np
import csv


def runExp01(predictor,
        max_step,
        save_dir,
        test_file_path,
        logger,
        each_class_sizes=None,
        num_cores=4
        ):
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
    if test_file_path[-4:] == '.pkl':
        with open(test_file_path, 'rb') as fr:
            test_data=pickle.load(fr)['test']
    else:
        test_data = dict()
        for i in range(max_step+1):
            if i == 0:
                with open(os.path.join(test_file_path, f'neg{max_step}.smi'), 'r') as fr:
                    test_data[0] = fr.read().splitlines()
            else:
                with open(os.path.join(test_file_path, f'pos{i}.smi'), 'r') as fr:
                    test_data[i] = fr.read().splitlines()

    for i in range(max_step+1):
        if each_class_sizes[i]:
            try:
                data = test_data[i][:each_class_sizes[i]]
                each_class_sizes[i] = len(data)
            except:
                data = test_data[max_step+1][:each_class_sizes[i]]
                each_class_sizes[i] = len(data)
        else:
            try:
                data = test_data[i][:each_class_sizes[i]]
                each_class_sizes[i] = len(data)
            except:
                data = test_data[max_step+1][:each_class_sizes[i]]
                each_class_sizes[i] = len(data)
        test_smi_list += data

    # 1. result save path setting.
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    csv_file = os.path.join(save_dir, 'eval_metrics.csv')
    mcc_array_file= os.path.join(save_dir, 'mcc_confusion_matrix.txt')
    bin_array_file = os.path.join(save_dir, 'bin_confusion_matrix.txt')

    # 2. BCC! - comparison with SA and SC
    for step in range(max_step+1):
        if step == 0:
            logger(f'  Neg: {each_class_sizes[step]}')
        else:
            logger(f'  Pos{step}: {each_class_sizes[step]}')

    # 2-1. get scores SA score, SC score, and SVS
    # SA score and SC score were already rescaled into [0, 1].
    logger('\n  Calculating scores...')
    logger('  calculating SA score... (not rescaled!)', end='\t')
    p = Pool(num_cores)
    SAScores = p.map_async(getSAScore, test_smi_list)
    SAScores.wait()
    p.close()
    p.join()
    SAScores=SAScores.get()
    #SAScores = np.array([0 for i in range(len(test_smi_list))]).astype(float)
    logger('  Done.')
    logger('  calculating SC score... (not rescaled!)', end='\t')
    SCScores = getSCScore(test_smi_list)
    logger('  Done.')
    logger('  calculating SVS...', end='\t')
    SVSs = predictor.smiListToScores(test_smi_list)
    logger('  Done.')

    # 2-2. Setting for BCC test.
    true_label_list = []
    for idx, l in enumerate(each_class_sizes):
        if idx ==0:
            true_label_list += [0 for i in range(l)]
        else:
            true_label_list += [1 for i in range(l)]
    true_label_list = np.array(true_label_list)

    # 2-2. Use our model as as binary classification model.
    # Threshold must be max_step+0.5
    threshold = max_step+0.5
    bin_label_list = (SVSs<threshold).astype(int)

    # 2-3. Confusion matrix and evaluate metrics
    logger('\n  ----- BCC Evaluation -----')
    bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=bin_label_list, neg_label = 0)
    logger('  1. Our model(SVS)')
    bin_acc, bin_prec, bin_recall, bin_critical_err = \
            bin_conf_matrix.get_accuracy(), bin_conf_matrix.get_precision(), bin_conf_matrix.get_recall(), bin_conf_matrix.get_critical_error()
    logger('   Bin_acc, Bin_prec, Bin_recall, Bin_critical_err, auroc =')
    logger(f'    {bin_acc}, {bin_prec}, {bin_recall},{bin_critical_err}', end=', ')
    logger(get_AUROC(true_label_list, -1*np.array(SVSs)))
    logger('    auroc was calculated by reversed score.')

    # 2-4. SA score and SC score
    logger('  2. SA score')
    sas_auroc = get_AUROC(true_label_list, -1*np.array(SAScores))
    logger(f'   auroc = {sas_auroc}, auroc was calculated by reversed score.')

    logger('  3. SC score')
    scs_auroc = get_AUROC(true_label_list, -1*np.array(SCScores))
    logger(f'   auroc = {scs_auroc}, auroc was calculated by reversed score.')

    # 3. Setting for MCC test.
    true_label_list = []
    for idx, l in enumerate(each_class_sizes):
        true_label_list += [idx for i in range(l)]
    true_label_list = np.array(true_label_list)

    SVSs_for_each_class = dict()
    SAScores_for_each_class = dict()
    SCScores_for_each_class = dict()
    index = 0
    for i in range(len(each_class_sizes)):
        n = each_class_sizes[i]
        key = str(i)
        SVSs_for_each_class[key] = SVSs[index:index + n]
        SAScores_for_each_class[key] = SAScores[index:index + n]
        SCScores_for_each_class[key] = SCScores[index:index + n]
        index += n
    datas = dict()
    datas['sa'] = SAScores_for_each_class
    datas['sc'] = SCScores_for_each_class
    datas['svs'] = SVSs_for_each_class

    # Save pickle files
    with open(os.path.join(save_dir,'scores.pkl'),'wb') as f:
        pickle.dump(datas,f)

    logger('\n  ----- MCC Evaluation -----')
    pred_label_list = np.around(np.where(SVSs>max_step+1, max_step+1, SVSs))
    pred_label_list = np.where(pred_label_list==float(max_step+1), 0, pred_label_list)
    MCC_conf_matrix = UnbalMultiConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list, numb_classes=max_step+1)
    mcc_acc, macro_avg_precision, macro_avg_recall, macro_avg_f1_score = \
        MCC_conf_matrix.get_accuracy(), MCC_conf_matrix.get_macro_avg_precision(), \
        MCC_conf_matrix.get_macro_avg_recall(), MCC_conf_matrix.get_macro_avg_f1_score()
    logger('  mcc_acc, macro_avg_precision, macro_avg_recall, macro_avg_f1_score = ')
    logger(f'  {mcc_acc}, {macro_avg_precision}, {macro_avg_recall}, {macro_avg_f1_score}')

    bin_result_dict = {
            'bin_acc': bin_acc, 'bin_prec': bin_prec, 'bin_recall': bin_recall,'bin_critical_err': bin_critical_err,
            }
    mcc_result_dict = {
            'mcc_acc': mcc_acc, 'macro_avg_precision': macro_avg_precision, 'macro_avg_recall': macro_avg_recall,
            'macro_avg_f1_score': macro_avg_f1_score, 
            }

    with open(csv_file, 'w', newline='') as fw:
        wr=csv.writer(fw)
        wr.writerow(bin_result_dict.keys())
        wr.writerow(bin_result_dict.values())
        wr.writerow([])
        wr.writerow(mcc_result_dict.keys())
        wr.writerow(mcc_result_dict.values())
    np.savetxt(mcc_array_file, MCC_conf_matrix.conf_matrix, fmt='%.0f')
    np.savetxt(bin_array_file, bin_conf_matrix.conf_matrix, fmt='%.0f')

    return True

def runExp03(predictor,
        max_step, 
        save_dir, 
        test_file_path,
        logger
        ):
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
    each_class_sizes = []
    test_smi_list, true_smis, false_smis = [], [], []
    test_data = dict()
    for i in range(max_step+1):
        if i == 0:
            with open(os.path.join(test_file_path, f'neg{max_step}.smi'), 'r') as fr:
                test_data[0] = fr.read().splitlines()
        else:
            with open(os.path.join(test_file_path, f'pos{i}.smi'), 'r') as fr:
                test_data[i] = fr.read().splitlines()

    for i in range(max_step+1):
        data = test_data[i]
        each_class_sizes.append(len(data))
        test_smi_list += data

    # 1. result save path setting.
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    csv_file = os.path.join(save_dir, 'eval_metrics.csv')
    mcc_array_file= os.path.join(save_dir, 'mcc_confusion_matrix.txt')
    bin_array_file = os.path.join(save_dir, 'bin_confusion_matrix.txt')

    # 2. As a binary classifier
    for step in range(max_step+1):
        if step == 0:
            logger(f'  Neg: {each_class_sizes[step]}')
        else:
            logger(f'  Pos{step}: {each_class_sizes[step]}')

    # 2-1. get scores, SVS
    logger('\n  Calculating scores...')
    logger('  calculating SVS...', end='\t')
    SVSs = predictor.smiListToScores(test_smi_list)
    logger('  Done.')

    # 2-2. Setting for BCC test.
    true_label_list = []
    for idx, l in enumerate(each_class_sizes):
        if idx ==0:
            true_label_list += [0 for i in range(l)]
        else:
            true_label_list += [1 for i in range(l)]
    true_label_list = np.array(true_label_list)

    # 2-2. Use our model as as binary classification model.
    # Threshold must be max_step+0.5
    threshold = max_step+0.5
    bin_label_list = (SVSs<threshold).astype(int)

    # 2-3. Confusion matrix and evaluate metrics
    logger('\n  ----- BCC Evaluation -----')
    bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=bin_label_list, neg_label = 0)
    bin_acc, bin_prec, bin_recall, bin_critical_err = \
            bin_conf_matrix.get_accuracy(), bin_conf_matrix.get_precision(), bin_conf_matrix.get_recall(), bin_conf_matrix.get_critical_error()
    init_pos_ratio, ratio_change, filtering_ratio, = \
            bin_conf_matrix.get_initial_pos_ratio(), bin_conf_matrix.get_ratio_change(), bin_conf_matrix.get_filtering_ratio()
    logger('   Bin_acc, Bin_prec, Bin_recall, Bin_critical_err, auroc =')
    logger(f'    {bin_acc}, {bin_prec}, {bin_recall}, {bin_critical_err}', end=', ')
    logger(get_AUROC(true_label_list, SVSs))
    logger('   Init_pos_ratio, Ratio_change =')
    logger(f'    {init_pos_ratio}, {ratio_change}')
    logger('   filtering_ratio =')
    logger(f"    Pos: {filtering_ratio['pos']}, Neg: {filtering_ratio['neg']}, Total: {filtering_ratio['tot']}")

    # 3. Setting for MCC test.
    true_label_list = []
    for idx, l in enumerate(each_class_sizes):
        true_label_list += [idx for i in range(l)]
    true_label_list = np.array(true_label_list)

    logger('\n  ----- MCC Evaluation -----')
    pred_label_list = np.around(np.where(SVSs>max_step+1, max_step+1, SVSs))
    pred_label_list = np.where(pred_label_list==float(max_step+1), 0, pred_label_list)
    MCC_conf_matrix = UnbalMultiConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list, numb_classes=max_step+1)
    mcc_acc, macro_avg_precision, macro_avg_recall, macro_avg_f1_score = \
        MCC_conf_matrix.get_accuracy(), MCC_conf_matrix.get_macro_avg_precision(), MCC_conf_matrix.get_macro_avg_recall(), MCC_conf_matrix.get_macro_avg_f1_score()
    logger('  mcc_acc, macro_avg_precision, macro_avg_recall, macro_avg_f1_score = ')
    logger(f'   {mcc_acc}, {macro_avg_precision}, {macro_avg_recall}, {macro_avg_f1_score}')

    mcc_result_dict = {
            'mcc_acc': mcc_acc, 'macro_avg_precision': macro_avg_precision,'macro_avg_recall': macro_avg_recall, 'macro_avg_f1_score': macro_avg_f1_score
            }

    bin_result_dict = {
            'bin_acc': bin_acc, 'bin_precision': bin_prec, 'bin_recall': bin_recall,
            'init_pos_ratio': init_pos_ratio, 'ratio_change (pos:neg)': ratio_change, 'critical_err': 1-bin_acc,
            'filtered_out_ratio_pos': filtering_ratio['pos'],'filtered_out_ratio_neg': filtering_ratio['neg'],'filtered_out_ratio_total': filtering_ratio['tot']
            }

    with open(csv_file, 'w', newline='') as fw:
        wr=csv.writer(fw)
        wr.writerow(bin_result_dict.keys())
        wr.writerow(bin_result_dict.values())
        wr.writerow([])
        wr.writerow(mcc_result_dict.keys())
        wr.writerow(mcc_result_dict.values())
    np.savetxt(mcc_array_file, MCC_conf_matrix.conf_matrix, fmt='%.0f')
    np.savetxt(bin_array_file, bin_conf_matrix.conf_matrix, fmt='%.0f')

    return True

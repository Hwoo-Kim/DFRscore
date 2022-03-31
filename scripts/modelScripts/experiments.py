import sys, os
import pickle
from multiprocessing import Pool
sys.path.append(f'{os.path.abspath(os.path.dirname(__file__))}')
sys.path.append(f'{os.path.dirname(os.path.abspath(os.path.dirname(__file__)))}')
from metrics import BinaryConfusionMatrix, UnbalMultiConfusionMatrix, get_AUROC
from getScores import getSCScore, getSAScore, rescale_score

import numpy as np
import csv


def runExp01(predictor,
        problem,
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
    save_dir = f'{save_dir}_{problem}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
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
    if problem == 'classification':
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

    if problem == 'classification':
        # 2-2-1. Extract Max Probability from raw probability result.
        pred_label_list1 = (np.argmax(SVS_probs, axis=1)!=0).astype(int)

        # 2-2-2. Extract Accumulated Probability from raw probability result.
        pred_label_list2 = (SVS_probs[:,0]<0.5).astype(int)

        # 2-2-3. Using rescaled score from SVSs
        threshold = max_step+0.5
        pred_label_list3 = (SVSs<threshold).astype(int)

        # 2-3. Confusion matrix and evaluate metrics
        logger('\n  ----- BCC Evaluation -----')
        Max_bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list1, neg_label = 0)
        Acc_bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list2, neg_label = 0)
        Scalar_bin = BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list3, neg_label = 0)
        logger('  1. Max Probability')
        max_bin_acc, max_bin_prec, max_bin_critical_err = \
                Max_bin_conf_matrix.get_accuracy(), Max_bin_conf_matrix.get_precision(), Max_bin_conf_matrix.get_critical_error()
        logger('   Bin_acc, Bin_prec, Bin_critical_err, auroc =')
        logger(f'   {max_bin_acc}, {max_bin_prec}, {max_bin_critical_err}', end=', ')
        #logger(get_AUROC(true_label_list, pred_label_list1))
        logger('Cannot be defined in the mannor of max probability.')

        logger('  2. Accumulated Probability')
        acc_bin_acc, acc_bin_prec, acc_bin_critical_err = \
                Acc_bin_conf_matrix.get_accuracy(), Acc_bin_conf_matrix.get_precision(), Acc_bin_conf_matrix.get_critical_error()
        logger('   Bin_acc, Bin_prec, Bin_critical_err, auroc =')
        logger(f'   {acc_bin_acc}, {acc_bin_prec}, {acc_bin_critical_err}', end=', ')
        logger(get_AUROC(true_label_list, 1-SVS_probs[:,0]))

        logger('  3. Scalar score')
        logger(f'   used threshold: {threshold}')
        scalar_bin_acc, scalar_bin_prec, scalar_bin_critical_err = \
                Scalar_bin.get_accuracy(), Scalar_bin.get_precision(), Scalar_bin.get_critical_error()
        logger('   Bin_acc, Bin_prec, Bin_critical_err, auroc =')
        logger(f'   {scalar_bin_acc}, {scalar_bin_prec}, {scalar_bin_critical_err}', end=', ' )
        logger(get_AUROC(true_label_list, SVSs))

        # 2-4. SA score and SC score
        logger('  4. SA score')
        sas_auroc = get_AUROC(true_label_list, SAScores)
        logger(f'   auroc = {sas_auroc}')

        logger('  5. SC score')
        scs_auroc = get_AUROC(true_label_list, SCScores)
        logger(f'   auroc = {scs_auroc}')

        # 3. Setting for MCC test.
        true_label_list = []
        for idx, l in enumerate(each_class_sizes):
            true_label_list += [idx for i in range(l)]
        true_label_list = np.array(true_label_list)
        logger('\n  ----- MCC Evaluation -----')
        pred_label_list = np.argmax(SVS_probs, axis=1).astype(int)
        MCC_conf_matrix = UnbalMultiConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list, numb_classes=max_step+1)
        mcc_acc, macro_avg_precision, macro_avg_recall, macro_avg_f1_score = \
            MCC_conf_matrix.get_accuracy(), MCC_conf_matrix.get_macro_avg_precision(), MCC_conf_matrix.get_macro_avg_recall(), MCC_conf_matrix.get_macro_avg_f1_score()
        logger('  mcc_acc, macro_avg_precision, macro_avg_recall, macro_avg_f1_score = ')
        logger(f'  {mcc_acc}, {macro_avg_precision}, {macro_avg_recall}, {macro_avg_f1_score}')

        bin_result_dict = {
                'max_bin_acc': max_bin_acc, 'max_bin_prec': max_bin_prec, 'max_bin_critical_err': max_bin_critical_err,
                'acc_bin_acc': acc_bin_acc, 'acc_bin_prec': acc_bin_prec, 'acc_bin_critical_err': acc_bin_critical_err,
                'scalar_bin_acc': scalar_bin_acc, 'scalar_bin_prec': scalar_bin_prec, 'scalar_bin_critical_err': scalar_bin_critical_err,
                'sa_auroc': sas_auroc, 'sc_auroc':scs_auroc
                }
        mcc_result_dict = {
                'mcc_acc': mcc_acc, 'macro_avg_precision': macro_avg_precision, 'macro_avg_recall': macro_avg_recall,
                'macro_avg_f1_score': macro_avg_f1_score, 
                }

        first_row, second_row = [], []
        with open(csv_file, 'w', newline='') as fw:
            wr=csv.writer(fw)
            wr.writerow(bin_result_dict.keys())
            wr.writerow(bin_result_dict.values())
            wr.writerow([])
            wr.writerow(mcc_result_dict.keys())
            wr.writerow(mcc_result_dict.values())
        np.savetxt(mcc_array_file, MCC_conf_matrix.conf_matrix, fmt='%.0f')
        np.savetxt(bin_array_file, Max_bin_conf_matrix.conf_matrix, fmt='%.0f')

        return True

    elif problem == 'regression':
        # 2-2. Use our model as as binary classification model.
        # Threshold must be max_step+0.5
        threshold = max_step+0.5
        print(SVSs)
        bin_label_list = (SVSs<threshold).astype(int)

        # 2-3. Confusion matrix and evaluate metrics
        logger('\n  ----- BCC Evaluation -----')
        bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=bin_label_list, neg_label = 0)
        logger('  1. Our model(SVS)')
        bin_acc, bin_prec, bin_recall, bin_critical_err = \
                bin_conf_matrix.get_accuracy(), bin_conf_matrix.get_precision(), bin_conf_matrix.get_recall(), bin_conf_matrix.get_critical_error()
        logger('   Bin_acc, Bin_prec, Bin_recall, Bin_critical_err, auroc =')
        logger(f'   {bin_acc}, {bin_prec}, {bin_recall},{bin_critical_err}', end=', ')
        logger(get_AUROC(true_label_list, SVSs))

        # 2-4. SA score and SC score
        logger('  2. SA score')
        sas_auroc = get_AUROC(true_label_list, SAScores)
        logger(f'   auroc = {sas_auroc}')

        logger('  3. SC score')
        scs_auroc = get_AUROC(true_label_list, SCScores)
        logger(f'   auroc = {scs_auroc}')

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

        first_row, second_row = [], []
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
        problem,
        max_step, 
        save_dir, 
        test_file_path,
        logger,
        each_class_sizes=None
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

    # 1. result save path setting.
    save_dir = f'{save_dir}_{problem}'
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
    if problem=='classification':
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

    if problem == 'classification':
        # 2-2-1. Extract Max Probability from raw probability result.
        pred_label_list1 = (np.argmax(SVS_probs, axis=1)!=0).astype(int)

        # 2-2-2. Extract Accumulated Probability from raw probability result.
        pred_label_list2 = (SVS_probs[:,0]<0.5).astype(int)

        # 2-2-3. Using rescaled score from SVSs
        threshold = max_step+0.5
        pred_label_list3 = (SVSs<threshold).astype(int)

        # 2-3. Confusion matrix and evaluate metrics
        logger('\n  ----- BCC Evaluation -----')
        Max_bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list1, neg_label = 0)
        Acc_bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list2, neg_label = 0)
        Scalar_bin = BinaryConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list3, neg_label = 0)

        logger('  1. Max Probability')
        bin_acc, bin_precision, critical_err = Max_bin_conf_matrix.get_accuarcy(), Max_bin_conf_matrix.get_precision(), Max_bin_conf_matrix.get_critical_error()
        init_pos_ratio, ratio_change, filtering_ratio, = \
                Max_bin_conf_matrix.get_initial_pos_ratio(), Max_bin_conf_matrix.get_ratio_change(), Max_bin_conf_matrix.get_filtering_ratio()
        logger('   Bin_acc, Bin_prec, critical_err, auroc =')
        logger(f'   {bin_acc}, {bin_precision}, {critical_err}', end=', ')
        #logger(get_AUROC(true_label_list, pred_label_list1))
        logger('Cannot be defined in the mannor of max probability.')
        logger('   Init_pos_ratio, Ratio_change, filtering_ratio =')
        logger(f'   {init_pos_ratio}, {ratio_change}, {filtering_ratio}')

        logger('  2. Accumulated Probability')
        bin_acc, bin_precision, critical_err = Acc_bin_conf_matrix.get_accuarcy(), Acc_bin_conf_matrix.get_precision(), Acc_bin_conf_matrix.get_critical_error()
        init_pos_ratio, ratio_change, filtering_ratio, = \
                Acc_bin_conf_matrix.get_initial_pos_ratio(), Acc_bin_conf_matrix.get_ratio_change(), Acc_bin_conf_matrix.get_filtering_ratio()
        logger('   Bin_acc, Bin_prec, critical_err, auroc =')
        logger(f'   {bin_acc}, {bin_precision}, {critical_err}', end=', ')
        logger(get_AUROC(true_label_list, 1-SVS_probs[:,0]))
        logger('   Init_pos_ratio, Ratio_change, filtering_ratio =')
        logger(f'   {init_pos_ratio}, {ratio_change}, {filtering_ratio}')

        logger('  3. Scalar score')
        bin_acc, bin_prec, critical_err = \
                Scalar_bin.get_accuracy(), Scalar_bin.get_precision(), Scalar_bin.get_critical_error()
        init_pos_ratio, ratio_change, filtering_ratio, = \
                Scalar_bin.get_initial_pos_ratio(), Scalar_bin.get_ratio_change(), Scalar_bin.get_filtering_ratio()
        logger('   Bin_acc, Bin_prec, Bin_critical_err, auroc =')
        logger(f'   {bin_acc}, {bin_prec}, {critical_err}', end=', ' )
        logger(get_AUROC(true_label_list, SVSs))
        logger('   Init_pos_ratio, Ratio_change, filtering_ratio =')
        logger(f'   {init_pos_ratio}, {ratio_change}, {filtering_ratio}')

        # 3. Setting for MCC test.
        true_label_list = []
        for idx, l in enumerate(each_class_sizes):
            true_label_list += [idx for i in range(l)]
        true_label_list = np.array(true_label_list)
        logger('\n  ----- MCC Evaluation -----')
        pred_label_list = np.argmax(SVS_probs, axis=1).astype(int)
        MCC_conf_matrix = UnbalMultiConfusionMatrix(true_list=true_label_list, pred_list=pred_label_list, numb_classes=max_step+1)

        mcc_acc, macro_avg_precision, macro_avg_recall, macro_avg_f1_score = \
            MCC_conf_matrix.get_accuracy(), MCC_conf_matrix.get_macro_avg_precision(), MCC_conf_matrix.get_macro_avg_recall(), MCC_conf_matrix.get_macro_avg_f1_score()
        logger('   mcc_acc, macro_avg_precision, macro_avg_recall, macro_avg_f1_score = ')
        logger(f'   {mcc_acc}, {macro_avg_precision}, {macro_avg_recall}, {macro_avg_f1_score}')

        mcc_result_dict = {
                'mcc_acc': mcc_acc, 'macro_avg_precision': macro_avg_precision,'macro_avg_recall': macro_avg_recall, 'macro_avg_f1_score': macro_avg_f1_score
                }

        bin_result_dict = {
                'bin_acc': bin_acc, 'bin_precision': bin_precision,
                'init_pos_ratio': init_pos_ratio, 'ratio_change (pos:neg)': ratio_change, 'critical_err': critical_err,
                'filtered_out_ratio_pos': filtering_ratio['pos'],'filtered_out_ratio_neg': filtering_ratio['neg'],'filtered_out_ratio_tot': filtering_ratio['tot']
                }

        first_row, second_row = [], []
        with open(csv_file, 'w', newline='') as fw:
            wr=csv.writer(fw)
            wr.writerow(bin_result_dict.keys())
            wr.writerow(bin_result_dict.values())
            wr.writerow([])
            wr.writerow(mcc_result_dict.keys())
            wr.writerow(mcc_result_dict.values())
        np.savetxt(mcc_array_file, MCC_conf_matrix.conf_matrix, fmt='%.0f')
        np.savetxt(bin_array_file, Max_bin_conf_matrix.conf_matrix, fmt='%.0f')

        return True

    elif problem == 'regression':
        # 2-2. Use our model as as binary classification model.
        # Threshold must be max_step+0.5
        threshold = max_step+0.5
        bin_label_list = (SVSs<threshold).astype(int)

        # 2-3. Confusion matrix and evaluate metrics
        logger('\n  ----- BCC Evaluation -----')
        bin_conf_matrix= BinaryConfusionMatrix(true_list=true_label_list, pred_list=bin_label_list, neg_label = max_step+1)
        logger('  1. Our model(SVS)')
        bin_acc, bin_prec, bin_recall, bin_critical_err = \
                bin_conf_matrix.get_accuracy(), bin_conf_matrix.get_precision(), bin_conf_matrix.get_recall(), bin_conf_matrix.get_critical_error()
        init_pos_ratio, ratio_change, filtering_ratio, = \
                bin_conf_matrix.get_initial_pos_ratio(), bin_conf_matrix.get_ratio_change(), bin_conf_matrix.get_filtering_ratio()
        logger('   Bin_acc, Bin_prec, Bin_recall, Bin_critical_err, auroc =')
        logger(f'   {bin_acc}, {bin_prec}, {bin_recall}, {bin_critical_err}', end=', ')
        logger(get_AUROC(true_label_list, SVSs))
        logger('   Init_pos_ratio, Ratio_change, filtering_ratio =')
        logger(f'   {init_pos_ratio}, {ratio_change}, {filtering_ratio}')

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
        logger('   mcc_acc, macro_avg_precision, macro_avg_recall, macro_avg_f1_score = ')
        logger(f'   {mcc_acc}, {macro_avg_precision}, {macro_avg_recall}, {macro_avg_f1_score}')

        mcc_result_dict = {
                'mcc_acc': mcc_acc, 'macro_avg_precision': macro_avg_precision,'macro_avg_recall': macro_avg_recall, 'macro_avg_f1_score': macro_avg_f1_score
                }

        bin_result_dict = {
                'bin_acc': bin_acc, 'bin_precision': bin_prec, 'bin_recall': bin_recall,
                'init_pos_ratio': init_pos_ratio, 'ratio_change (pos:neg)': ratio_change, 'critical_err': critical_err,
                'filtered_out_ratio_pos': filtering_ratio['pos'],'filtered_out_ratio_neg': filtering_ratio['neg'],'filtered_out_ratio_tot': filtering_ratio['tot']
                }

        first_row, second_row = [], []
        with open(csv_file, 'w', newline='') as fw:
            wr=csv.writer(fw)
            wr.writerow(bin_result_dict.keys())
            wr.writerow(bin_result_dict.values())
            wr.writerow([])
            wr.writerow(mcc_result_dict.keys())
            wr.writerow(mcc_result_dict.values())
        np.savetxt(mcc_array_file, MCC_conf_matrix.conf_matrix, fmt='%.0f')
        np.savetxt(bin_array_file, Max_bin_conf_matrix.conf_matrix, fmt='%.0f')

        return True

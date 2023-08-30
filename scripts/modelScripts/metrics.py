import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, roc_auc_score


class ConfMatrix:
    def __init__(
        self, true_list, pred_list, labels=None, sample_weight=None, normalize=None
    ):
        self.conf_matrix = confusion_matrix(
            y_true=true_list,
            y_pred=pred_list,
            labels=labels,
            sample_weight=sample_weight,  # same size with y_true and y_pred
            normalize=normalize,  # {'true', 'pred', 'all'}
        )

    def get_numb_data(self):
        return np.sum(self.conf_matrix)

    def get_numb_pos(self, neg_label):
        return np.sum(self.conf_matrix[:neg_label, :]) + np.sum(
            self.conf_matrix[neg_label + 1 :, :]
        )

    def get_numb_neg(self, neg_label):
        return np.sum(self.conf_matrix[neg_label, :])

    def get_numb_correct(self):
        return np.sum(self.conf_matrix.diagonal())

    def get_numb_false(self):
        return np.sum(self.conf_matrix) - np.sum(self.conf_matrix.diagonal())

    def get_class_precision(self, col_idx):
        column = self.conf_matrix[:, col_idx]
        return round(float(self.conf_matrix[col_idx, col_idx] / np.sum(column)), 3)

    def get_class_recall(self, row_idx):
        row = self.conf_matrix[row_idx, :]
        return round(float(self.conf_matrix[row_idx, row_idx] / np.sum(row)), 3)


class BinaryConfusionMatrix(ConfMatrix):
    def __init__(self, true_list, pred_list, neg_label):
        super().__init__(true_list=true_list, pred_list=pred_list)
        self.neg_label = int(neg_label)
        self.pos_label = int(1 - neg_label)

    def get_accuracy(self):  # for computing critical error. not explicit use.
        return round(float(self.get_numb_correct() / self.get_numb_data()), 3)

    def get_precision(self):
        return round(float(self.get_class_precision(col_idx=self.pos_label)), 3)

    def get_recall(self):
        return round(float(self.get_class_recall(row_idx=self.pos_label)), 3)

    def get_f1_score(self):
        prec, recall = self.get_precision(self.pos_label), self.get_recall(
            self.pos_label
        )
        return round(float(2 / ((1 / prec) + (1 / recall))), 3)

    def get_critical_error(self):
        return round(float(1 - self.get_accuracy()), 3)

    def get_initial_pos_ratio(self):  # for EXP03.
        return round(float(self.get_numb_pos(self.neg_label) / self.get_numb_data()), 3)

    def get_ratio_change(self):  # for EXP03.
        initial_ratio = f"{int(self.get_numb_pos(self.neg_label))}:{int(self.get_numb_neg(self.neg_label))}"
        after_filtering_ratio = f"{int(self.conf_matrix[self.pos_label][self.pos_label])}:{int(self.conf_matrix[self.neg_label][self.pos_label])}"
        return " >> ".join([initial_ratio, after_filtering_ratio])

    def get_filtering_ratio(self):  # for EXP03.
        neg_filtered_ratio = round(
            float(
                self.conf_matrix[self.neg_label, self.neg_label]
                / self.get_numb_neg(self.neg_label)
            ),
            3,
        )
        pos_filtered_ratio = round(
            float(
                self.conf_matrix[self.pos_label, self.neg_label]
                / self.get_numb_pos(self.neg_label)
            ),
            3,
        )
        tot_filtered_ratio = round(
            float(np.sum(self.conf_matrix[:, self.neg_label]) / self.get_numb_data()), 3
        )
        return {
            "pos": pos_filtered_ratio,
            "neg": neg_filtered_ratio,
            "tot": tot_filtered_ratio,
        }

    def get_main_results(self):
        return (
            self.get_accuracy(),
            self.get_precision(),
            self.get_recall(),
            self.get_critical_error(),
        )


class UnbalMultiConfusionMatrix(ConfMatrix):  # for EXP01, EXP03. MCCs.
    def __init__(self, true_list, pred_list, numb_classes):
        super().__init__(true_list=true_list, pred_list=pred_list)
        self.numb_classes = numb_classes

    def get_accuracy(self):  # = Micro-avg prec, Micro-avg recall, Micro-avg F1 score.
        return round(float(self.get_numb_correct() / self.get_numb_data()), 3)

    def get_macro_avg_precision(self):
        precs = np.array(
            [
                self.get_class_precision(class_idx)
                for class_idx in range(self.numb_classes)
            ]
        )
        return round(float(np.sum(precs) / self.numb_classes), 3)

    def get_macro_avg_recall(self):
        recalls = np.array(
            [self.get_class_recall(class_idx) for class_idx in range(self.numb_classes)]
        )
        return round(float(np.sum(recalls) / self.numb_classes), 3)

    def get_macro_avg_f1_score(self):
        prec, recall = self.get_macro_avg_precision(), self.get_macro_avg_recall()
        return round(float(2 / ((1 / prec) + (1 / recall))), 3)


def get_AUROC(y_true, y_score):
    return round(roc_auc_score(y_true, y_score), 3)


def get_pearsonr(y_true, y_score):  # for linearly related.
    return round(pearsonr(y_true, y_score)[0], 3)


def get_spearmanr(y_true, y_score):
    return round(spearmanr(y_true, y_score)[0], 3)


def get_kendalltau(y_true, y_score):
    return round(kendalltau(y_true, y_score)[0], 3)


if __name__ == "__main__":
    # first case
    true_list = np.concatenate([np.zeros(2), np.ones(2), np.ones(2) * 2])
    prob_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 2.0])
    print(true_list)
    print(prob_list)
    print(get_pearsonr(true_list, prob_list))
    print(get_spearmanr(true_list, prob_list))
    print(get_kendalltau(true_list, prob_list))

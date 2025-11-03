import bisect
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
def softmax(X):
    """
    softmax函数实现

    参数：
    x --- 一个二维矩阵, m * n,其中m表示向量个数，n表示向量维度

    返回：
    softmax计算结果
    """
    assert (len(X.shape) == 2)
    row_max = np.max(X, axis=1).reshape(-1, 1)
    X -= row_max
    X_exp = np.exp(X)
    s = X_exp / np.sum(X_exp, axis=1, keepdims=True)

    return s
def compute_p_value(scores, null_p_values):
    """Get the p-value for each score by examining the list null distribution
    where scores are obtained by a certain probability.

    NOTE: uses score2pval function

    Parameters
    ----------
    scores : pd.Series
        series of observed scores
    null_p_values: pd.Series
        Empirical null distribution, index are scores and values are p values

    Returns
    -------
    pvals : pd.Series
        Series of p values for scores
    """
    num_scores = len(scores)
    pvals = pd.Series(np.zeros(num_scores))
    null_p_val_scores = list(reversed(null_p_values.index.tolist()))
    #null_p_values = null_p_values.ix[null_p_val_scores].copy()
    null_p_values.sort_values(inplace=True, ascending=False)
    pvals = scores.apply(lambda x: score2pval(x, null_p_val_scores, null_p_values))
    return pvals


def score2pval(score, null_scores, null_pvals):
    """Looks up the P value from the empirical null distribution based on the provided
    score.

    NOTE: null_scores and null_pvals should be sorted in ascending order.

    Parameters
    ----------
    score : float
        score to look up P value for
    null_scores : list
        list of scores that have a non-NA value
    null_pvals : pd.Series
        a series object with the P value for the scores found in null_scores

    Returns
    -------
    pval : float
        P value for requested score
    """
    # find position in simulated null distribution
    pos = bisect.bisect_right(null_scores, score)

    # if the score is beyond any simulated values, then report
    # a p-value of zero
    if pos == null_pvals.size and score > null_scores[-1]:
        return 0
    # condition needed to prevent an error
    # simply get last value, if it equals the last value
    elif pos == null_pvals.size:
        return null_pvals.iloc[pos-1]
    # normal case, just report the corresponding p-val from simulations
    else:
        return null_pvals.iloc[pos]


def cummin(x):
    """A python implementation of the cummin function in R"""
    for i in range(1, len(x)):
        if x[i-1] < x[i]:
            x[i] = x[i-1]
    return x


def bh_fdr(pval):
    """A python implementation of the Benjamani-Hochberg FDR method.

    This code should always give precisely the same answer as using
    p.adjust(pval, method="BH") in R.

    Parameters
    ----------
    pval : list or array
        list/array of p-values

    Returns
    -------
    pval_adj : np.array
        adjusted p-values according the benjamani-hochberg method
    """
    pval_array = np.array(pval)
    sorted_order = np.argsort(pval_array)
    original_order = np.argsort(sorted_order)
    pval_array = pval_array[sorted_order]

    # calculate the needed alpha
    n = float(len(pval))
    pval_adj = np.zeros(int(n))
    i = np.arange(1, int(n)+1, dtype=float)[::-1]  # largest to smallest
    pval_adj = np.minimum(1, cummin(n/i * pval_array[::-1]))[::-1]
    return pval_adj[original_order]
def null_distribution(result_df,path):


    # driver scores
    driver_score_cts = result_df['score'].value_counts()



    driver_score_cts = driver_score_cts.sort_index(ascending=False)
    driver_score_cum_cts = driver_score_cts.cumsum()
    driver_score_pvals = driver_score_cum_cts / float(driver_score_cts.sum())
    score_ix = set(driver_score_pvals.index)
    score_pvals = pd.DataFrame(index=list(score_ix))
    score_pvals = score_pvals.sort_index(ascending=False)
    score_pvals['p-value'] = driver_score_pvals
    score_pvals.to_csv(path, sep='\t',
                       index_label='score')




def p_val(labeled_X, labeled_y,model,true_score,result_path,threshold=0.1):
    lx = []
    for ii in range(len(labeled_X)):
        if labeled_y[ii] == 0:
            lx.append(labeled_X[ii])

    df = pd.DataFrame(lx)
    iter_times = 5
    n,m = labeled_X.shape
    scores = []
    for jj in range(m):
        tmp_df = df.copy()
        for k in range(iter_times):
            prng = np.random.RandomState(k)
            permute_order = prng.choice(len(df),size=len(df),replace=False)
            tmp_df.iloc[:,jj] = df.iloc[permute_order,jj].values
            new_X = tmp_df.values
            prob_y = model.predict_proba(new_X)
            prob = softmax(prob_y)
            # [scores.append(round(score[1],3)) for score in pred_y]
            [scores.append(score[1]) for score in prob]
    tmp_df = pd.DataFrame()
    tmp_df['score'] = scores
    nullPath = './result/null_dis.csv'
    null_distribution(tmp_df,nullPath)
    null_pval = pd.read_csv(nullPath,sep='\t',index_col=0)
    final_df = pd.DataFrame()
    score = true_score['score'].copy()
    score.sort_values(inplace=True, ascending=False)
    final_df['Score'] = score
    final_df['p-value'] = compute_p_value(score, null_pval['p-value'].dropna())
    final_df['q-value'] = bh_fdr(final_df['p-value'])
    final_df['is_cancer_related(<=threshold)'] = final_df['q-value'].map(lambda x: 1 if x < threshold else 0)
    final_df.to_csv(result_path,sep='\t')

#
def get_metric(X, y,model):
    performance = list()
    accuracy = list()

    pred_y = model.evaluate(X, y)
    y_scores = model.predict_proba(X)
    y_true = y
    auc = roc_auc_score(y_true, y_scores)

    prc = average_precision_score(y, pred_y[:, 1])
    accuracy = accuracy_score(y_true, pred_y)


    return accuracy, auc, prc


class BinaryClassificationEvaluator:
    """
    Binary Classification Evaluator
    Provides comprehensive binary classification metrics and visualization tools
    """

    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        Initialize the evaluator

        Parameters:
        - y_true: True labels
        - y_pred: Predicted labels
        - y_pred_proba: Predicted probabilities (optional, for AUC calculation)
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_proba = np.array(y_pred_proba) if y_pred_proba is not None else None

        # Validate input data
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        if y_pred_proba is not None and len(y_true) != len(y_pred_proba):
            raise ValueError("y_true and y_pred_proba must have the same length")

    def calculate_basic_metrics(self):
        """
        Calculate basic evaluation metrics
        """
        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)

        # Precision
        metrics['precision'] = precision_score(self.y_true, self.y_pred, zero_division=0)

        # Recall
        metrics['recall'] = recall_score(self.y_true, self.y_pred, zero_division=0)

        # F1 Score
        metrics['f1_score'] = f1_score(self.y_true, self.y_pred, zero_division=0)

        # If prediction probabilities are provided, calculate AUC
        if self.y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(self.y_true, self.y_pred_proba)
            metrics['auc_pr'] = average_precision_score(self.y_true, self.y_pred_proba)

        return metrics

    def calculate_detailed_metrics(self):
        """
        Calculate detailed evaluation metrics
        """
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()

        metrics = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }

        # Calculate various metrics
        total = tn + fp + fn + tp
        metrics['total_samples'] = int(total)

        # Sensitivity/Recall (TPR)
        metrics['sensitivity'] = metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Specificity (TNR)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        # Precision (PPV)
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0

        # NPV (Negative Predictive Value)
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0

        # FPR (False Positive Rate)
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0

        # FNR (False Negative Rate)
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

        # FDR (False Discovery Rate)
        metrics['fdr'] = fp / (fp + tp) if (fp + tp) > 0 else 0

        # Accuracy
        metrics['accuracy'] = (tp + tn) / total if total > 0 else 0

        # F1 Score
        metrics['f1_score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        # If prediction probabilities are provided, calculate AUC
        if self.y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(self.y_true, self.y_pred_proba)
            metrics['auc_pr'] = average_precision_score(self.y_true, self.y_pred_proba)

        return metrics

    def get_confusion_matrix(self):
        """
        Get confusion matrix
        """
        return confusion_matrix(self.y_true, self.y_pred)

    def plot_confusion_matrix(self, normalize=False, title="Confusion Matrix"):
        """
        Plot confusion matrix heatmap
        """
        cm = self.get_confusion_matrix()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def plot_roc_curve(self):
        """
        Plot ROC curve
        """
        if self.y_pred_proba is None:
            print("ROC curve requires prediction probabilities.")
            return

        fpr, tpr, _ = roc_curve(self.y_true, self.y_pred_proba)
        auc_score = roc_auc_score(self.y_true, self.y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def plot_precision_recall_curve(self):
        """
        Plot Precision-Recall curve
        """
        if self.y_pred_proba is None:
            print("Precision-Recall curve requires prediction probabilities.")
            return

        precision_vals, recall_vals, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_true, self.y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall_vals, precision_vals, color='blue', lw=2,
                 label=f'P-R curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.show()

    def print_report(self):
        """
        Print complete evaluation report
        """
        basic_metrics = self.calculate_basic_metrics()
        detailed_metrics = self.calculate_detailed_metrics()

        print("=" * 50)
        print("BINARY CLASSIFICATION EVALUATION REPORT")
        print("=" * 50)

        print("\nBASIC METRICS:")
        print(f"Accuracy:  {basic_metrics['accuracy']:.4f}")
        print(f"Precision: {basic_metrics['precision']:.4f}")
        print(f"Recall:    {basic_metrics['recall']:.4f}")
        print(f"F1-Score:  {basic_metrics['f1_score']:.4f}")

        if 'auc_roc' in basic_metrics:
            print(f"AUC-ROC:   {basic_metrics['auc_roc']:.4f}")
            print(f"AUC-PR:    {basic_metrics['auc_pr']:.4f}")

        print("\nDETAILED METRICS:")
        print(f"True Negatives:  {detailed_metrics['true_negatives']}")
        print(f"False Positives: {detailed_metrics['false_positives']}")
        print(f"False Negatives: {detailed_metrics['false_negatives']}")
        print(f"True Positives:  {detailed_metrics['true_positives']}")

        print(f"Sensitivity:     {detailed_metrics['sensitivity']:.4f}")
        print(f"Specificity:     {detailed_metrics['specificity']:.4f}")
        print(f"NPV:             {detailed_metrics['npv']:.4f}")
        print(f"FPR:             {detailed_metrics['fpr']:.4f}")
        print(f"FNR:             {detailed_metrics['fnr']:.4f}")
        print(f"FDR:             {detailed_metrics['fdr']:.4f}")

        print(f"Total Samples:   {detailed_metrics['total_samples']}")

        print("=" * 50)


def evaluate_binary_classifier(y_true, y_pred, y_pred_proba=None, plot=True):
    """
    Convenience function: Evaluate binary classification model

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_pred_proba: Predicted probabilities (optional)
    - plot: Whether to plot charts
    """
    evaluator = BinaryClassificationEvaluator(y_true, y_pred, y_pred_proba)

    # Print report
    evaluator.print_report()

    if plot and y_pred_proba is not None:
        # Plot confusion matrix
        evaluator.plot_confusion_matrix()
        evaluator.plot_confusion_matrix(normalize=True, title="Normalized Confusion Matrix")

        # Plot ROC curve
        evaluator.plot_roc_curve()

        # Plot Precision-Recall curve
        evaluator.plot_precision_recall_curve()

    return evaluator


def evaluate_binary_classifier(y_true, y_pred, y_pred_proba=None, plot=True):
    """
    Convenience function: Evaluate binary classification model

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_pred_proba: Predicted probabilities (optional)
    - plot: Whether to plot charts
    """
    evaluator = BinaryClassificationEvaluator(y_true, y_pred, y_pred_proba)

    # Print report
    evaluator.print_report()

    if plot and y_pred_proba is not None:
        # Plot confusion matrix
        evaluator.plot_confusion_matrix()
        evaluator.plot_confusion_matrix(normalize=True, title="Normalized Confusion Matrix")

        # Plot ROC curve
        evaluator.plot_roc_curve()

        # Plot Precision-Recall curve
        evaluator.plot_precision_recall_curve()

    return evaluator
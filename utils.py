import numpy as np
import pandas as pd
import bisect

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
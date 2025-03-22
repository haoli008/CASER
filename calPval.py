import numpy as np
import pandas as pd
import p_value as pv
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
    iter_times = 10
    n,m = labeled_X.shape
    scores = []
    for jj in range(m):
        tmp_df = df.copy()
        for k in range(iter_times):
            prng = np.random.RandomState(k)
            permute_order = prng.choice(len(df),size=len(df),replace=False)
            tmp_df.iloc[:,jj] = df.iloc[permute_order,jj].values
            new_X = tmp_df.values
            pred_y = model.predict_proba(new_X)
            # [scores.append(round(score[1],3)) for score in pred_y]
            [scores.append(score[1]) for score in pred_y]

    tmp_df = true_score
    nullPath = './result/null_dis.csv'
    null_distribution(tmp_df,nullPath)
    null_pval = pd.read_csv(nullPath,sep='\t',index_col=0)

    score = tmp_df['score'].copy()
    score.sort_values(inplace=True, ascending=False)
    tmp_df['p-value'] = pv.compute_p_value(score, null_pval['p-value'].dropna())
    tmp_df['q-value'] = pv.bh_fdr(tmp_df['p-value'])
    tmp_df['is_cancer_related'] = tmp_df['q-value'].map(lambda x: 1 if x < threshold else 0)
    tmp_df.to_csv(result_path,sep='\t')



import pandas as pd
from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
from calPval import p_val
import pickle
from sklearn.svm import SVC
from imblearn.over_sampling import BorderlineSMOTE

SMOTE = BorderlineSMOTE(random_state=42, kind="borderline-1")

def uTri_Training(labeled_X, labeled_y, unlabeled_X):
    labeled_X, labeled_y = SMOTE.fit_resample(labeled_X, labeled_y)
    base_estimator = SVC(C=100, kernel='rbf', probability=True, random_state=20)
    base_estimator_2 = SVC(C=50, kernel='sigmoid', probability=True, random_state=40)
    base_estimator_3 = SVC(C=25, kernel='sigmoid', probability=True, random_state=25)
    model = Tri_Training(base_estimator=base_estimator, base_estimator_2=base_estimator_2,
                         base_estimator_3=base_estimator_3)
    model.fit(X=labeled_X, y=labeled_y, unlabeled_X=unlabeled_X)
    # prob_y = model.predict_proba(unlabeled_X)
    # pred_y = model.predict(unlabeled_X)
    return model

def predict(knonwn_gene_path,unknonwn_gene_path,result_path,pre_train=False,model_path=None):
    df = pd.read_csv(knonwn_gene_path, sep='\t',index_col=0)
    # Genw       VEST_score_missense  VEST_score_frameshift_indels  ...  label（is_caner_related)
    # A1CF   0.4211              0.0059                        ...   1
    # AAAS   0.3861              0.0347                        ...   0
    # AAK1   0.2987              0.0136                        ...   0
    columns_to_keep = [col for col in df.columns if col != 'label']
    train_X = df[columns_to_keep]
    train_y = df['label']
    unknonwn_X = pd.read_csv(unknonwn_gene_path, sep='\t',index_col=0)
    if not pre_train:
        model = uTri_Training(train_X.values,train_y.values,unknonwn_X.values)
    else:
        model = pickle.load(model_path)
    prob_y = model.predict_proba(unknonwn_X)
    pred_y = model.predict(unknonwn_X)
    true_score = unknonwn_X.copy()

    true_score['score'] = prob_y[:, 1]
    p_val(unknonwn_X.values, pred_y, model,true_score,result_path)

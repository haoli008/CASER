import pandas as pd
from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
from utils import p_val
import pickle
from sklearn.svm import SVC
from imblearn.over_sampling import BorderlineSMOTE
from utils import softmax
import ML_models as ml
SMOTE = BorderlineSMOTE(random_state=42, kind="borderline-1")

def predict(knonwn_gene_path,unknonwn_gene_path,result_path,model=None,pre_train=False,model_path=None):
    df = pd.read_csv(knonwn_gene_path, sep='\t',index_col=0)
    # Genw       VEST_score_missense  VEST_score_frameshift_indels  ...  label（is_caner_related)
    # A1CF   0.4211              0.0059                        ...   1
    # AAAS   0.3861              0.0347                        ...   0
    # AAK1   0.2987              0.0136                        ...   0
    columns_to_keep = [col for col in df.columns if col != 'label']
    train_X = df[columns_to_keep]
    train_y = df['label']
    unknonwn_X = pd.read_csv(unknonwn_gene_path, sep='\t',index_col=0)
    if not pre_train and model is None:
        model = ml.tri_training
    else:
        model = pickle.load(model_path)
    pred_y, prob_y,trained_model = model(train_X.values, train_y.values, unknonwn_X.values)
    prob = softmax(prob_y)

    true_score = unknonwn_X.copy()

    true_score['score'] = prob[:, 1]
    p_val(unknonwn_X.values, pred_y, trained_model,true_score,result_path)
def predict_subtype(knonwn_gene_paths,unknonwn_gene_paths,result_paths,model=None,pre_train=False,model_path=None):
    for idx in range(len(knonwn_gene_paths)):
        knonwn_gene_path = knonwn_gene_path[idx]
        unknonwn_gene_path = unknonwn_gene_path[idx]
        result_path = result_path[idx]
        df = pd.read_csv(knonwn_gene_path, sep='\t',index_col=0)
        # Genw       VEST_score_missense  VEST_score_frameshift_indels  ...  label（is_caner_related)
        # A1CF   0.4211              0.0059                        ...   1
        # AAAS   0.3861              0.0347                        ...   0
        # AAK1   0.2987              0.0136                        ...   0
        columns_to_keep = [col for col in df.columns if col != 'label']
        train_X = df[columns_to_keep]
        train_y = df['label']
        unknonwn_X = pd.read_csv(unknonwn_gene_path, sep='\t',index_col=0)
        if not pre_train and model is None:
            model = ml.tri_training
        else:
            model = pickle.load(model_path)
        pred_y,prob_y = model(train_X.values,train_y.values,unknonwn_X.values)
        prob = softmax(prob_y)
        true_score = unknonwn_X.copy()

        true_score['score'] = prob[:, 1]
        p_val(unknonwn_X.values, pred_y, model,true_score,result_path)

import pandas as pd
from LAMDA_SSL.Algorithm.Classification.Tri_Training import Tri_Training
from utils.tools import p_val
import pickle
from sklearn.svm import SVC
from imblearn.over_sampling import BorderlineSMOTE
from utils.tools import softmax
import utils.ML_models as ml

# Initialize SMOTE for handling imbalanced data
SMOTE = BorderlineSMOTE(random_state=42, kind="borderline-1")


def CASER(gene_path, unkonwn_gene_path, res_path, model=None, pre_train=False, model_path=None):
    """
    Main CASER prediction function

    Parameters:
    - gene_PATH: Path to known gene data (with labels)
    - unKonwn_gene_path: Path to unknown gene data for prediction
    - res_path: Path to save results
    - model: Model to use (default is tri_training)
    - pre_train: Whether to use pre-trained model
    - model_path: Path to pre-trained model
    """
    # Read training data
    df = pd.read_csv(gene_path, sep='\t', index_col=0)

    # Separate features and labels
    columns_to_keep = [col for col in df.columns if col != 'label']
    train_X = df[columns_to_keep]
    train_y = df['label']

    # Read data to be predicted
    unknown_X = pd.read_csv(unkonwn_gene_path, sep='\t', index_col=0)

    # Select model
    if not pre_train and model is None:
        model = ml.tri_training
    elif pre_train and model_path is not None:
        model = pickle.load(open(model_path, 'rb'))

    # Train model and make predictions
    pred_y, prob_y, trained_model = model(train_X.values, train_y.values, unknown_X.values)

    # Calculate prediction probabilities
    prob = softmax(prob_y)

    # Create result DataFrame
    true_score = unknown_X.copy()
    true_score['score'] = prob[:, 1]

    # Calculate p-values and save results
    p_val(unknown_X.values, pred_y, trained_model, true_score, res_path)


def predict(known_gene_path, unknown_gene_path, result_path, model=None, pre_train=False, model_path=None):
    """
    Prediction function (maintaining backward compatibility)
    """
    return CASER(known_gene_path, unknown_gene_path, result_path, model, pre_train, model_path)


def predict_subtype(known_gene_paths, unknown_gene_paths, result_paths, model=None, pre_train=False, model_path=None):
    """
    Subtype prediction function

    Parameters:
    - known_gene_paths: List of paths to known gene files
    - unknown_gene_paths: List of paths to unknown gene files for prediction
    - result_paths: List of paths to save results
    - model: Model to use
    - pre_train: Whether to use pre-trained model
    - model_path: Path to pre-trained model
    """
    for idx in range(len(known_gene_paths)):
        known_gene_path = known_gene_paths[idx]
        unknown_gene_path = unknown_gene_paths[idx]
        result_path = result_paths[idx]

        # Read training data
        df = pd.read_csv(known_gene_path, sep='\t', index_col=0)

        # Separate features and labels
        columns_to_keep = [col for col in df.columns if col != 'label']
        train_X = df[columns_to_keep]
        train_y = df['label']

        # Read data to be predicted
        unknown_X = pd.read_csv(unknown_gene_path, sep='\t', index_col=0)

        # Select model
        if not pre_train and model is None:
            model = ml.tri_training
        elif pre_train and model_path is not None:
            model = pickle.load(open(model_path, 'rb'))

        # Train model and make predictions
        pred_y, prob_y, trained_model = model(train_X.values, train_y.values, unknown_X.values)

        # Calculate prediction probabilities
        prob = softmax(prob_y)

        # Create result DataFrame
        true_score = unknown_X.copy()
        true_score['score'] = prob[:, 1]

        # Calculate p-values and save results
        p_val(unknown_X.values, pred_y, trained_model, true_score, result_path)




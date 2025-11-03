"""
CASER (Cancer-Associated Epigenetic Regulator identification) Usage Examples
This script demonstrates different ways to use the CASER tool for predicting cancer-associated epigenetic regulators.
"""

from CASER import predict, predict_subtype
import utils.ML_models as ml
import numpy as np
from utils.tools import evaluate_binary_classifier
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(BASE_DIR)


def basic_prediction_example():
    """
    Basic single prediction example
    Input file format:
    Gene    VEST_score_missense    VEST_score_frameshift_indels    ...    label
    A1CF    0.4211                 0.0059                          ...    1
    AAAS    0.3861                 0.0347                          ...    0
    AAK1    0.2987                 0.0136                          ...    0

    Output file format:
    Gene      Score          p-value              q-value               is_cancer_related
    ZNF217    0.76280177     0.0001692126179831   0.009896958830103773    1
    SATB2     0.7569754      0.0002654619970193   0.009896958830103773    1
    ...
    CSNK2A3   0.2800701      0.984669957774466    0.9886165307515179      0
    HSPA1A    0.2791872      0.9897121833084948   0.9916916076751118      0
    TSSK6     0.25791362     0.9951704545454544   0.9951704545454544      0
    """
    # Define file paths for basic prediction
    known_gene_path = os.path.join(BASE_DIR, "data", "geneSet.txt")  # Path to training data with labels
    unknown_gene_path = os.path.join(BASE_DIR, "data", "unknown_gene.txt")  # Path to data for prediction
    res_path = os.path.join(BASE_DIR, "result", "final_result.txt")  # Path to save results

    # Execute basic prediction
    predict(known_gene_path, unknown_gene_path, res_path)
    print("Basic prediction completed. Results saved to:", res_path)


def subtype_prediction_example():
    """
    Subtype prediction example for different functional categories of epigenetic regulators:
    - cERs: General cancer-associated epigenetic regulators
    - Readers: Proteins that read epigenetic marks
    - Remodelers: Chromatin remodeling complexes
    - Erasers: Enzymes that remove epigenetic marks
    - Writers: Enzymes that add epigenetic marks
    """
    known_gene_paths = [
        './data/cERs_gene.txt',
        './data/readers_gene.txt',
        './data/remodelers_gene.txt',
        './data/erasers_gene.txt',
        './data/writers_gene.txt'
    ]

    unknown_gene_paths = [
        './data/unknown_cERs_gene.txt',
        './data/unknown_readers_gene.txt',
        './data/unknown_remodelers_gene.txt',
        './data/unknown_erasers_gene.txt',
        './data/unknown_writers_gene.txt'
    ]

    result_paths = [
        './data/cERs_final_result.txt',
        './data/readers_final_result.txt',
        './data/remodelers_final_result.txt',
        './data/erasers_final_result.txt',
        './data/writers_final_result.txt'
    ]

    predict_subtype(known_gene_paths, unknown_gene_paths, result_paths)
    print("Subtype prediction completed. Results saved to respective directories.")


def custom_model_prediction_example():
    """
    Example using alternative models
    Available model options:
    - Deep semi-supervised methods: TemporalEnsembling, FlexMatch, VAT, MixMatch, LadderNetwork, UDA
    - Machine semi-supervised methods: SemiBoost, Tri_Training, Co_Training, LapSVM, Assemble, TSVM, SSGMM
    - Machine supervised methods: GDBT, Xgboost, LogisticRegression, RandomForestClassifier, SVC
    """
    known_gene_path = './data/geneSet.txt'
    unknown_gene_path = './data/unknown_gene.txt'
    res_path = './result/final_result.txt'

    # Use tri-training model (default)
    model = ml.tri_training

    # Alternative: Use deep learning model
    # model = dl.flexmatch

    # Execute prediction with selected model
    predict(known_gene_path, unknown_gene_path, res_path, model=model)
    print(f"Custom model prediction completed with {model.__name__}. Results saved to:", res_path)


def run_all_examples():
    """
    Run all example functions
    """
    print("Running CASER basic prediction example...")
    basic_prediction_example()

    print("\nRunning CASER subtype prediction example...")
    # subtype_prediction_example()

    print("\nRunning CASER custom model prediction example...")
    # custom_model_prediction_example()


def evaluate_example():
    """
    Run the model evaluation functions (This is mock data—not actual experimental or clinical data)
    """
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    # True labels (0: negative, 1: positive)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    # Predicted labels (simulated model predictions)
    y_pred = np.random.choice([0, 1], size=n_samples, p=[0.65, 0.35])

    # Predicted probabilities (simulated model outputs)
    y_pred_proba = np.random.uniform(0, 1, size=n_samples)

    # Evaluate model
    evaluator = evaluate_binary_classifier(y_true, y_pred, y_pred_proba, plot=True)

    # Can also get metrics only
    metrics = evaluator.calculate_detailed_metrics()
    print("\nMetrics as dictionary:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_all_examples()
    evaluate_example()

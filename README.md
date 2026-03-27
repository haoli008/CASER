# CASER: Cancer-Associated Epigenetic Regulator Identification

## Overview

**CASER** (Cancer-Associated Epigenetic Regulator identification) is a sophisticated semi-supervised machine learning framework designed to systematically prioritize cancer-associated epigenetic regulators (cERs) through comprehensive multi-omics data integration. The identification of reliable cERs is crucial for advancing cancer diagnosis, therapeutic target discovery, and understanding the molecular mechanisms underlying cancer biology.

While numerous epigenetic regulators (ERs) have been implicated in cancer progression, a significant number remain uncharacterized in the context of cancer phenotypes. CASER addresses this critical gap by implementing a tri-training-based semi-supervised learning approach that leverages both labeled and unlabeled data to identify novel cERs with high confidence.

## Key Features

- **Multi-omics Integration**: Combines mutation, genomic, epigenetic, and transcriptomic features for comprehensive analysis
- **Semi-supervised Learning**: Utilizes tri-training methodology to maximize learning from limited labeled data
- **Functional Categorization**: Classifies cERs into four functional roles: writers, erasers, readers, and remodelers
- **High Performance**: Demonstrates superior performance compared to various machine learning and deep learning models
- **Biological Validation**: Identified novel cERs show comparable cancer-driving potential to known cancer driver genes
- **Therapeutic Relevance**: cERs are enriched in gene-drug network modules, highlighting therapeutic potential

## Methodology

CASER integrates comprehensive omics data including:

- **Mutation features**: VEST scores for missense, frameshift, and inframe indels
- **Genomic features**: Copy number variations, structural variants
- **Epigenetic features**: DNA methylation, histone modifications, super-enhancer associations
- **Transcriptomic features**: Gene expression levels, alternative splicing patterns

The tri-training approach enables robust predictions even when labeled training data is limited, making it particularly valuable for rare or understudied cancer types.

## Installation

### 1) Create and activate a virtual environment

Windows (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional (only if you use deep semi-supervised models in `utils/DL_models.py`):

```bash
python -m pip install torch
```

## Step-by-Step Running Guide

### Quick Start (run built-in example)

```bash
python example.py
```

This will:
- Train a tri-training model on `data/geneSet.txt`
- Predict on `data/unknown_gene.txt`
- Save results to `result/final_result.txt`

### Command-Line Interface (recommended)

```bash
python run_caser.py --known ./data/geneSet.txt --unknown ./data/unknown_gene.txt --out ./result/final_result.txt
```

Switch model:

```bash
python run_caser.py --known ./data/geneSet.txt --unknown ./data/unknown_gene.txt --out ./result/final_result.txt --model co_training
```

Subtype prediction (comma-separated lists must match in length):

```bash
python run_caser.py --mode subtype --known ./data/cERs_gene.txt,./data/readers_gene.txt,./data/remodelers_gene.txt,./data/erasers_gene.txt,./data/writers_gene.txt --unknown ./data/unknown_cERs_gene.txt,./data/unknown_readers_gene.txt,./data/unknown_remodelers_gene.txt,./data/unknown_erasers_gene.txt,./data/unknown_writers_gene.txt --out ./result/cERs_final_result.txt,./result/readers_final_result.txt,./result/remodelers_final_result.txt,./result/erasers_final_result.txt,./result/writers_final_result.txt
```

### A) Train on new data and predict (recommended workflow)

CASER is semi-supervised: it trains on **labeled** data and uses **unlabeled** data for prediction in the same run.

Step 1: Prepare data files
- **Known (labeled) file**: tab-separated, includes a `label` column (1 = cancer-related, 0 = not cancer-related)
- **Unknown (unlabeled) file**: same feature columns, **no** `label` column

Step 2: Run the prediction script (this trains + predicts in one run)

```python
from CASER import predict

known_gene_path = "./data/your_known_gene.txt"
unknown_gene_path = "./data/your_unknown_gene.txt"
res_path = "./result/your_result.txt"

predict(known_gene_path, unknown_gene_path, res_path)
print("Done. Results saved to:", res_path)
```

Run it:

```bash
python -c "from CASER import predict; predict('./data/your_known_gene.txt','./data/your_unknown_gene.txt','./result/your_result.txt')"
```

### B) Train a model on new data and save it (advanced)

If you want to **train once** and **reuse** the trained model later, you can call the model API directly and save it with `pickle`.

```python
import pandas as pd
import pickle
import utils.ML_models as ml

known_gene_path = "./data/your_known_gene.txt"
unlabeled_pool_path = "./data/your_unlabeled_pool.txt"
model_path = "./result/caser_model.pkl"

df = pd.read_csv(known_gene_path, sep="\t", index_col=0)
X = df.drop(columns=["label"]).values
y = df["label"].values

unlabeled_X = pd.read_csv(unlabeled_pool_path, sep="\t", index_col=0).values

pred_y, prob_y, trained_model = ml.tri_training(X, y, unlabeled_X)

with open(model_path, "wb") as f:
    pickle.dump(trained_model, f)

print("Model saved to:", model_path)
```

> Note: The current `predict()` function always trains internally. If you want **pure inference** with a saved model, load the model from `model_path` and call its `predict`/`predict_proba` methods directly, then follow the same result formatting as in `CASER.py`.  

### C) Subtype-specific prediction (writers/erasers/readers/remodelers)

```python
from CASER import predict_subtype

known_gene_paths = [
    "./data/cERs_gene.txt",
    "./data/readers_gene.txt",
    "./data/remodelers_gene.txt",
    "./data/erasers_gene.txt",
    "./data/writers_gene.txt",
]

unknown_gene_paths = [
    "./data/unknown_cERs_gene.txt",
    "./data/unknown_readers_gene.txt",
    "./data/unknown_remodelers_gene.txt",
    "./data/unknown_erasers_gene.txt",
    "./data/unknown_writers_gene.txt",
]

result_paths = [
    "./result/cERs_final_result.txt",
    "./result/readers_final_result.txt",
    "./result/remodelers_final_result.txt",
    "./result/erasers_final_result.txt",
    "./result/writers_final_result.txt",
]

predict_subtype(known_gene_paths, unknown_gene_paths, result_paths)
```

### D) Switch to other models

```python
from CASER import predict
import utils.ML_models as ml

known_gene_path = "./data/your_known_gene.txt"
unknown_gene_path = "./data/your_unknown_gene.txt"
res_path = "./result/your_result.txt"

# Example: use co-training instead of tri-training
model = ml.co_training
predict(known_gene_path, unknown_gene_path, res_path, model=model)
```

## Input File Specifications

### Known Gene Set (`known_gene_path`)

Training data with known cancer association labels:

```
Gene    VEST_score_missense    VEST_score_frameshift_indels    ...    label
A1CF    0.4211                 0.0059                          ...    1
AAAS    0.3861                 0.0347                          ...    0
AAK1    0.2987                 0.0136                          ...    0
```

- `label`: 1 = cancer-related, 0 = not cancer-related

### Unknown Gene Set (`unknown_gene_path`)

Candidate genes for prediction:

```
Gene      VEST_score_missense    VEST_score_frameshift_indels    VEST_score_inframe_indels    ...    SEA_super_enhancer_percentage
A1CF      0.4211                 0.0059                          0                          ...    0.2411
ABRAXAS2  0.3365                 0.0126                          0                          ...    0.2057
ACTB      0.4931                 0.0187                          0.0157                       ...    0.8298
ACTL6B    0.3972                 0.0035                          0                          ...    0.078
```

## Output File Format

The prediction results include confidence scores and statistical significance measures:

```
Gene      Score          p-value              q-value               is_cancer_related
ZNF217    0.76280177     0.0001692126179831   0.009896958830103773  1
SATB2     0.7569754      0.0002654619970193   0.009896958830103773  1
...
CSNK2A3   0.2800701      0.984669957774466    0.9886165307515179    0
HSPA1A    0.2791872      0.9897121833084948   0.9916916076751118    0
TSSK6     0.25791362     0.9951704545454544   0.9951704545454544    0
```

- **Score**: Prediction confidence (higher = more likely cancer-associated)
- **p-value**: Statistical significance of prediction
- **q-value**: False Discovery Rate (FDR)-adjusted p-value
- **is_cancer_related**: Binary classification (1 = cancer-associated, 0 = not associated)

## Usage Examples

### Single Prediction Task

```python
from CASER import predict, predict_subtype
from utils import DL_models as dl, ML_models as ml

known_gene_path = '../../data/geneSet.txt'  # Training data with labels
unknown_gene_path = '../../data/unknown_gene.txt'  # Data to be predicted
res_path = '../../result/final_result.txt'  # Output file path

predict(known_gene_path, unknown_gene_path, res_path)
```

### Subtype-Specific Prediction

For predicting cERs within specific functional categories:

```python
from CASER import predict_subtype
import utils.ML_models as ml
import utils.DL_models as dl

# Define paths for different functional categories
known_gene_path = ['./data/cERs_gene.txt',
                   './data/readers_gene.txt', './data/remodelers_gene.txt',
                   './data/erasers_gene.txt', './data/writers_gene.txt']

unknown_gene_path = ['./data/unknown_cERs_gene.txt',
                     './data/unknown_readers_gene.txt', './data/unknown_remodelers_gene.txt',
                     './data/unknown_erasers_gene.txt', './data/unknown_writers_gene.txt']

res_path = ['./data/cERs_final_result.txt',
            './data/readers_final_result.txt', './data/remodelers_final_result.txt',
            './data/erasers_final_result.txt', './data/writers_final_result.txt']

predict_subtype(known_gene_path, unknown_gene_path, res_path)
```

### Custom Model Selection

Switch between different machine learning approaches:

```python
from CASER import predict
import utils.ML_models as ml
import utils.DL_models as dl

known_gene_path = './data/geneSet.txt'
unknown_gene_path = './data/unknown_gene.txt'
res_path = './result/final_result.txt'

# Use default tri-training model
model = ml.tri_training

# Alternative models available:
# Deep semi-supervised: TemporalEnsembling, FlexMatch, VAT, MixMatch, LadderNetwork, UDA
# Machine semi-supervised: SemiBoost, Tri_Training, Co_Training, LapSVM, Assemble, TSVM, SSGMM
# Supervised: GDBT, Xgboost, LogisticRegression, RandomForestClassifier, SVC

predict(known_gene_path, unknown_gene_path, res_path, model=model)
```

## Validation and Performance
```python
import numpy as np
from utils.tools import evaluate_binary_classifier

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
```
CASER has been rigorously validated through multiple approaches:

- **Computational benchmarking**: Superior performance on independent gene sets compared to alternative methods
- **Functional analysis**: Novel cERs demonstrate comparable cancer-driving potential to known driver genes
- **Network analysis**: Enrichment of cERs in gene-drug network modules, suggesting therapeutic relevance

**The example code is in the example.py file; it's recommended to start from this part.**


## Applications

- **Cancer biomarker discovery**: Identify novel diagnostic and prognostic markers
- **Therapeutic target prioritization**: Focus on high-confidence cERs for drug development
- **Functional genomics**: Understand the role of epigenetic regulators in cancer biology
- **Personalized medicine**: Integrate with patient omics data for precision oncology

## System Requirements

- Python 3.7+
- Machine learning libraries (scikit-learn 1.2.1, PyTorch 2.7)
- Data processing libraries (pandas 2.2.2, numpy 1.26.4)

## Citation

If you use CASER in your research, please cite:

**A semi-supervised model with multi-omics data integration prioritizes cancer-associated epigenetic regulator genes**.*PLOS Computational Biology*, under review

## Contact and Support

For questions, suggestions, or technical support, please contact the development team.

lvjie@wiucas.ac.cn,lh0083@126.com

------

*Note: This tool represents a significant advancement in computational cancer genomics and provides valuable resources for understanding the role of epigenetic regulators in cancer biology.*

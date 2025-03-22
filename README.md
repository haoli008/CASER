# CASER
Prioritizing a reliable list of cancer-associated Epigenetic Regulators (cERs) is important for cancer diagnosis and drug target discovery. Though many ERs have been suggested to have cancer driver roles, we anticipate more cERs to be discovered through computational analyses. Many cERs are not associated with dysregulated cancer phenotype. Here, we proposed a semi-supervised machine-learning method based on the tri-training model, and termed as Cancer-Associated Epigenetic Regulator identification (CASER). CASER integrated relatively comprehensive omics data including mutation, genomic, epigenetic and transcriptomic features and prioritize cERs as well as the four categories of functional roles. CASER achieved the better overall performance for the prioritization of unbiased cERs when evaluating on an independent gene set compared with various other machine-learning and deep-learning models. The identified novel cERs demonstrate comparable cancer driving potential and cell survival essentiality compared with known cancer driver genes. We also successfully validated two potential cERs in two cell lines, suggesting the usefulness of the CASER approach. In addition, cERs are enriched in modules in the gene-drug network. Therefore, our study suggests the feasibility of identifying cERs based on the integration of a large number of omics features by a semi-supervised model. The identified cERs could be the valuable resources for further functional studies, providing a better understanding of the role of cERs in cancer biology.
## usage
```python
CASER(known_gene_path,unknown_gene_path,res_path)
```
## 
## usage
```python
CASER(known_gene_path,unknown_gene_path,res_path)
```
### Example
## usage
```python
from CASER import CASER
# 
known_gene_path = './data/geneSet.txt' # original data path
unknown_gene_path = './data/unknonwn_gene.txt' # Data path that need to be predicted
res_path = './result/final_result.txt' # Result file path
CASER(known_gene_path,unknown_gene_path,res_path) 
```
## 
 

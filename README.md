# Imputation Strategies for Complex Multimodal Data in Rare Disease Research

This work investigates the challenges posed by highly incomplete datasets in rare disease research and compares various imputation methods.

## Repository Folder Structure

```
.
├── docs              # Information on dataset structure
├── notebooks         # Data summary and merging notebooks
├── results           # Imputation performance results
└── src               # Source code of all the experiments
```

The subfolders src and results include their own README files that explains their contents in more detail.

## Abstract
Rare diseases present unique challenges due to their rareness, leading to diagnostic delays and limited research opportunities. The integration of multimodal data can help with rare disease research and understanding these diseases, but highly incomplete datasets stand as a critical obstacle in this regard. Therefore, this thesis compares traditional imputation methods (mean imputation, KNN imputation, MICE, MissForest) and deep learning-based methods (JAMIE and AutoComplete) on the SCIVIAS rare disease dataset, which consists of HPO terms, diagnosis labels, genes with CADD scores, and protein measurements. Results show that tree-based methods such as MissForest perform consistently well across various levels of missingness. As for deep learning-based methods, JAMIE shows relatively stable performance, but AutoComplete has a steep performance drop with increasing levels of missingness. In addition, this thesis investigates whether the process of embedding diagnosis labels improves imputation results. The results reveal no clear advantages. In conclusion, the findings lay out noteworthy implications for understanding trade-offs between imputation performance and computational cost and give an overview of insights to help inform the selection of imputation strategies on multimodal datasets.
# CSV Data Analysis Summary

## 1. Dataset Overview

## 1.1. Subjects
- **Rows**: 2444
- **Columns**: 1
    - `SCIDPSEUDONYM`: Patient identifier (int64)

---

## 1.2. Labevents_HPO
Clinical lab data (urine and blood tests) and unstructured data from medical history questionnaires and doctor’s letters, translated into HPO terms. HPO IDs are from [HPO JAX](https://hpo.jax.org).

- **Rows**: 2438
- **Columns**: 2
    - `SCIDPSEUDONYM`: Patient identifier (int64)
    - `hpoTermId`: HPO ID of the phenotype
- **Unique patient IDs**: 1158

**Example Data**:

| SCIDPSEUDONYM | hpoTermId  |
|---------------|------------|
| 0             | HP:0031808 |
| 1             | HP:0031808 |
| 2             | HP:0012101 |

### Most frequent HPO terms**:

| hpoTermId   | Count
|-------------| -----
| HP:0031808  |  939
| HP:0032437  | 821
| HP:[id]   | 264
| HP:[id]  |   69
| HP:[id]  |   47
| HP:[id]  |   40
| HP:[id]    | 39    
| HP:[id]    | 21    
| HP:[id]    | 18    
| HP:[id]    | 18    

- Number of unique hpoTermId values: 41
- Number of hpoTermId values that appear only once: 6

---

## 1.3. Diag
- **Rows**: 2483
- **Columns**: 4
    - `subject_id`: Patient identifier (int64)
    - `diag`: Name of the diagnosis
    - `genetically_confirmed`: Confirmation through genetic testing, marked as Yes, No or Unknown
    - `testset`: Study group, marked as Yes or No
- **Unique patient IDs**: 2217

**Example Data**:

| subject_id | diag  | genetically_confirmed | testset  |
|---------------|------------|---------------|------------|
| 0             | cholestanol storage disease | yes            | no |
| 1            | [disease label]  | no            | no |
| 2            | [disease label]  | no            | no |

### Diagnosis Category Distribution

| Diagnosis                | Count |
|--------------------------|-------|
| control                  | 312   |
| cystic fibrosis          | 100   |
| [disease label]           | 100   |
| [disease label]          | 48    |
| [disease label] | 43    |

- Number of unique diag values: 657
- Number of diag values that appear only once: 392

**Yes/No Distribution in Genetically Confirmed**

| Response | Count 
|----------|-------
| Yes      | 468  
| No       | 2013   

**Yes/No Distribution in Testset**

| Response | Count 
|----------|-------
| Yes      | 247  
| No       | 2236  

---

## 1.4. Gene
- **Rows**: 1284
- **Columns**: 3
    - `SCIDPSEUDONYM`: Patient identifier (int64)
    - `gene`: Name of the gene
    - `cadd`: CADD score
- **Unique patient IDs**: 82

**Example Data**:
| SCIDPSEUDONYM | gene   | cadd |
|---------------|--------|------|
| 0          | [gene]  | 6,23    |
| 1          | [gene]   | 7,62   |
| 2          | [gene]  | 17,33  |

### Most Frequent Genes

| Gene       | Count |
|------------|-------|
| [gene]    | 68    |
| [gene]     | 62    |
| [gene]     | 56    |
| [gene]   | 55    |
| [gene]      | 40    |

- Number of unique gene values: 463
- Number of gene values that appear only once: 330

### CADD Summary Statistics 

| Statistic  | Value      |
|------------|------------|
| count      | 1284 |
| mean       | 7.54    |
| std        | 1.59   |
| min        | 6.01    |
| 25%        | 6.54    |
| 50%        | 7.19    |
| 75%        | 7.89    |
| max        | 17.33   |

---

## 1.5. ProteinsUrin
- **Rows**: 407003
- **Columns**: 3
    - `SCIDPSEUDONYM`: Patient identifier (int64)
    - `Gene`: Name of the gene
    - `VALUE`: Quantitative values for each subject (Only description available for the values)
- **Unique patient IDs**: 573

**Example Data**:
| SCIDPSEUDONYM | Gene   | VALUE |
|---------------|--------|------|
| 0          | [protein]  | 19.158146    |
| 1          | [protein]   | 12.043361   |
| 2          | [protein] | 13.276957 |

### Most Frequent Protein

| Gene       | Count |
|------------|-------|
| [protein]    | 547    |
| [protein]     | 547    |
| [protein]     | 547    |
| [protein]   | 547    |
| [protein]   | 547    |

- Number of unique Gene values: 2016
- Number of Gene values that appear only once: 11

### VALUE Summary Statistics

| Statistic    | Value         |
|--------------|---------------|
| count                | 407,003       |
| mean                 | 3,287.151     |
| std          | 246,222.5     |
| min              | -6.388233     |
| 25%      | 11.15848      |
| 50% | 12.72573  |
| 75%      | 14.68111      |
| max             | 52,057,930.0  |

---

## 2. Data Overlaps

| Overlap Combination                       | Count of Subjects |
|-------------------------------------------|-------------------|
| Subjects and Diagnosis                    | 2217 subjects     |
| Subjects and HPO                          | 1158 subjects     |
| Subjects and Gene                         | 82 subjects       |
| Subjects and Protein                      | 573 subjects      |
| Diagnosis and HPO                         | 1053 subjects     |
| Diagnosis and Gene                        | 71 subjects       |
| Diagnosis and Protein                     | 512 subjects      |
| HPO and Gene                              | 76 subjects       |
| HPO and Protein                           | 367 subjects      |
| Gene and Protein                          | 35 subjects       |
| Subjects and Diagnosis and HPO            | 1053 subjects     |
| Subjects and Diagnosis and Gene           | 71 subjects       |
| Subjects and Diagnosis and Protein        | 512 subjects      |
| Subjects and HPO and Gene                 | 76 subjects       |
| Subjects and HPO and Protein              | 367 subjects      |
| Subjects and Gene and Protein             | 35 subjects       |
| Diagnosis and HPO and Gene                | 67 subjects       |
| Diagnosis and HPO and Protein             | 327 subjects      |
| Diagnosis and Gene and Protein            | 29 subjects       |
| HPO and Gene and Protein                  | 34 subjects       |
| Subjects and Diagnosis and HPO and Gene   | 67 subjects       |
| Subjects and Diagnosis and HPO and Protein| 327 subjects      |
| Subjects and Diagnosis and Gene and Protein | 29 subjects    |
| Subjects and HPO and Gene and Protein     | 34 subjects       |
| Diagnosis and HPO and Gene and Protein    | 28 subjects       |
| Subjects and Diagnosis and HPO and Gene and Protein | 28 subjects |

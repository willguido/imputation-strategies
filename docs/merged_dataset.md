# Merged Data

| SCIDPSEUDYNOM | Diagnosis                              | HPO_<id> | ... | Gene_<gene>       | ... | Protein_<protein>    |
|---------------|---------------------------------------|----------------|-----|------------------|-----|------------------|
| Subject_1     | Cholestanol storage disease (Genetically confirmed: Yes); Another Dx | 1.0            | ... | 7.89  | ... | 17.42    |

### Column Encoding Descriptions
- **SCIDPSEUDYNOM**: Unique identifier for subjects
- **Diagnosis**: Text-encoded as `<diagnosis_description> (Genetically confirmed: yes/no)` for each subject. Multiple diagnoses are separated by semicolons.
- **HPO Columns**: Binary encoded (`1.0` for presence, `0.0` for absence) for each HPO term.
- **Gene Columns**: Numeric values representing the CADD score for each gene.
- **Protein Columns**: Numeric values representing the measured VALUE for each protein.

### Dataset Dimensions
- **Number of Rows (Subjects):** 2345
- **Number of Columns:** 2522
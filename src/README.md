# Source Folder

This folder includes:

- code ran for deep learning-based imputation methods JAMIE and AutoComplete. 
- scripts for diagnosis embedding generation experiments, imputation with KNN, predicting without imputation, and evaluation of results with DOID distances.
- implementation of traditional imputation methods mean imputation, KNN, MICE, and MissForest.
- general utility functions for evaluation, imputation, masking, and subsetting.

Additional notes:
- To run the scripts for deep learning-based methods, the repositories of JAMIE and AutoComplete must be cloned onto your machine, and the provided files should be placed within those repositories.
- As the data files cannot be shared, their paths are removed and replaced with descriptions of the expected data instead.
- The main purpose of these files are to document the implementation details. They may need configuration. They are not a plug-and-play solution. One may need to adapt them to their environment.
- `requirements.txt` files are created with `conda list --export > requirements.txt`. Some packages may need adjustments depending on your system and Python version.
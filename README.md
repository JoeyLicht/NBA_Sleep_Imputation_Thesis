# NBA_Sleep_Imputation_Thesis

## Code

### athletic_performance.py
Cleans, merges, and normalizes athletic and sleep data, then runs regression analyses to study the relationship between sleep features and athletic performance metrics. 

### dataset_description.py
Provides detailed descriptions of the datasets used, including the structure, variables, and any preprocessing steps taken to prepare the data for analysis.

### df_cleaning.py
Performs various data cleaning tasks, including date and time conversion, duplicate removal, and column filtering for NBA sleep and athletic data. It also merges multiple datasets, describes column characteristics, and includes helper functions for data conversion and saving.

### matrix_completion.py
Defines a class to identify similar sleep patterns for imputation by normalizing data, creating heatmaps of logged nights, and generating matrices of sleep features by weekday.

### sleep_impution.py
Classes and functions to analyze and impute missing sleep data using various methods such as KNN, linear interpolation, and quadratic/linear fitting. It includes tools for data normalization, plotting, and performing full analysis on individual and collective datasets.

### utils.py
Loading and saving utility functions.


## Results (Not Uploaded to Git)

### Athletic Performance Correlation
- Sleep to athletic performance correlation (`athletic_performance.py`).

### By Day
- Tracks differences in sleep by day of the week.

### Final Thesis Figures
- Contains plots used in licht_lichtj_meng_EECS_2024_thesis.pdf.

### Sleep Imputation
- Results for sleep tracking imputation (`sleep_imputation.py`).

## Data (Not Uploaded to Git)
Contains the project datasets, both as raw data and cleaned versions (cleaned using `df_cleaning.py`).

# Suicide Study in Poland
## Overview
This project focuses on analyzing suicide incidents in Poland using data compiled by the National Police Headquarters. This dataset represents one of the most comprehensive collections of suicide attempts and fatalities in the country, spanning from 2013 to 2023. The data originates directly from police reports, which are uniformly written across all regions, ensuring consistency and accuracy. The dataset is regularly updated on an annual basis, making it a valuable resource for understanding suicide trends in Poland.

## Data
The study uses a dataset gathered by the National Police Headquarters, which has been collecting data on suicide incidents since 1999. However, for the purposes of this study, the focus is on data from 2013 to 2023. The dataset includes detailed reports from police officers, incorporating information from various sources, including family members, the individual attempting the suicide, witnesses, and farewell letters.

## Repository Structure
- `src/`: Contains Python and R scripts for data preparation, analysis, and model development.
	- `python_project/`: Python scripts for data imputation, mapping, analysis, feature engineering and logistic regression modeling.
		- `config.py`: Configuration settings for the Python project.
		- `data_imputation.py`: Script for imputing missing data using probability distributions.
		- `data_mapping.py`: Script for mapping raw data into structured formats.
		- `data_summary.py`: Script for generating summary statistics from the data.
		- `data_visualizations.py`: Script for generating visualizations related to the dataset.
		- `feature_engineering.py`: Script for performing feature engineering to prepare data for modeling.
		- `logreg_model.py`: Script for building and evaluating a logistic regression model.
	- `r_project/`: R script for latent class analysis (LCA).
		- `lca_analysis_poLCA.R`: R script for performing latent class analysis using the poLCA package.
- `ippan_suicide_study_python_env.yml`: Environment configuration file for Python dependencies.
- `ippan_suicide_study_r_env.yml`: Environment configuration file for R dependencies.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Hubert26/ippan_suicide_study/blob/main/LICENSE.txt) file for details.

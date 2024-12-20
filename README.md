# Suicide Study in Poland
## Overview
This project focuses on analyzing suicide incidents in Poland using data compiled by the National Police Headquarters. This dataset represents one of the most comprehensive collections of suicide attempts and fatalities in the country, spanning from 2013 to 2023. The data originates directly from police reports, which are uniformly written across all regions, ensuring consistency and accuracy. The dataset is regularly updated on an annual basis, making it a valuable resource for understanding suicide trends in Poland.

## Data
The study uses a dataset gathered by the National Police Headquarters, which has been collecting data on suicide incidents since 1999. However, for the purposes of this study, the focus is on data from 2013 to 2023. The dataset includes detailed reports from police officers, incorporating information from various sources, including family members, the individual attempting the suicide, witnesses, and farewell letters.

## Repository Structure
This repository has the following structure:

- `.git/`: Contains version control information.
- `.vscode/`: Configuration files for Visual Studio Code.
- `data/`: Contains datasets used in the analysis.
- `docs/`: Documentation related to the project.
- `notebooks/`: Jupyter notebooks for drafts and exploratory analysis
- `results/`: Output results from analyses.
- `src/`: Source code for data preparation, analysis, and modeling.
    - `data_processing/`: Scripts for processing data.
    - `models/`: Scripts related to model development.
    - `visualizations/`: Scripts for generating visualizations.
- `.env`: Environment variables configuration.
- `.gitignore`: Specifies files to ignore in version control.
- `LICENSE.txt`: Licensing information.
- `README.md`: Project documentation.
- `environment.yml`: Configuration for environment setup.
- `ippan_suicide_study.code-workspace`: Workspace configuration for the IDE.

## Installation
### 1. Clone the repository:
   ```bash
   git clone https://github.com/Hubert26/ippan_suicide_study.git

### 2. Create and Activate the Environment
+ Navigate to the ippan_suicide_study project directory:
```
cd /path/to/ippan_suicide_study
```
+ Create the Conda environment from the `environment.yml` file:
```
conda env create -f environment.yml
```
+ Activate the environment:
```
conda activate pytho_r_env
```

### 3. Configure R in Jupiter Notebook
+ In terminal run `R`
+ Install IRkernel
```
IRkernel::installspec(name = "python_r_env", displayname = "R (python_r_env)")
```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Hubert26/ippan_suicide_study/blob/main/LICENSE.txt) file for details.

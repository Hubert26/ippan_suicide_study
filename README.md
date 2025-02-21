# Suicide Study in Poland
## Overview
This project focuses on analyzing suicide incidents in Poland using data compiled by the National Police Headquarters. This dataset represents one of the most comprehensive collections of suicide attempts and fatalities in the country, spanning from 2013 to 2023. The data originates directly from police reports, which are uniformly written across all regions, ensuring consistency and accuracy. The dataset is regularly updated on an annual basis, making it a valuable resource for understanding suicide trends in Poland.

## Data
The study uses a dataset gathered by the National Police Headquarters, which has been collecting data on suicide incidents since 1999. However, for the purposes of this study, the focus is on data from 2013 to 2023. The dataset includes detailed reports from police officers, incorporating information from various sources, including family members, the individual attempting the suicide, witnesses, and farewell letters.

## Repository Structure
This repository has the following structure:

- `README.md`: Project documentation.
- `LICENSE.txt`: Licensing information.
- `.git/`: Contains version control information.
- `.gitignore`: Specifies files to ignore in version control.
- `.vscode/`: Configuration files for Visual Studio Code.
- `.devontainer.json`: Configuration for developing inside a container using VS Code Remote - Containers extension.
- `docker-compose.yml`: Manages multi-container Docker applications and sets up the development environment.
- `Dockerfile`: Defines the Docker image, including dependencies and environment setup.
- `pyproject.toml`: Configuration for Python tools such as Black and Ruff.
- `environment.yml`: Configuration for environment setup.
- `.env`: Environment variables configuration.
- `settings.yaml`: Configuration settings for data paths
- `data/`: Contains datasets used in the analysis.
- `results/`: Output results from analyses.
- `plots/`: Visualizations
- `notebooks/`: Jupyter notebooks for drafts and exploratory analysis
- `src/`: Source code for data preparation, analysis, and modeling.
    - `helpers/`: Utility functions and configuration
    - `data_processing/`: Scripts for processing data.
    - `models/`: Scripts related to model development.
    - `visualizations/`: Scripts for generating visualizations.

## Installation
### 1. Clone the repository:
   ```bash
   git clone https://github.com/Hubert26/ippan_suicide_study.git
   ```

### 2. Create and Activate the Environment
+ Navigate to the ippan_suicide_study project directory:
    ```bash
    cd /path/to/ippan_suicide_study
    ```
### 3. Set up Docker Environment
Ensure you have Docker and Docker Compose installed on your system.

### 4. Build and Run the Docker Container
    ```bash
    docker compose up --build
    ```
    
This will:
+ Create a container with all necessary dependencies for Python, R, and Jupyter Notebook.
+ Install all packages specified in environment.yml.
+ Start the development environment in a containerized setup.

### Notes:
+ All code changes and installed packages persist because the project directory is mounted as a volume in Docker.
+ This setup uses PYTHONPATH=app/, so you can import modules using the structure relative to the app/ folder.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Hubert26/ippan_suicide_study/blob/main/LICENSE.txt) file for details.

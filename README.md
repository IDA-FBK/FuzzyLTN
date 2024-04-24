# FuzzyLTN

A novel interpretable and adaptable neuro-symbolic framework that integrates Fuzzy Systems with Logical Tensor Networks

## Directories

- **data/**: Contains everything related to data.
  - *Description*: This directory contains the script for loading the datasets. 

- **experiments/**: Contains everything related to experiments.
  - *Description*: This directory includes configurations, results, and evaluation scripts.

- **models/**: Contains models and operators.
  - *Description*: Includes models used and operators (e.g., "and" and "or").

## Running the Code

### 1. Set up the Project Environment

- First, ensure you have Conda installed. If not, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- Then, create the project environment using the provided fuzzyltn_env.yml file:
  ```bash
  conda env create -f fuzzyltn_env.yml
  ```

- And activate it:

  ```bash
  conda activate fuzzyltn
  ```

### 2. Set up Configuration

- Ensure you have a configuration file located in the `experiments/configurations/<dataset>/` directory.

- This file should contain experiment settings such as number of seeds, neuron types, number of membership functions (MFs), activation function, and optimizer (see conf file inside `experiments/configurations/iris/` ).

### 3. Command-line Arguments

- The script accepts command-line arguments to specify dataset, path to configuration file, and directory to store results.

- Use the following command-line arguments:
  - `-dataset`: Specify the dataset to use (default is "iris").
  - `-path_to_conf`: Provide path to configuration file (default is `./experiments/configurations/iris/conf-00.json`).
  - `-path_to_results`: Define directory where results will be saved (default is `./experiments/results/iris/`).

### 4. Run the Script

- Run the main script using Python:
  ```bash
  python main.py -dataset <dataset> -path_to_conf ./experiments/configurations/<dataset>/<name_of_conf>.json -path_to_results ./experiments/results/<dataset>/
  ```

### 5. Results 

Results are stored in `./results/<dataset>/` directory. This directory contains:

- Plots, fuzzy rules and axioms generated during each experiment.
- `runs_results.csv`: A CSV file storing the results of each run.
- `mean_std_results.csv`: A CSV file storing the mean and standard deviation of grouped runs (by NeuronType and MFS).


## Authors
- Paulo Vitor De Campos Souza: pdecampossouza@fbk.eu
- Gianluca Apriceno: apriceno@fbk.eu
- Mauro Dragoni: dragoni@fbk.eu
## License
For open source projects, say how it is licensed.

import argparse
import os.path

import numpy as np
import pandas as pd
from data.data import get_data
from experiments.configurations.configurations import get_configuration
from experiments.evaluation import evaluate_interpretability
from experiments.calculate import calculate_avg_results
from experiments.utils import save_list_in_a_file
from models.models import FNNModel


def run_experiment(
    train_data,
    test_data,
    neuron_type,
    num_mfs,
    activation,
    optimizer,
    i_seed,
    rng_seed,
    results_df,
    path_to_results,
):
    """
    Run an experiment with the given configuration and save the results.

    Parameters:
    - train_data (tuple): Tuple containing features and labels for the training set.
    - test_data (tuple): Tuple containing features and labels for the testing set.
    - neuron_type (str): Type of neuron to use in the FNN model.
    - num_mfs (int): Number of membership functions for each input dimension.
    - activation (str): Activation function to use in the FNN model.
    - optimizer (str): Optimizer algorithm to use for training the FNN model.
    - i_seed (int): Seed for the experiment.
    - rng_seed (int): Seed for random number generation in the FNN model.
    - results_df (DataFrame): DataFrame to store the results of each experiment.
    - path_to_results (str): Path to the directory where experiment results will be saved.

    Returns:
    None
    """

    exp_str = f"/exp-seed_{i_seed}_neurontype_{neuron_type}_nummfs_{num_mfs}_activation_{activation}/"
    path_to_exp_results = path_to_results + exp_str

    if not os.path.exists(path_to_exp_results):
        os.makedirs(path_to_exp_results, exist_ok=True)

    x_train, y_train = train_data[0], train_data[1]
    x_test, y_test = test_data[0], test_data[1]

    print(f"\n---\nModel: {neuron_type} with {num_mfs} MFs")
    fnn_model = FNNModel(
        num_mfs=num_mfs,
        neuron_type=neuron_type,
        activation=activation,
        optimizer=optimizer,
        visualizeMF=False,
        rng_seed=rng_seed,
    )

    print("\nSummary of Performance Metrics:")
    fnn_model.train_model(x_train, y_train)
    evaluation_metrics = fnn_model.evaluate_model(x_test, y_test)
    rules = fnn_model.generate_fuzzy_rules()

    # Save fuzzy rules to a file
    save_list_in_a_file(rules, path_to_exp_results + "fuzzy_rules.txt")

    fnn_model.generate_fuzzy_axioms()
    # Save axioms in a file
    save_list_in_a_file(fnn_model.axioms, path_to_exp_results + "fuzzy_axiom.txt")

    results_df.loc[len(results_df)] = [
        i_seed,
        neuron_type,
        num_mfs,
        evaluation_metrics["accuracy"],
        evaluation_metrics["fscore"],
        evaluation_metrics["recall"],
        evaluation_metrics["precision"],
    ]

    evaluate_interpretability(fnn_model, x_test, path_to_exp_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dataset", type=str, default="liver", help="specify the dataset to use"
    )
    parser.add_argument(
        "-path_to_conf",
        type=str,
        default="./experiments/configurations/iris/conf-00.json",
        help="configuration file for the current experiment",
    )
    parser.add_argument(
        "-path_to_results",
        type=str,
        default="./experiments/results/liver3/",
        help="directory where to store the results",
    )

    args = parser.parse_args()

    dataset = args.dataset
    path_to_conf = args.path_to_conf
    path_to_results = args.path_to_results
    conf = get_configuration(path_to_conf)

    # experiment setting
    num_seeds = conf["num_seeds"]
    neuron_types = conf["neuron_types"]
    num_mfs_options = conf["num_mfs_options"]
    activation = conf["activation"]
    optimizer = conf["optimizer"]

    data_train, data_test = get_data(dataset)

    # this store the results of each run
    results_df = pd.DataFrame(
        columns=[
            "Seed",
            "NeuronType",
            "MFs",
            "Accuracy",
            "Fscore",
            "Recall",
            "Precision",
        ]
    )

    for i_seed in range(num_seeds):
        # run_rng
        rng_seed = np.random.default_rng(i_seed)
        for neuron_type in neuron_types:
            for num_mfs in num_mfs_options:
                run_experiment(
                    data_train,
                    data_test,
                    neuron_type,
                    num_mfs,
                    activation,
                    optimizer,
                    i_seed,
                    rng_seed,
                    results_df,
                    path_to_results,
                )

    # save results
    results_df.to_csv(path_to_results + "runs_results.csv")
    # compute mean and sd
    calculate_avg_results(results_df, path_to_results)

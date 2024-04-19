import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

from experiments.calculate import calculate_consistency_matrix, calculate_distinguishability_matrix


def plot_distinguishability_heatmap(rules_dictionary, path_to_results):
    """
    Plot the distinguishability heatmap for the fuzzy rules.

    Parameters:
    - rules_dictionary (list): List containing dictionaries representing fuzzy rules.
    - path_to_results (str): Path to the directory where the heatmap image will be saved.

    Returns:
    None
    """

    # Distinguishability
    # if hasattr(fnn_model, "calculate_distinguishability_matrix"):
    distinguishability_matrix = calculate_distinguishability_matrix(rules_dictionary)
    x_max, y_max = distinguishability_matrix.shape
    # print("Distinguishability Matrix:\n", distinguishability_matrix)

    plt.figure(figsize=(10, 8))
    # Using seaborn for the heatmap, if available
    sns.heatmap(distinguishability_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=np.arange(1, x_max+1), yticklabels=np.arange(1, y_max+1))
    # Get the current Axes instance
    ax = plt.gca()
    # Set the ticks of the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.linspace(0, 1, num=6))

    plt.title("Fuzzy Rules Distinguishability Matrix")
    plt.xlabel("Rule Index")
    plt.ylabel("Rule Index")
    plt.savefig(path_to_results + "distinguishability_matrix.png")


def plot_consistency_matrix(V, path_to_results):
    """
    Plot the consistency matrix using a heatmap.

    Parameters:
        V (numpy.ndarray): The matrix of logical neuron outputs used for generating the consistency matrix.
        path_to_results (str): Path to the directory where the heatmap image will be saved.

    Returns:
        None
    """

    # Consistency
    # if hasattr(fnn_model, "calculate_consistency_matrix"):
    consistency_matrix = calculate_consistency_matrix(V)
    x_max, y_max = consistency_matrix.shape
    # print("Consistency Matrix:\n", consistency_matrix)

    plt.figure(figsize=(10, 8))

    sns.heatmap(consistency_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=np.arange(1, x_max+1), yticklabels=np.arange(1, y_max+1))
    # Get the current Axes instance
    ax = plt.gca()
    # Set the ticks of the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.linspace(0, 1, num=6))

    plt.title("Fuzzy Rules Consistency Matrix")
    plt.xlabel("Rule Index")
    plt.ylabel("Rule Index")
    plt.savefig(path_to_results + "consistency_matrix.png")


def plots_uncovered_samples(x_test, uncovered_samples_indices, path_to_results):
    """
    Plot the test samples with uncovered samples highlighted.

    Parameters:
        x_test (numpy.ndarray): Test samples.
        uncovered_samples_indices (numpy.ndarray): Indices of uncovered samples.
        path_to_results (str): Path to the directory where the plot will be saved.

    Returns:
        None
    """

    # Plot all samples and highlight those not covered
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test[:, 0], x_test[:, 1], alpha=0.5, label="Covered Samples")
    plt.scatter(
        x_test[uncovered_samples_indices, 0],
        x_test[uncovered_samples_indices, 1],
        color="red",
        label="Uncovered Samples",
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Test Samples with Uncovered Samples Highlighted")
    plt.savefig(path_to_results + "test_samples_and_uncovered.png")


def plot_similarity_matrix(similarity_matrix, path_to_results):
    """
    Plots a similarity matrix using a heatmap.

    Parameters:
        similarity_matrix (numpy.ndarray): A numpy array representing the similarity matrix.
        path_to_results (str): Path to the directory where the plot will be saved.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    x_max, y_max = similarity_matrix.shape

    sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=np.arange(1, x_max+1), yticklabels=np.arange(1, y_max+1))
    # Get the current Axes instance
    ax = plt.gca()
    # Set the ticks of the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.linspace(0, 1, num=6))

    plt.title("Fuzzy Rules Similarity Matrix")
    plt.xlabel("Rule Index")
    plt.ylabel("Rule Index")
    plt.savefig(path_to_results + "similarity_matrix.png")


def visualize_membership_functions(feature_index, centers, sigmas):
    """
    Visualizes the membership functions for a given feature.

    Parameters:
        feature_index (int): Index of the feature.
        centers (numpy.ndarray): Array containing the centers of the membership functions.
        sigmas (numpy.ndarray): Array containing the standard deviations of the membership functions.

    Returns:
        None
    """
    x = np.linspace(centers[0] - 3 * sigmas[0], centers[-1] + 3 * sigmas[-1], 1000)
    plt.figure()
    for center, sigma in zip(centers, sigmas):
        y = norm.pdf(x, center, sigma)
        plt.plot(x, y, label=f"Center: {center:.2f}, Sigma: {sigma:.2f}")
    plt.title(f"Membership Functions for Feature {feature_index + 1}")
    plt.xlabel("Feature Value")
    plt.ylabel("Membership Degree")
    plt.legend()
    plt.show()


### THE FOLLOWING FUNCTIONS HAVE BEEN NOT USED YET, IF YOU WANT TO USE THEM DECOMMENT THEM AND ADD THE DOCUMENTATION

# def visualize_logic_neurons_output_heatmap(logic_outputs, path_to_results):
#     # Here, you would have the self.logic_output matrix filled with the outputs from the logical neurons
#     # Now, let's create the heatmap using plt.imshow to visualize self.logic_output
#     plt.figure(figsize=(10, 8))
#     # Assuming self.logic_output is 2D and we want to visualize the outputs per sample
#     plt.imshow(logic_outputs, aspect="auto", cmap="viridis")
#     plt.colorbar(label="Logical Neuron Output")
#     plt.title("Heatmap of Logical Outputs")
#     plt.xlabel("Logical Neuron Index")
#     plt.ylabel("Sample Index")
#     plt.show()

# def plot_rule_activation(input_data, rule_index):
#     """
#     Plot the activation level of a specific fuzzy rule for a set of samples.
#
#     Parameters:
#     - input_data: Input data, a NumPy array of shape (n_samples, 2). Currently supports 2 features for plotting.
#     - rule_index: Index of the fuzzy rule to visualize.
#     """
#     # Calculate the activation level of the specified rule for each sample
#     activations = calculate_rule_activation(input_data, rule_index)
#
#     # Create a scatter plot of the samples, coloring them by their activation level
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(input_data[:, 0], input_data[:, 1], c=activations, cmap="viridis", s=50, alpha=0.5, edgecolor="k")
#
#     plt.colorbar(scatter, label="Activation Level")
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.title(f"Activation Levels for Rule {rule_index}")
#     plt.show()

# def plot_e_completeness(e_completeness_score):
#     """
#     Plots a bar chart showing the percentage of e-completeness.
#
#     :param e_completeness_score: The calculated e-completeness percentage.
#     """
#     fig, ax = plt.subplots()
#     ax.bar(["e-Completeness"], [e_completeness_score], color="skyblue")
#     ax.set_ylabel("Percentage (%)")
#     ax.set_title("e-Completeness of Test Samples")
#     ax.set_ylim(0, 100)  # Set the Y-axis limit to 0-100%
#
#     # Add the percentage value above the bar
#     for i, v in enumerate([e_completeness_score]):
#         ax.text(i, v + 1, f"{v:.2f}%", ha="center", va="bottom")
#
#     plt.show()


# def plot_activation_heatmap(logic_outputs):
#     """
#     Plots a heatmap of neuron activation levels for the test samples.
#     Assumes `logic_outputs` contains the neuron activation levels.
#     """
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(logic_outputs, cmap="coolwarm", cbar_kws={"label": "Activation Level"})
#     plt.title("Neuron Activation Heatmap for Test Samples")
#     plt.xlabel("Neuron Index")
#     plt.ylabel("Test Sample Index")
#     plt.show()
#

##### ARE THESE TWO DIFFERENT?

# def plot_activation_heatmap2(x_test, rule_params):
#     """
#     Plots a heatmap of rule activation levels for the test samples.
#
#     :param X_test: The set of test samples.
#     :param rule_params: The rule parameters to calculate activation.
#     """
#     activation_matrix = np.array(
#         [
#             [calculate_rule_activation_index(rule) for rule in rule_params]
#             for input_vector in x_test
#         ]
#     )
#
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(
#         activation_matrix, cmap="coolwarm", cbar_kws={"label": "Activation Level"}
#     )
#     plt.title("Rule Activation Heatmap for Test Samples")
#     plt.xlabel("Rule Index")
#     plt.ylabel("Test Sample Index")
#     plt.show()

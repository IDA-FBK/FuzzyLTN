from experiments.calculate import (
    calculate_ecompleteness_and_get_uncovered_samples,
    calculate_similarity_matrix,
)

from experiments.plots import (
    plot_consistency_matrix,
    plot_distinguishability_heatmap,
    plot_similarity_matrix,
    plots_uncovered_samples,
)


def evaluate_interpretability(fnn_model, x_test, path_to_results, epsilon=0.136):
    """
    Evaluate the interpretability of the FNN model by analyzing various interpretability metrics.

    Parameters:
    - fnn_model (FNNModel): Instance of the FNN model.
    - x_test (numpy.ndarray): Features of the test dataset.
    - path_to_results (str): Path to the directory where evaluation results will be saved.
    - epsilon (float): Threshold for determining rule activation in e-completeness calculation. Default is 0.136.

    Returns:
    None
    """
    # if too slow comment this part

    # Similarity
    similarity_matrix = calculate_similarity_matrix(fnn_model.rules_dictionary)
    print("Similarity Matrix:\n", similarity_matrix)
    plot_similarity_matrix(similarity_matrix, path_to_results)

    # e-completeness
    fuzzy_outputs = fnn_model.fuzzification_layer(x_test)
    logic_outputs = fnn_model.logic_neurons_layer(fuzzy_outputs)
    completeness_score, uncovered_samples_indices = ( calculate_ecompleteness_and_get_uncovered_samples(logic_outputs, epsilon))

    print(f"e-Completeness Score: {completeness_score}%")

    plots_uncovered_samples(x_test, uncovered_samples_indices, path_to_results)

    # Distinguishability
    plot_distinguishability_heatmap(fnn_model.rules_dictionary, path_to_results)

    # Consistency
    plot_consistency_matrix(fnn_model.V, path_to_results)


### THE FOLLOWING FUNCTION HAS BEEN NOT USED YET, IF YOU WANT TO USE IT DECOMMENT IT AND ADD THE DOCUMENTATION

# def analyze_feature_sensitivity(model, x_base, feature_index, value_range, path_to_results):
#     """
#     Analyze the sensitivity of the model output to changes in a single feature.
#
#     Parameters:
#     - model: The FNN model instance.
#     - X_base: A base input array (one sample) to use for the analysis.
#     - feature_index: Index of the feature to analyze.
#     - value_range: Range of values (min, max) to test for the feature.
#     """
#     values = np.linspace(value_range[0], value_range[1], 100)
#     outputs = []
#
#     for value in values:
#         x_test = x_base.copy()
#         x_test[0, feature_index] = value
#         output = model.predict(x_test.reshape(1, -1))
#         outputs.append(output)
#
#     plt.plot(values, outputs)
#     plt.xlabel(f"Feature {feature_index} Value")
#     plt.ylabel("Model Output")
#     plt.title(f"Sensitivity of Model Output to Feature {feature_index}")
#     plt.savefig(path_to_results + "feature_sensitivity.png")

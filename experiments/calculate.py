import numpy as np


def calculate_avg_results(results_df, path_to_results):
    """
       Calculate the mean and standard deviation of resultsand save them to a CSV file.

       Parameters:
       - results_df (DataFrame): DataFrame containing results with columns: 'NeuronType', 'MFs', 'Accuracy', 'Fscore', 'Recall', 'Precision'.
       - path_to_results (str): Path to the directory where the CSV file with mean and standard deviation results will be saved.

       Returns:
       None
    """

    # Define column names for mean and standard deviation
    mean_std_cnames = ["mean", "std"]

    # Group by NeuronType and MFs, then calculate mean and standard deviation
    agg_grouped_results = results_df.groupby(["NeuronType", "MFs"]).agg({
        "Accuracy": mean_std_cnames, "Fscore": mean_std_cnames,
        "Recall": mean_std_cnames, "Precision": mean_std_cnames
    })
    # Rename columns for clarity
    agg_grouped_results.columns = [
        "Accuracy Mean", "Accuracy Std", "Fscore Mean", "Fscore Std", "Recall Mean", "Recall Std", "Precision Mean",
        "Precision Std"
    ]

    print(agg_grouped_results)
    # number of decimals is set to 3
    agg_grouped_results.to_csv(path_to_results + "mean_std_results.csv", float_format="%.3f")


def calculate_overlap(mu1, sigma1, mu2, sigma2):
    """
    Calculates the overlap between two Gaussian membership functions.

    Parameters:
    - mu1, sigma1 (float): Mean and standard deviation of the first Gaussian membership function.
    - mu2, sigma2 (float): Mean and standard deviation of the second Gaussian membership function.

    Return:
    - overlap (float): The overlap between the two membership functions.
    """

    d = np.abs(mu1 - mu2)
    mean_sigma = (sigma1 + sigma2) / 2
    overlap = np.exp(-(d ** 2) / (2 * mean_sigma ** 2))
    return 1 - overlap  # Returning 1 - overlap to represent distinguishability


def calculate_e_completeness(Z, epsilon):
    """
    Calculates the e-completeness score for the test dataset based on the activations of the logical neurons.

    Parameters:
    - Z (numpy.ndarray): Matrix of logical neuron outputs for the test set, where each column represents a neuron.
    - epsilon (float) : Threshold for determining whether a rule has been activated.

    Return:
    - e_completeness_score (float): The percentage of samples that are "covered" by at least one activated rule above epsilon.
    """

    # Checks whether each rule has been activated above epsilon for each sample
    activations_above_epsilon = Z >= epsilon

    # Checks if at least one rule has been activated above epsilon for each sample
    samples_covered = activations_above_epsilon.any(axis=1)

    # Calculate the percentage of samples covered
    e_completeness_score = samples_covered.mean() * 100

    return e_completeness_score


def calculate_consistency_matrix(V):
    """
        Calculate a consistency matrix for the set of fuzzy rules based on the rule consequents.

        Parameters:
        - V (numpy.ndarray): An array representing the consequents of fuzzy rules. Each row represents a rule,
          and each column represents the consequent values for each dimension.

        Returns:
        - consistency_matrix (numpy.ndarray): A square matrix representing the consistency between fuzzy rules.
          Each element (i, j) in the matrix indicates the consistency between rule i and rule j based on their consequents.
          The values range from 0 to 1, where 0 indicates large difference (low consistency) and 1 indicates little
          or no difference (high consistency). The diagonal elements are set to 1, representing maximum consistency
          with oneself.
    """

    num_rules = len(V)
    consistency_matrix = np.zeros((num_rules, num_rules))

    for i in range(num_rules):
        for j in range(i + 1, num_rules):
            difference = np.abs(V[i] - V[j])
            consistency = np.exp(-difference)
            consistency_matrix[i, j] = consistency_matrix[j, i] = consistency

    for i in range(num_rules):
        consistency_matrix[i, i] = 1.0  # Maximum consistency with yourself

    return consistency_matrix


def calculate_distinguishability_matrix(rules):
    """
        Calculates a distinguishability matrix for a set of fuzzy rules based on their Gaussian membership functions.

        Parameters:
        - rules (list of dicts): List containing information about each fuzzy rule. Each dictionary should have keys
          "centers" and "sigmas", where "centers" is an array representing the means of Gaussian membership functions,
          and "sigmas" is an array representing the standard deviations of Gaussian membership functions.

        Returns:
        - distinguishability_matrix (numpy.ndarray): A square matrix representing the distinguishability between fuzzy rules.
          Each element (i, j) in the matrix indicates the distinguishability between rule i and rule j based on their
          Gaussian membership functions. The values range from 0 to 1, where 0 indicates no distinguishability (complete overlap),
          and 1 indicates maximum distinguishability (no overlap). The diagonal elements are set to 1, representing
          maximum distinguishability with oneself.
    """

    num_rules = len(rules)
    distinguishability_matrix = np.zeros((num_rules, num_rules))

    for i in range(num_rules):
        overlaps = []
        for j in range(i + 1, num_rules):
            mu1 = np.array(rules[i]["centers"])
            sigma1 = np.array(rules[i]["sigmas"])
            mu2 = np.array(rules[j]["centers"])
            sigma2 = np.array(rules[j]["sigmas"])

            overlap = calculate_overlap(mu1, sigma1, mu2, sigma2)
            overlaps.append(overlap)

            if overlaps:
                average_overlap = np.mean(overlaps)
            else:
                average_overlap = 0

            distinguishability_matrix[i, j] = distinguishability_matrix[j, i] = average_overlap

    for i in range(num_rules):
        distinguishability_matrix[i, i] = 1.0  # Maximum distinguishability with yourself

    return distinguishability_matrix


def calculate_ecompleteness_and_get_uncovered_samples(all_activations, epsilon):
    """
        Calculates the completeness score and identifies uncovered samples based on activation thresholds.

        Parameters:
        - all_activations (numpy.ndarray): Matrix of rule activations for each sample, where each row represents a sample
          and each column represents a rule.
        - epsilon (float): Activation threshold for determining whether a rule has been activated for a sample.

        Returns:
        - completeness_score (float): The percentage of samples for which at least one rule was activated above the threshold.
          It represents how well the fuzzy rules cover the samples.
        - uncovered_samples_indices (numpy.ndarray): Indices of uncovered samples, i.e., samples for which no rules were
          activated above the threshold. It helps in identifying samples that may need further attention or refinement
          in the fuzzy rule system.
    """

    # Apply the epsilon activation threshold to determine whether a rule has been activated for each sample
    is_activated = all_activations >= epsilon

    # Calculates the percentage of samples for which at least one rule was activated
    completeness_score = np.any(is_activated, axis=1).mean()

    # Identify uncovered samples (i.e. samples for which no rules were activated above the threshold)
    uncovered_samples_indices = np.where(~np.any(is_activated, axis=1))[0]

    return completeness_score, uncovered_samples_indices


def calculate_similarity_matrix(rules):
    """
        Calculates a similarity matrix for a set of fuzzy rules based on their parameters.

        Parameters:
        - rules (list of dicts): List containing information about each fuzzy rule. Each dictionary should have keys
          "centers" and "sigmas", where "centers" is an array representing the means of Gaussian membership functions,
          and "sigmas" is an array representing the standard deviations of Gaussian membership functions.

        Returns:
        - similarity_matrix (numpy.ndarray): A square matrix representing the similarity between fuzzy rules.
          Each element (i, j) in the matrix indicates the similarity between rule i and rule j based on their
          parameters (mean and standard deviation). The values range from 0 to 1, where 0 indicates no similarity
          (maximum distance), and 1 indicates maximum similarity (identical parameters). The diagonal elements are set
          to 1, representing maximum similarity with oneself.
    """
    num_rules = len(rules)
    similarity_matrix = np.zeros((num_rules, num_rules))

    for i in range(num_rules):
        for j in range(i + 1, num_rules):
            centers_i = np.array(rules[i]["centers"])
            sigmas_i = np.array(rules[i]["sigmas"])
            centers_j = np.array(rules[j]["centers"])
            sigmas_j = np.array(rules[j]["sigmas"])

            # Calculates the Euclidean distance between rule parameters
            distance = np.sqrt(
                np.sum((centers_i - centers_j) ** 2 + (sigmas_i - sigmas_j) ** 2)
            )
            similarity = np.exp(-distance)
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    # The similarity of a rule to itself is maximum
    np.fill_diagonal(similarity_matrix, 1.0)

    return similarity_matrix


### THE FOLLOWING FUNCTIONS HAVE BEEN NOT USED YET, IF YOU WANT TO USE THEM DECOMMENT THEM AND ADD THE DOCUMENTATION

# def calculate_rule_activation(input_vector, rule_params):
#     """
#     Calculates the activation level of a rule for a given input vector.
#     """
#     activation_levels = []
#     for input_val, params in zip(input_vector, rule_params):
#         center, sigma = params["centers"], params["sigmas"]
#         # Calculates the degree of relevance for this input
#         pertinence_degree = np.exp(-((input_val - center) ** 2) / (2 * sigma**2))
#         activation_levels.append(pertinence_degree)
#
#     # The activation of the rule is the minimum of the degrees of relevance
#     rule_activation = min(activation_levels)
#     return rule_activation


##### ARE THESE TWO DIFFERENT?

# def calculate_rule_activation(input_data, rule_index):
#     """
#     Calculate the activation level of a specific fuzzy rule for each sample in X.
#
#     Parameters:
#     - input_data: Input data, a NumPy array of shape (n_samples, n_features).
#     - rule_index: Index of the fuzzy rule for which to calculate activation levels.
#
#     Returns:
#     - A NumPy array of shape (n_samples,) containing the activation level of the rule for each sample.
#     """
#     # Placeholder implementation: Replace this with your actual logic to compute rule activation
#     # This might involve fuzzification of X, then applying the specific rule logic
#     activations = np.random.rand(input_data.shape[0])  # Example: Random activations
#     return activations


# def calculate_rule_activation_index(rule_index, logic_outputs):
#     """
#     Returns the activation level of a specific rule (logical neuron) for each sample in X.
#
#     Parâmetros:
#     - rule_index: Index of the rule (logical neuron) whose activation is to be captured.
#
#     Retorna:
#     - activations: The specific rule activation levels for all samples.
#     """
#     # Assuming that self.logic_output is the Z matrix with the outputs of the logical neurons
#     activations = logic_outputs[:, rule_index]
#     return activations

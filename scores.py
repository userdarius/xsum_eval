import numpy as np
from scipy.special import logsumexp


def context_entails_response(context, responses, model):
    """
    Check if the context entails the response with improved robustness

    Args:
        context: Source text
        responses: List of generated summaries
        model: Entailment model instance
    Returns:
        float: Entailment score between 0 and 2
    """
    if not responses:
        return 0.0

    votes = []
    for response in responses:
        try:
            score = model.check_implication(context, response)
            votes.append(float(score))
        except Exception as e:
            print(f"Warning: Entailment check failed: {e}")
            votes.append(0.0)

    if not votes:
        return 0.0

    # Average the votes and ensure output is between 0 and 2
    mean_score = np.mean(votes)
    return float(np.clip(mean_score, 0.0, 2.0))


def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(text1, text2):

        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(
            text2, text1, example=example
        )  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and (
                [1, 1] != implications
            )

        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods, agg="sum_normalized"):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == "sum_normalized":
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id


def predictive_entropy(log_probs):
    """Compute MC estimate of entropy with numerical stability.

    Args:
        log_probs: List of log probabilities
    Returns:
        float: Entropy estimate
    """
    # Convert to numpy array if not already
    log_probs = np.array(log_probs)

    # Normalize log probabilities
    log_probs_normalized = log_probs - logsumexp(log_probs)

    # Calculate entropy
    entropy = -np.sum(np.exp(log_probs_normalized) * log_probs_normalized)

    return entropy


def predictive_entropy_rao(log_probs):
    """Compute Rao's entropy estimate with numerical stability.

    Args:
        log_probs: List of log probabilities
    Returns:
        float: Rao entropy estimate
    """
    # Convert to numpy array if not already
    log_probs = np.array(log_probs)

    # Normalize log probabilities using log-sum-exp trick
    log_probs_normalized = log_probs - logsumexp(log_probs)
    probs = np.exp(log_probs_normalized)

    # Calculate Rao entropy
    entropy = -np.sum(probs * log_probs)

    return entropy


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from cluster assignments.

    Args:
        semantic_ids: List of semantic cluster IDs
    Returns:
        float: Cluster assignment entropy
    """
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations

    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy

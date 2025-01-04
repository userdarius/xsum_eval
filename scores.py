import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging


def context_entails_response(context, responses, model):
    """
    Check if each summary is entailed by the context (document)
    Returns the average entailment score across all summaries
    0 = contradiction
    1 = neutral
    2 = entailment
    """
    votes = []
    for idx, response in enumerate(responses):
        vote = model.check_implication(response, context)
        print(f"Summary {idx+1}: {response}")
        print(
            f"Entailment score: {vote} ({['contradiction', 'neutral', 'entailment'][vote]})\n"
        )
        votes.append(vote)

    mean_score = np.mean(votes)
    print(f"All votes: {votes}")
    print(f"Mean entailment score: {mean_score}")
    return mean_score


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

    # After assigning all IDs, log the clusters
    clusters = {}
    for idx, sem_id in enumerate(semantic_set_ids):
        if sem_id not in clusters:
            clusters[sem_id] = []
        clusters[sem_id].append(strings_list[idx])

    # Log each cluster
    logging.info("Semantic clusters:")
    for cluster_id, texts in clusters.items():
        logging.info(f"\nCluster {cluster_id}:")
        for text in texts:
            logging.info(f"  - {text}")

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
    """Compute MC estimate of entropy with normalized probabilities.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)`, i.e. the average token likelihood.
    """
    # Normalize to prevent underflow and get probabilities that sum to 1
    log_probs_normalized = log_probs - np.max(log_probs)
    probs = np.exp(log_probs_normalized)
    probs = probs / np.sum(probs)

    # Calculate raw entropy using original log probabilities
    raw_entropy = -np.sum(probs * log_probs)

    # Normalize by the maximum possible entropy for the number of samples
    n_samples = len(log_probs)
    max_entropy = np.log(
        n_samples
    )  # Maximum entropy is ln(n) for n equally likely outcomes

    # Scale the entropy to [0, ln(n)] range
    normalized_entropy = raw_entropy / (np.abs(np.mean(log_probs)))
    normalized_entropy = normalized_entropy * max_entropy

    return normalized_entropy


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = -(probabilities * np.log(probabilities)).sum()
    return entropy

import numpy as np


def safe_log(probabilities):
    """
    Compute the logarithm of probabilities, keeping
    zero probabilities equal to negative infinity.
    """

    log_probabilities = np.full_like(
        probabilities,
        -np.inf,
        dtype=float,
    )

    positive = probabilities > 0

    log_probabilities[positive] = np.log(
        probabilities[positive]
    )

    return log_probabilities


def logsumexp(values):
    """
    Compute log(sum(exp(values))) in a numerically stable way.
    """
    maximum = np.max(values)

    if np.isneginf(maximum):
        return -np.inf

    return maximum + np.log(
        np.exp(values - maximum).sum()
    )


def gaussian_log_probabilities(
    observations,
    means,
    variances,
):
    """
    Compute Gaussian log-probabilities for every
    observation and every hidden state.

    Diagonal covariance matrices are assumed.
    """

    number_of_observations, number_of_features = (
        observations.shape
    )

    number_of_states = means.shape[0]

    if np.any(variances <= 0):
        raise ValueError(
            "Gaussian variances must be positive."
        )

    log_probabilities = np.empty(
        (
            number_of_observations,
            number_of_states,
        )
    )

    for state in range(number_of_states):

        differences = (
            observations - means[state]
        )

        log_determinant = np.sum(
            np.log(variances[state])
        )

        quadratic_term = np.sum(
            differences**2 / variances[state],
            axis=1,
        )

        log_probabilities[:, state] = (
            -0.5
            * (
                number_of_features
                * np.log(2 * np.pi)
                + log_determinant
                + quadratic_term
            )
        )

    return log_probabilities


def forward_log(
    observations,
    initial_probabilities,
    transition_matrix,
    means,
    variances,
):
    """
    Run the forward algorithm in log-space.

    Returns
    -------
    log_alpha : numpy.ndarray
        Forward messages for every time and state.

    log_likelihood : float
        Log-probability of the complete observation sequence.
    """

    emission_log_probabilities = (
        gaussian_log_probabilities(
            observations,
            means,
            variances,
        )
    )

    number_of_observations = observations.shape[0]
    number_of_states = len(initial_probabilities)

    log_initial_probabilities = safe_log(
        initial_probabilities
    )

    log_transition_matrix = safe_log(
        transition_matrix
    )

    log_alpha = np.empty(
        (
            number_of_observations,
            number_of_states,
        )
    )

    log_alpha[0] = (
        log_initial_probabilities
        + emission_log_probabilities[0]
    )

    for time in range(
        1,
        number_of_observations,
    ):

        for current_state in range(
            number_of_states
        ):

            previous_terms = (
                log_alpha[time - 1]
                + log_transition_matrix[
                    :,
                    current_state,
                ]
            )

            log_alpha[
                time,
                current_state,
            ] = (
                emission_log_probabilities[
                    time,
                    current_state,
                ]
                + logsumexp(previous_terms)
            )

    log_likelihood = logsumexp(
        log_alpha[-1]
    )

    return log_alpha, log_likelihood

def backward_log(
    observations,
    transition_matrix,
    means,
    variances,
):
    """
    Run the backward algorithm in log-space.
    Returns
    -------
    log_beta : numpy.ndarray
        Backward messages for every time and state.
    """

    emission_log_probabilities = (
        gaussian_log_probabilities(
            observations,
            means,
            variances,
        )
    )

    number_of_observations = observations.shape[0]
    number_of_states = transition_matrix.shape[0]

    log_transition_matrix = safe_log(
        transition_matrix
    )

    log_beta = np.empty(
        (
            number_of_observations,
            number_of_states,
        )
    )

    log_beta[-1] = 0.0

    for time in range(
        number_of_observations - 2,
        -1,
        -1,
    ):

        for current_state in range(
            number_of_states
        ):

            next_terms = (
                log_transition_matrix[
                    current_state,
                    :
                ]
                + emission_log_probabilities[
                    time + 1,
                    :
                ]
                + log_beta[
                    time + 1,
                    :
                ]
            )

            log_beta[
                time,
                current_state,
            ] = logsumexp(next_terms)

    return log_beta


def smoothed_state_probabilities(
    observations,
    initial_probabilities,
    transition_matrix,
    means,
    variances,
):
    """
    Compute posterior state probabilities
    p(z_t = k | x_1:T) using forward-backward.
    """

    log_alpha, log_likelihood = forward_log(
        observations,
        initial_probabilities,
        transition_matrix,
        means,
        variances,
    )

    log_beta = backward_log(
        observations,
        transition_matrix,
        means,
        variances,
    )

    log_posterior = (
        log_alpha
        + log_beta
        - log_likelihood
    )

    posterior = np.exp(log_posterior)

    posterior = (
        posterior
        / posterior.sum(
            axis=1,
            keepdims=True,
        )
    )
    return posterior


def viterbi_log(
    observations,
    initial_probabilities,
    transition_matrix,
    means,
    variances,
):
    """
    Find the most likely hidden-state sequence
    using the Viterbi algorithm in log-space.
    """

    emission_log_probabilities = (
        gaussian_log_probabilities(
            observations,
            means,
            variances,
        )
    )

    number_of_observations = observations.shape[0]
    number_of_states = len(initial_probabilities)

    log_initial_probabilities = safe_log(
        initial_probabilities
    )

    log_transition_matrix = safe_log(
        transition_matrix
    )

    viterbi_scores = np.empty(
        (
            number_of_observations,
            number_of_states,
        )
    )

    backpointers = np.zeros(
        (
            number_of_observations,
            number_of_states,
        ),
        dtype=int,
    )

    viterbi_scores[0] = (
        log_initial_probabilities
        + emission_log_probabilities[0]
    )

    for time in range(
        1,
        number_of_observations,
    ):

        for current_state in range(
            number_of_states
        ):

            previous_scores = (
                viterbi_scores[time - 1]
                + log_transition_matrix[
                    :,
                    current_state,
                ]
            )

            best_previous_state = np.argmax(
                previous_scores
            )

            viterbi_scores[
                time,
                current_state,
            ] = (
                previous_scores[
                    best_previous_state
                ]
                + emission_log_probabilities[
                    time,
                    current_state,
                ]
            )

            backpointers[
                time,
                current_state,
            ] = best_previous_state

    best_last_state = np.argmax(
        viterbi_scores[-1]
    )

    best_log_probability = (
        viterbi_scores[
            -1,
            best_last_state,
        ]
    )

    best_path = np.empty(
        number_of_observations,
        dtype=int,
    )

    best_path[-1] = best_last_state

    for time in range(
        number_of_observations - 1,
        0,
        -1,
    ):

        best_path[time - 1] = (
            backpointers[
                time,
                best_path[time],
            ]
        )
    return best_path, best_log_probability

def posterior_entropy(posterior_probabilities):
    """
    Compute normalized entropy of posterior
    hidden-state probabilities.

    The returned values are between 0 and 1.
    """

    number_of_states = posterior_probabilities.shape[1]

    safe_probabilities = np.clip(
        posterior_probabilities,
        1e-12,
        1.0,
    )

    entropy = -np.sum(
        safe_probabilities
        * np.log(safe_probabilities),
        axis=1,
    )

    normalized_entropy = (
        entropy / np.log(number_of_states)
    )

    return normalized_entropy
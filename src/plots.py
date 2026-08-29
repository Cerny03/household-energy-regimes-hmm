import numpy as np
import matplotlib.pyplot as plt

from src.config import (
    FIGURES_DIR,
    DATETIME_COLUMN,
    MODEL_FEATURE_COLUMNS,
)

from src.inference import (
    smoothed_state_probabilities,
    posterior_entropy,
)


def plot_model_selection(
    gmm_results,
    hmm_results,
):
    """
    Plot validation log-likelihood for GMM and HMM
    as a function of the number of latent states.
    """

    gmm_k = [
        result["k"]
        for result in gmm_results
    ]

    gmm_scores = [
        result["validation_log_likelihood"]
        for result in gmm_results
    ]

    hmm_k = [
        result["k"]
        for result in hmm_results
    ]

    hmm_scores = [
        result["validation_log_likelihood"]
        for result in hmm_results
    ]

    plt.figure(figsize=(7, 5))

    plt.plot(
        gmm_k,
        gmm_scores,
        marker="o",
        label="GMM",
    )

    plt.plot(
        hmm_k,
        hmm_scores,
        marker="o",
        label="HMM",
    )

    plt.xlabel("Number of components / states")
    plt.ylabel("Validation average log-likelihood")
    plt.xticks(gmm_k)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "model_selection.png",
        dpi=200,
    )

    plt.close()

def plot_log_likelihood_comparison(
    gmm_scores,
    hmm_scores,
):
    """
    Compare GMM and HMM average log-likelihood
    on train, validation and test sets.
    """

    datasets = [
        "Train",
        "Validation",
        "Test",
    ]

    positions = np.arange(
        len(datasets)
    )

    width = 0.35

    plt.figure(figsize=(7, 5))

    plt.bar(
        positions - width / 2,
        gmm_scores,
        width,
        label="GMM",
    )

    plt.bar(
        positions + width / 2,
        hmm_scores,
        width,
        label="HMM",
    )

    plt.xticks(
        positions,
        datasets,
    )

    plt.ylabel("Average log-likelihood")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "gmm_vs_hmm_log_likelihood.png",
        dpi=200,
    )

    plt.close()

def plot_transition_matrix(model):
    """
    Plot the HMM transition matrix.
    """

    transition_matrix = model.transmat_

    plt.figure(figsize=(6, 5))

    image = plt.imshow(
        transition_matrix,
        vmin=0,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Transition probability",
    )

    number_of_states = model.n_components

    plt.xticks(
        range(number_of_states)
    )

    plt.yticks(
        range(number_of_states)
    )

    plt.xlabel("Next state")
    plt.ylabel("Current state")

    for i in range(number_of_states):

        for j in range(number_of_states):

            plt.text(
                j,
                i,
                f"{transition_matrix[i, j]:.2f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "hmm_transition_matrix.png",
        dpi=200,
    )

    plt.close()

def plot_hmm_state_profiles(
    state_means,
):
    """
    Plot mean energy consumption for each HMM state.
    """

    feature_names = [
        "Global",
        "Kitchen",
        "Laundry",
        "Water heater / AC",
    ]

    number_of_states = state_means.shape[0]
    number_of_features = state_means.shape[1]

    positions = np.arange(
        number_of_states
    )

    width = 0.18

    plt.figure(figsize=(9, 5))

    for feature_index in range(
        number_of_features
    ):

        offset = (
            feature_index
            - (number_of_features - 1) / 2
        ) * width

        plt.bar(
            positions + offset,
            state_means[:, feature_index],
            width,
            label=feature_names[
                feature_index
            ],
        )

    plt.xticks(
        positions,
        [
            f"State {state}"
            for state in range(
                number_of_states
            )
        ],
    )

    plt.ylabel("Mean hourly energy (kWh)")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "hmm_state_profiles.png",
        dpi=200,
    )

    plt.close()

def plot_temporal_comparison(
    gmm_temporal_results,
    hmm_temporal_results,
):
    """
    Compare temporal persistence of GMM and HMM
    state assignments on validation data.
    """

    model_names = [
        "GMM",
        "HMM",
    ]

    switch_rates = [
        gmm_temporal_results[
            "switch_rate"
        ] * 100,
        hmm_temporal_results[
            "switch_rate"
        ] * 100,
    ]

    plt.figure(figsize=(6, 5))

    plt.bar(
        model_names,
        switch_rates,
    )

    plt.ylabel("State switches (%)")
    plt.title(
        "Temporal fragmentation on validation data"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "temporal_switch_rate.png",
        dpi=200,
    )

    plt.close()


def plot_hmm_timeline(
    model,
    standardized_data,
    original_data,
    max_hours=168,
):
    """
    Plot one continuous validation sequence with
    global energy, Viterbi states and posterior uncertainty.
    """

    sequence_ids = (
        standardized_data["sequence_id"]
        .dropna()
        .unique()
    )

    first_sequence_id = sequence_ids[0]

    standardized_sequence = (
        standardized_data[
            standardized_data["sequence_id"]
            == first_sequence_id
        ]
        .copy()
        .reset_index(drop=True)
    )

    original_sequence = (
        original_data[
            original_data["sequence_id"]
            == first_sequence_id
        ]
        .copy()
        .reset_index(drop=True)
    )

    number_of_hours = min(
        max_hours,
        len(standardized_sequence),
    )

    standardized_sequence = (
        standardized_sequence.iloc[
            :number_of_hours
        ]
    )

    original_sequence = (
        original_sequence.iloc[
            :number_of_hours
        ]
    )

    observations = (
        standardized_sequence[
            MODEL_FEATURE_COLUMNS
        ]
        .to_numpy()
    )

    _, states = model.decode(
        observations,
        algorithm="viterbi",
    )

    covariances = model.covars_

    if covariances.ndim == 2:
        variances = covariances
    else:
        variances = np.diagonal(
            covariances,
            axis1=1,
            axis2=2,
        )

    posterior = (
        smoothed_state_probabilities(
            observations,
            model.startprob_,
            model.transmat_,
            model.means_,
            variances,
        )
    )

    entropy = posterior_entropy(
        posterior
    )

    timestamps = original_sequence[
        DATETIME_COLUMN
    ]

    global_energy = original_sequence[
        "global_energy_kwh"
    ]

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 8),
        sharex=True,
    )

    axes[0].plot(
        timestamps,
        global_energy,
    )

    axes[0].set_ylabel(
        "Global energy\n(kWh)"
    )

    axes[1].step(
        timestamps,
        states,
        where="post",
    )

    axes[1].set_ylabel(
        "Viterbi state"
    )

    axes[1].set_yticks(
        range(model.n_components)
    )

    axes[2].plot(
        timestamps,
        entropy,
    )

    axes[2].set_ylabel(
        "Normalized\nentropy"
    )

    axes[2].set_xlabel(
        "Time"
    )

    figure.autofmt_xdate()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "hmm_timeline.png",
        dpi=200,
    )

    plt.close(figure)
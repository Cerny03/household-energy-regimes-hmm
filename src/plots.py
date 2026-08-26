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
        #K = numero di componenti
        result["k"]
        for result in gmm_results
    ]

    gmm_scores = [
        #estrae le validation log-likelihood
        result["validation_log_likelihood"]
        for result in gmm_results
    ]

    hmm_k = [
        #K = numero di stati nascosti
        result["k"]
        for result in hmm_results
    ]

    hmm_scores = [
        result["validation_log_likelihood"]
        for result in hmm_results
    ]

    plt.figure(figsize=(7, 5))
    #crea una nuova figura.

    plt.plot(
        gmm_k,
        gmm_scores,
        marker="o",
        label="GMM",
    )

    plt.plot(
        #nello stesso grafico avrai due linee
        hmm_k,
        hmm_scores,
        marker="o",
        label="HMM",
    )

    plt.xlabel("Number of components / states")
    plt.ylabel("Validation average log-likelihood")
    plt.title("Model selection")
    plt.xticks(gmm_k)
    plt.legend()
    plt.grid(alpha=0.3)
    #alpha=0.3 significa trasparenza al 30%.

    plt.tight_layout()
    #sistema automaticamente gli spazi per evitare che titoli ed etichette escano dalla figura.

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
    #larghezza di ogni barra.

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
    plt.title("GMM vs HMM")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "gmm_vs_hmm_log_likelihood.png",
        dpi=200,
    )

    plt.close()

def plot_transition_matrix(model):
    #visualizza la matrice di transizione dell'HMM.
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
        #barra laterale che mostra la corrispondenza tra intensità e probabilità.
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
    plt.title("HMM transition matrix")

    for i in range(number_of_states):

        for j in range(number_of_states):

            plt.text(
                #scrive dentro la casella il valore numerico.
                j,
                i,
                f"{transition_matrix[i, j]:.2f}",
                ha="center",
                #horizontal alignment
                va="center",
                #vertical alignment
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
            #sposta le barre lateralmente
            feature_index
            - (number_of_features - 1) / 2
        ) * width

        plt.bar(
            positions + offset,
            state_means[:, feature_index],
            #per questa feature, prendimi la media in tutti gli stati.
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
    plt.title("HMM state profiles")
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
    #figura con tre pannelli allineati nel tempo: consumo energetico reale, stato HMM scelto da Viterbi, incertezza dell'HMM
    model,
    #HMM già addestrato.
    standardized_data,
    original_data,
    max_hours=168,
    #quante ore al massimo vuoi rappresentare. Una settimana
):
    """
    Plot one continuous validation sequence with
    global energy, Viterbi states and posterior uncertainty.
    """

    sequence_ids = (
        standardized_data["sequence_id"]
        #sequence_id identifica le sequenze di ore consecutive senza buchi.
        .dropna()
        .unique()
    )

    first_sequence_id = sequence_ids[0]
    #per il grafico mostro la prima sequenza temporale continua.

    standardized_sequence = (
        standardized_data[
            standardized_data["sequence_id"]
            == first_sequence_id
        ]
        #mantiene solo le righe appartenenti alla sequenza 1.
        .copy()
        .reset_index(drop=True)
    )

    original_sequence = (
        #stessa sequenza, ma dai dati non standardizzati.
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
        #Taglia entrambe le sequenze
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
        #trasformi da DataFrame pandas a matrice NumPy.
    )

    _, states = model.decode(
        #qual è la sequenza di stati nascosti complessivamente più probabile per queste osservazioni
        observations,
        algorithm="viterbi",
    )

    covariances = model.covars_
    #Recuperi le covarianze gaussiane dell'HMM.

    if covariances.ndim == 2:
        variances = covariances
        #Se sono già rappresentate come: K × J cioè una varianza per ogni stato e feature, non devi fare nulla.
    else:
        variances = np.diagonal(
            #se hai una matrice di covarianza completa per ogni stato, prendi soltanto la diagonale
            covariances,
            axis1=1,
            axis2=2,
        )

    posterior = (
        #utilizzi Forward-Backward. dato l'intero segmento osservato, quali sono le probabilità dei diversi stati in quell'ora
        smoothed_state_probabilities(
            observations,
            model.startprob_,
            model.transmat_,
            model.means_,
            variances,
        )
    )

    entropy = posterior_entropy(
        #entropy≈0 HMM molto sicuro. entropy≈1 HMM molto incerto. Ottengo un valore per ogni ora.
        posterior
    )

    timestamps = original_sequence[
        #Recuperi tempo e consumo originale
        DATETIME_COLUMN
    ]

    global_energy = original_sequence[
        #consumo globale originale in kWh.
        "global_energy_kwh"
    ]

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 8),
        sharex=True,
    )

    axes[0].plot(
        #Primo pannello: consumo energetico
        timestamps,
        global_energy,
    )

    axes[0].set_ylabel(
        "Global energy\n(kWh)"
    )

    axes[0].set_title(
        "Household energy consumption and inferred HMM regimes"
    )

    axes[1].step(
        #Secondo pannello: stati Viterbi
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
        #Terzo pannello: entropia
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
    #ruota le date e le dispone automaticamente in modo più leggibile.

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "hmm_timeline.png",
        dpi=200,
    )

    plt.close(figure)
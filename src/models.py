from sklearn.mixture import GaussianMixture
import numpy as np

from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM
from src.inference import (
    forward_log,
    smoothed_state_probabilities,
    viterbi_log,
)

from src.config import (
    FIGURES_DIR,
    MODEL_FEATURE_COLUMNS,
    RESULTS_DIR,
    RANDOM_SEED,
    N_STATES_CANDIDATES,
    N_INITIALIZATIONS,
    MAX_ITERATIONS,
    CONVERGENCE_TOLERANCE,
    COVARIANCE_TYPE,
    MIN_COVARIANCE,
)

from src.inference import (
    forward_log,
    smoothed_state_probabilities,
    posterior_entropy,
    viterbi_log,
)

from src.plots import (
    plot_model_selection,
    plot_log_likelihood_comparison,
    plot_transition_matrix,
    plot_hmm_state_profiles,
    plot_temporal_comparison,
    plot_hmm_timeline,
)

def fit_hmm_once(
    #crea e addestra un singolo Gaussian HMM.
    train_observations,
    #matrice dei dati: numero ore valide × numero feature
    train_lengths,
    #dice al modello come queste 20.000 ore sono divise in sequenze temporali continue.
    number_of_states,
    #quanti stati nascosti vuoi nell'HMM.
    random_seed,
):
    """
    Fit one Gaussian HMM initialization.
    """

    model = GaussianHMM(
        n_components=number_of_states,
        covariance_type=COVARIANCE_TYPE,
        min_covar=MIN_COVARIANCE,
        #Non vuoi che durante l'addestramento una varianza diventi praticamente 0
        n_iter=MAX_ITERATIONS,
        tol=CONVERGENCE_TOLERANCE,
        random_state=random_seed,
        implementation="log",
        #eseguire i calcoli dell'HMM usando il log-domain.
    )

    model.fit(
        #cerca di imparare: prob iniziali, matr di transizione, medie gaussiane e covarianze/varianze gaussiane per ogni stato nascosto
        train_observations,
        train_lengths,
    )
    #.fit() cerca i parametri che rendono i dati di training il più probabili possibile.

    return model

def fit_best_hmm_for_k(
    train_observations,
    train_lengths,
    number_of_states,
):
    """
    Fit several HMM initializations for a fixed number
    of states and keep the best training solution.
    """

    best_model = None
    best_train_score = -float("inf")
    #qualunque score reale sarà maggiore di −∞.

    for initialization in range(N_INITIALIZATIONS):

        current_seed = (
            RANDOM_SEED + initialization
            #le inizializzazioni sono diverse ma riproducibili.
        )

        model = fit_hmm_once(
            train_observations,
            train_lengths,
            number_of_states,
            current_seed,
        )

        train_score = model.score(
            #calcola la log-likelihood dei dati sotto il modello. P(X|THETA)
            train_observations,
            train_lengths,
        )
        if train_score > best_train_score:
            #EM può finire in massimi locali.
            best_train_score = train_score
            best_model = model

    return best_model


def average_hmm_log_likelihood(
    #Questa funzione misura quanto bene un HMM spiega un dataset.
    model,
    observations,
    lengths,
):
    """
    Compute HMM log-likelihood per observation.
    """

    total_log_likelihood = model.score(
        #calcola P(X|THETA) per tutto il dataset, tenendo conto della struttura delle sequenze. 
        observations,
        lengths,
    )

    average_log_likelihood = (
        total_log_likelihood
        / len(observations)
    )

    return average_log_likelihood

def train_hmm_candidates(
    #proviamo HMM con diversi numeri di stati.
    train_observations,
    train_lengths,
    validation_observations,
    validation_lengths,
):
    """
    Train Gaussian HMMs with different numbers of states.
    """

    models = {}
    #dizionario che conterrà i modelli veri e propri
    results = []

    for number_of_states in N_STATES_CANDIDATES:

        model = fit_best_hmm_for_k(
            train_observations,
            train_lengths,
            number_of_states,
        )

        train_log_likelihood = (
            average_hmm_log_likelihood(
                model,
                train_observations,
                train_lengths,
            )
        )

        validation_log_likelihood = (
            average_hmm_log_likelihood(
                model,
                validation_observations,
                validation_lengths,
            )
        )

        result = {
            "k": number_of_states,
            "train_log_likelihood": train_log_likelihood,
            "validation_log_likelihood": validation_log_likelihood,
            "converged": model.monitor_.converged,
            "iterations": model.monitor_.iter,
        }

        models[number_of_states] = model
        #nel dizionario la chiave è il numeri di stati nascosti, il valore è il modello HMM addestrato.
        results.append(result)

    return models, results

def get_hmm_diagonal_variances(model):
    """
    Extract diagonal variances from a Gaussian HMM.
    """

    covariances = model.covars_

    if covariances.ndim == 2:
        return covariances
    #se sono già nella forma K×J, restituiscile direttamente.

    return np.diagonal(
        #covariances non è più una semplice matrice 2D. È un array 3D: numero stati × numero feature × numero feature, lw feature sono i 4 tipi di consumo energetico.
        covariances,
        axis1=1,
        axis2=2,
        #fai la diagonale tra la dimensione delle righe e quella delle colonne, mantenendo separata la dimensione degli stati.
    )

def check_forward_algorithm(
    model,
    observations,
    lengths,
):
    """
    Compare our forward algorithm with hmmlearn
    on one continuous sequence.
    """

    first_sequence_length = lengths[0]

    first_sequence = observations[
        :first_sequence_length
    ]

    variances = get_hmm_diagonal_variances(
        model
    )

    _, custom_log_likelihood = forward_log(
        #forward_log(...) restituisce due cose: log_alpha, log_likelihood (log P(x1,...,xT))
        first_sequence,
        model.startprob_,
        model.transmat_,
        model.means_,
        variances,
    )

    hmmlearn_log_likelihood = model.score(
        first_sequence
    )

    difference = abs(
        #Se il tuo algoritmo è corretto, ti aspetti una differenza molto vicina a zero.
        custom_log_likelihood
        - hmmlearn_log_likelihood
    )

    print("\nForward algorithm check:")

    print("Sequence length:")
    print(first_sequence_length)

    print("\nOur forward log-likelihood:")
    print(f"{custom_log_likelihood:.10f}")

    print("\nhmmlearn log-likelihood:")
    print(f"{hmmlearn_log_likelihood:.10f}")

    print("\nAbsolute difference:")
    print(f"{difference:.10e}")


def check_viterbi_algorithm(
    model,
    observations,
    lengths,
):
    """
    Compare our Viterbi path with hmmlearn.
    """

    first_sequence_length = lengths[0]

    first_sequence = observations[
        :first_sequence_length
    ]

    variances = get_hmm_diagonal_variances(
        model
    )

    custom_path, custom_log_probability = (
        #restituisce sequenza di stati nascosti più probabile. e la log-probabilità di quel percorso Viterbi.
        viterbi_log(
            first_sequence,
            model.startprob_,
            model.transmat_,
            model.means_,
            variances,
        )
    )

    (
        hmmlearn_log_probability,
        hmmlearn_path,
    ) = model.decode(
        #model.decode(...) cerca il percorso di stati nascosti più probabile.
        first_sequence,
        algorithm="viterbi",
    )

    number_of_different_states = np.sum(
        custom_path != hmmlearn_path
    )

    log_probability_difference = abs(
        custom_log_probability
        - hmmlearn_log_probability
    )

    print("\nViterbi check:")

    print("Sequence length:")
    print(first_sequence_length)

    print("\nDifferent state assignments:")
    print(number_of_different_states)

    print("\nOur Viterbi log-probability:")
    print(f"{custom_log_probability:.10f}")

    print("\nhmmlearn Viterbi log-probability:")
    print(f"{hmmlearn_log_probability:.10f}")

    print("\nAbsolute log-probability difference:")
    print(f"{log_probability_difference:.10e}")

    print("\nFirst 20 Viterbi states:")
    print(custom_path[:20])

def select_best_hmm(models, results):
    """
    Select the HMM with the highest validation
    average log-likelihood.
    """

    best_model = None
    best_result = None
    best_validation_score = -float("inf")

    for result in results:

        current_score = result[
            "validation_log_likelihood"
        ]

        if current_score > best_validation_score:

            best_validation_score = current_score

            best_result = result

            best_model = models[
                result["k"]
                #recuperi il corrispondente modello
            ]

    return best_model, best_result

def get_hmm_parameters_original_scale(
    model,
    feature_means,
    feature_stds,
):
    """
    Convert HMM Gaussian emission parameters
    back to the original kWh scale.
    """

    training_means = feature_means.to_numpy()
    training_stds = feature_stds.to_numpy()

    state_means = (
        model.means_ * training_stds
        + training_means
    )

    variances = get_hmm_diagonal_variances(
        model
    )

    state_stds = (
        np.sqrt(variances)
        * training_stds
    )

    return state_means, state_stds

def analyze_hmm_uncertainty(
    model,
    observations,
    lengths,
):
    """
    Analyze posterior uncertainty over hidden states.
    """

    variances = get_hmm_diagonal_variances(
        model
    )

    all_entropies = []
    all_max_probabilities = []
    #per ogni ora, la probabilità dello stato più probabile.

    start = 0

    for sequence_length in lengths:

        end = start + sequence_length

        sequence = observations[
            start:end
        ]

        posterior = (
            #Forward + Backward.
            smoothed_state_probabilities(
                sequence,
                model.startprob_,
                model.transmat_,
                model.means_,
                variances,
            )
        )

        entropy = posterior_entropy(
            posterior
        )

        max_probabilities = np.max(
            posterior,
            axis=1,
        )

        all_entropies.extend(
            entropy
        )

        all_max_probabilities.extend(
            max_probabilities
        )

        start = end

    all_entropies = np.array(
        all_entropies
    )

    all_max_probabilities = np.array(
        all_max_probabilities
    )

    return {
        "mean_entropy": np.mean(all_entropies),
        #incertezza media del modello
        "median_entropy": np.median(all_entropies),
        "mean_max_probability": np.mean(
            all_max_probabilities
        ),
        "low_confidence_fraction": np.mean(
            all_max_probabilities < 0.80
        ),
    }

def print_hmm_uncertainty(results):
    """
    Print posterior uncertainty statistics.
    """

    print("\nHMM posterior uncertainty:")

    print(
        "Mean normalized entropy: "
        f"{results['mean_entropy']:.4f}"
    )

    print(
        "Median normalized entropy: "
        f"{results['median_entropy']:.4f}"
    )

    print(
        "Mean maximum posterior probability: "
        f"{results['mean_max_probability']:.4f}"
    )

    print(
        "Hours with posterior confidence < 0.80: "
        f"{results['low_confidence_fraction'] * 100:.2f}%"
    )

def print_hmm_states(
    model,
    state_means,
    state_stds,
):
    """
    Print Gaussian emission statistics for each HMM state.
    """
    feature_names = [
        "Global",
        "Kitchen",
        "Laundry",
        "Water heater / AC",
    ]

    print("\nHMM states:")

    for state in range(model.n_components):

        print(f"\nState {state}:")

        for feature_index, feature_name in enumerate(
            feature_names
        ):

            print(
                f"{feature_name}: "
                f"mean={state_means[state, feature_index]:.4f} kWh, "
                #prendi la media della feature feature_index nello stato state.
                f"std={state_stds[state, feature_index]:.4f} kWh"
            )

def compute_expected_state_durations(model):
    """
    Compute expected state durations implied by
    the HMM self-transition probabilities.
    """

    self_transition_probabilities = np.diag(
        model.transmat_
    )

    expected_durations = (
        1.0
        / (1.0 - self_transition_probabilities)
    )
    return expected_durations

def print_expected_state_durations(
    model,
    expected_durations,
):
    """
    Print self-transition probabilities and
    expected HMM state durations.
    """

    print("\nExpected HMM state durations:")

    for state in range(model.n_components):

        self_transition = model.transmat_[
            state,
            state,
        ]

        print(
            f"State {state}: "
            f"P(stay)={self_transition:.4f} | "
            f"expected duration={expected_durations[state]:.2f} hours"
        )

def analyze_hmm_temporal_behavior(
    model,
    observations,
    lengths,
):
    """
    Analyze temporal persistence of the HMM
    Viterbi state sequence.
    """

    _, states = model.decode(
        #Troviamo gli stati con Viterbi. Questo array dice: per ogni osservazione/ora, qual è lo stato del percorso Viterbi?
        observations,
        lengths,
        algorithm="viterbi",
    )

    total_transitions = 0
    total_switches = 0
    run_lengths = []

    start = 0

    for sequence_length in lengths:

        end = start + sequence_length

        sequence_states = states[
            start:end
        ]

        if len(sequence_states) >= 2:

            switches = (
                sequence_states[1:]
                != sequence_states[:-1]
            )

            total_transitions += (
                len(sequence_states) - 1
            )

            total_switches += switches.sum()

        current_run_length = 1
        #Un run è un blocco consecutivo dello stesso stato.

        for time in range(
            1,
            len(sequence_states),
        ):

            if (
                sequence_states[time]
                == sequence_states[time - 1]
            ):

                current_run_length += 1

            else:

                run_lengths.append(
                    current_run_length
                )

                current_run_length = 1

        run_lengths.append(
            current_run_length
        )

        start = end

    switch_rate = (
        total_switches
        / total_transitions
    )

    mean_run_length = np.mean(
        run_lengths
    )

    median_run_length = np.median(
        run_lengths
    )

    return {
        "total_transitions": total_transitions,
        "total_switches": total_switches,
        "switch_rate": switch_rate,
        "mean_run_length": mean_run_length,
        "median_run_length": median_run_length,
    }

def print_hmm_temporal_behavior(results):
    """
    Print temporal statistics of HMM Viterbi states.
    """
    print("\nHMM temporal behavior:")

    print("Number of consecutive-hour transitions:")
    print(results["total_transitions"])

    print("\nNumber of state switches:")
    print(results["total_switches"])

    print("\nSwitch rate:")
    print(f"{results['switch_rate'] * 100:.2f}%")

    print("\nMean state run length:")
    print(f"{results['mean_run_length']:.2f} hours")

    print("\nMedian state run length:")
    print(f"{results['median_run_length']:.2f} hours")

def print_hmm_transition_matrix(model):
    """
    Print the HMM transition matrix.
    """

    print("\nHMM transition matrix:")

    print(
        np.round(
            model.transmat_,
            4,
        )
    )

def print_hmm_results(results):
    """
    Print Gaussian HMM model-selection results.
    """

    print("\nHMM model selection:")

    for result in results:

        print(
            f"K={result['k']} | "
            f"train LL={result['train_log_likelihood']:.4f} | "
            f"validation LL={result['validation_log_likelihood']:.4f} | "
            f"converged={result['converged']} | "
            f"iterations={result['iterations']}"
        )

def evaluate_hmm(
    model,
    train_observations,
    train_lengths,
    validation_observations,
    validation_lengths,
    test_observations,
    test_lengths,
):
    """
    Evaluate the selected HMM on train,
    validation and test data.
    """

    train_score = average_hmm_log_likelihood(
        model,
        train_observations,
        train_lengths,
    )

    validation_score = average_hmm_log_likelihood(
        model,
        validation_observations,
        validation_lengths,
    )

    test_score = average_hmm_log_likelihood(
        model,
        test_observations,
        test_lengths,
    )

    return (
        train_score,
        validation_score,
        test_score,
    )

def print_hmm_evaluation(
    train_score,
    validation_score,
    test_score,
):
    """
    Print final HMM average log-likelihood scores.
    """

    print("\nSelected HMM evaluation:")

    print(
        "Training average log-likelihood: "
        f"{train_score:.4f}"
    )

    print(
        "Validation average log-likelihood: "
        f"{validation_score:.4f}"
    )

    print(
        "Test average log-likelihood: "
        f"{test_score:.4f}"
    )

def check_forward_backward_algorithm(
    model,
    observations,
    lengths,
):
    """
    Compare our smoothed state probabilities
    with hmmlearn.
    """

    first_sequence_length = lengths[0]

    first_sequence = observations[
        :first_sequence_length
    ]

    variances = get_hmm_diagonal_variances(
        #prendi dal modello HMM le varianze gaussiane associate a ogni stato e feature.
        model
    )

    custom_posterior = (
        #matrice, Ogni elemento è: probabilità che al tempo t lo stato nascosto sia k, avendo visto tutta la sequenza.
        smoothed_state_probabilities(
            first_sequence,
            model.startprob_,
            model.transmat_,
            model.means_,
            variances,
        )
    )

    hmmlearn_posterior = model.predict_proba(
        #stessa cosa con hmmlearn
        first_sequence
    )

    difference = np.max(
        #confronti le due matrici
        np.abs(
            custom_posterior
            - hmmlearn_posterior
        )
    )

    print("\nForward-backward check:")

    print("Posterior matrix shape:")
    print(custom_posterior.shape)

    print("\nMaximum absolute difference:")
    print(f"{difference:.10e}")

    print("\nFirst posterior distribution:")
    print(custom_posterior[0])

    print("\nSum of first posterior distribution:")
    print(custom_posterior[0].sum())


def fit_gmm(train_observations, number_of_components):
    """
    Fit a Gaussian Mixture Model to the training observations.
    """

    model = GaussianMixture(
        #teoricamente:EM garantisce miglioramento locale, non necessariamente ottimo globale
        n_components=number_of_components,
        #È semplicemente: K.
        covariance_type=COVARIANCE_TYPE,
        n_init=N_INITIALIZATIONS,
        #scikit-learn prova cinque inizializzazioni e conserva quella con il risultato migliore.
        max_iter=MAX_ITERATIONS,
        #una singola esecuzione di EM può effettuare al massimo 100 iterazioni.
        tol=CONVERGENCE_TOLERANCE,
        #Quando il miglioramento della lower bound usata dall'implementazione diventa inferiore alla tolleranza, EM viene fermato.
        random_state=RANDOM_SEED,
    )

    model.fit(train_observations)
    #il GMM non usa: train_lengths perché ignora completamente la struttura delle sequenze.
    return model


def train_gmm_candidates(
    train_observations,
    validation_observations,
):
    """
    Train one GMM for each candidate number of components.
    """

    models = {}
    results = []

    for number_of_components in N_STATES_CANDIDATES:

        model = fit_gmm(
            train_observations,
            number_of_components,
        )

        train_log_likelihood = model.score(
            #score(X) restituisce la log-likelihood media per osservazione sotto il GMM.
            train_observations
        )

        validation_log_likelihood = model.score(
            validation_observations
        )

        bic = model.bic(train_observations)
        #Il BIC introduce una penalizzazione per la complessità del modello.

        result = {
            "k": number_of_components,
            "train_log_likelihood": train_log_likelihood,
            "validation_log_likelihood": validation_log_likelihood,
            "bic": bic,
            "converged": model.converged_,
            "iterations": model.n_iter_,
        }

        models[number_of_components] = model
        results.append(result)

    return models, results


def select_best_gmm(models, results):
    """
    Select the GMM with the highest validation log-likelihood.
    """

    best_model = None
    best_result = None
    best_validation_score = -float("inf")

    for result in results:

        current_score = result[
            "validation_log_likelihood"
        ]

        if current_score > best_validation_score:

            best_validation_score = current_score
            best_result = result

            best_model = models[
                result["k"]
            ]
    return best_model, best_result


def print_gmm_results(results):
    """
    Print the results of GMM model selection.
    """

    print("\nGMM model selection:")

    for result in results:

        print(
            f"K={result['k']} | "
            f"train LL={result['train_log_likelihood']:.4f} | "
            f"validation LL={result['validation_log_likelihood']:.4f} | "
            f"BIC={result['bic']:.2f} | "
            f"converged={result['converged']} | "
            f"iterations={result['iterations']}"
        )

def get_gmm_parameters_original_scale(
    model,
    feature_means,
    feature_stds,
):
    """
    Convert GMM means and standard deviations back to kWh.
    """

    training_means = feature_means.to_numpy()
    training_stds = feature_stds.to_numpy()

    component_means = (
        #qua stiamo riportando le medie dei componenti del GMM alla scala originale dei dati in kWh
        model.means_ * training_stds
        + training_means
    )

    component_stds = (
        np.sqrt(model.covariances_)
        * training_stds
    )

    return component_means, component_stds


def print_gmm_components(
    model,
    component_means,
    component_stds,
):
    """
    Print mixture weights and component statistics
    in the original kWh scale.
    """

    feature_names = [
        "Global",
        "Kitchen",
        "Laundry",
        "Water heater / AC",
    ]

    print("\nGMM components:")

    for k in range(model.n_components):

        print(f"\nComponent {k}:")
        print(f"Weight = {model.weights_[k]:.4f}")

        for j, feature_name in enumerate(feature_names):

            print(
                f"{feature_name}: "
                f"mean={component_means[k, j]:.4f} kWh, "
                f"std={component_stds[k, j]:.4f} kWh"
            )

def analyze_gmm_temporal_behavior(
    model,
    standardized_data,
):
    """
    Analyze how frequently GMM component assignments
    change between consecutive hours.
    """

    valid_data = standardized_data.dropna(
        subset=MODEL_FEATURE_COLUMNS
    ).copy()

    observations = (
        valid_data[MODEL_FEATURE_COLUMNS]
        .to_numpy()
    )

    valid_data["gmm_component"] = model.predict(
        observations
    )
    #assegna ogni osservazione al componente GMM più probabile.

    total_transitions = 0
    total_switches = 0

    run_lengths = []

    for sequence_id, sequence in valid_data.groupby(
        "sequence_id"
        #prende quindi una sequenza alla volta.
    ):

        states = sequence["gmm_component"].to_numpy()

        if len(states) < 2:
            continue
        #Se c'è una sola ora, non puoi studiare transizioni

        total_transitions += len(states) - 1
        #quante transizioni sono possibili

        switches = states[1:] != states[:-1]
        #switches =[False, True, False, False, True, False], true significa qui il GMM ha cambiato componente.

        total_switches += switches.sum()

        current_run_length = 1
        #sto osservando almeno una ora nello stato corrente

        for i in range(1, len(states)):
            #Scorri gli stati dal secondo in poi

            if states[i] == states[i - 1]:

                current_run_length += 1

            else:

                run_lengths.append(
                    current_run_length
                )

                current_run_length = 1

        run_lengths.append(current_run_length)

    switch_rate = (
        total_switches / total_transitions
    )

    mean_run_length = np.mean(run_lengths)

    median_run_length = np.median(run_lengths)

    return {
        "total_transitions": total_transitions,
        "total_switches": total_switches,
        "switch_rate": switch_rate,
        "mean_run_length": mean_run_length,
        "median_run_length": median_run_length,
    }

def print_gmm_temporal_behavior(results):
    """
    Print temporal statistics of GMM assignments.
    """

    print("\nGMM temporal behavior:")

    print("Number of consecutive-hour transitions:")
    print(results["total_transitions"])

    print("\nNumber of component switches:")
    print(results["total_switches"])

    print("\nSwitch rate:")
    print(f"{results['switch_rate'] * 100:.2f}%")

    print("\nMean component run length:")
    print(f"{results['mean_run_length']:.2f} hours")

    print("\nMedian component run length:")
    print(f"{results['median_run_length']:.2f} hours")


def evaluate_gmm(
    model,
    train_observations,
    validation_observations,
    test_observations,
):
    """
    Evaluate the selected GMM on train, validation and test data.
    """

    train_score = model.score(train_observations)

    validation_score = model.score(
        validation_observations
    )

    test_score = model.score(test_observations)

    return train_score, validation_score, test_score

def print_model_comparison(
    gmm_train_score,
    gmm_validation_score,
    gmm_test_score,
    hmm_train_score,
    hmm_validation_score,
    hmm_test_score,
    gmm_temporal_results,
    hmm_temporal_results,
):
    """
    Print a direct comparison between GMM and HMM.
    """

    print("\nGMM vs HMM comparison:")

    print("\nAverage log-likelihood:")

    print(
        f"Training   | "
        f"GMM={gmm_train_score:.4f} | "
        f"HMM={hmm_train_score:.4f}"
    )

    print(
        f"Validation | "
        f"GMM={gmm_validation_score:.4f} | "
        f"HMM={hmm_validation_score:.4f}"
    )

    print(
        f"Test       | "
        f"GMM={gmm_test_score:.4f} | "
        f"HMM={hmm_test_score:.4f}"
    )

    print("\nTemporal behavior on validation:")

    print(
        f"Switch rate | "
        f"GMM={gmm_temporal_results['switch_rate'] * 100:.2f}% | "
        f"HMM={hmm_temporal_results['switch_rate'] * 100:.2f}%"
    )

    print(
        f"Mean run length | "
        f"GMM={gmm_temporal_results['mean_run_length']:.2f} h | "
        f"HMM={hmm_temporal_results['mean_run_length']:.2f} h"
    )

    print(
        f"Median run length | "
        f"GMM={gmm_temporal_results['median_run_length']:.2f} h | "
        f"HMM={hmm_temporal_results['median_run_length']:.2f} h"
    )

def print_gmm_evaluation(
    train_score,
    validation_score,
    test_score,
):
    """
    Print final GMM log-likelihood scores.
    """

    print("\nSelected GMM evaluation:")

    print(
        f"Training average log-likelihood: "
        f"{train_score:.4f}"
    )

    print(
        f"Validation average log-likelihood: "
        f"{validation_score:.4f}"
    )

    print(
        f"Test average log-likelihood: "
        f"{test_score:.4f}"
    )

def save_final_results(
    best_gmm_result,
    best_hmm_result,
    gmm_train_score,
    gmm_validation_score,
    gmm_test_score,
    hmm_train_score,
    hmm_validation_score,
    hmm_test_score,
    gmm_temporal_results,
    hmm_temporal_results,
    hmm_uncertainty_results,
):
    """
    Save the main project results to a text file.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        RESULTS_DIR / "final_results.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "ENERGY REGIMES PROJECT - FINAL RESULTS\n"
        )

        file.write(
            "======================================\n\n"
        )

        file.write(
            f"Best GMM components: "
            f"{best_gmm_result['k']}\n"
        )

        file.write(
            f"Best HMM states: "
            f"{best_hmm_result['k']}\n\n"
        )

        file.write(
            "AVERAGE LOG-LIKELIHOOD\n"
        )

        file.write(
            f"Train      | "
            f"GMM={gmm_train_score:.4f} | "
            f"HMM={hmm_train_score:.4f}\n"
        )

        file.write(
            f"Validation | "
            f"GMM={gmm_validation_score:.4f} | "
            f"HMM={hmm_validation_score:.4f}\n"
        )

        file.write(
            f"Test       | "
            f"GMM={gmm_test_score:.4f} | "
            f"HMM={hmm_test_score:.4f}\n\n"
        )

        file.write(
            "TEMPORAL BEHAVIOR ON VALIDATION\n"
        )

        file.write(
            f"GMM switch rate: "
            f"{gmm_temporal_results['switch_rate'] * 100:.2f}%\n"
        )

        file.write(
            f"HMM switch rate: "
            f"{hmm_temporal_results['switch_rate'] * 100:.2f}%\n"
        )

        file.write(
            f"GMM mean run length: "
            f"{gmm_temporal_results['mean_run_length']:.2f} h\n"
        )

        file.write(
            f"HMM mean run length: "
            f"{hmm_temporal_results['mean_run_length']:.2f} h\n"
        )

        file.write(
            f"GMM median run length: "
            f"{gmm_temporal_results['median_run_length']:.2f} h\n"
        )

        file.write(
            f"HMM median run length: "
            f"{hmm_temporal_results['median_run_length']:.2f} h\n\n"
        )

        file.write(
            "HMM POSTERIOR UNCERTAINTY\n"
        )

        file.write(
            f"Mean normalized entropy: "
            f"{hmm_uncertainty_results['mean_entropy']:.4f}\n"
        )

        file.write(
            f"Median normalized entropy: "
            f"{hmm_uncertainty_results['median_entropy']:.4f}\n"
        )

        file.write(
            f"Mean maximum posterior probability: "
            f"{hmm_uncertainty_results['mean_max_probability']:.4f}\n"
        )

        file.write(
            f"Posterior confidence < 0.80: "
            f"{hmm_uncertainty_results['low_confidence_fraction'] * 100:.2f}%\n"
        )

    print("\nFinal results saved in:")
    print(output_file)

"""

if __name__ == "__main__":

    from src.data import (
        load_raw_data,
        create_datetime,
        aggregate_hourly_data,
        split_hourly_data,
        standardize_splits,
        prepare_model_data,
    )

    raw_data = load_raw_data()

    time_data = create_datetime(raw_data)

    hourly_data = aggregate_hourly_data(time_data)

    train_data, validation_data, test_data = (
        split_hourly_data(hourly_data)
    )

    (
        train_standardized,
        validation_standardized,
        test_standardized,
        feature_means,
        feature_stds,
    ) = standardize_splits(
        train_data,
        validation_data,
        test_data,
    )

    train_observations, train_lengths = (
        prepare_model_data(train_standardized)
    )

    validation_observations, validation_lengths = (
        prepare_model_data(validation_standardized)
    )

    test_observations, test_lengths = (
        prepare_model_data(test_standardized)
    )

    hmm_models, hmm_results = train_hmm_candidates(
        train_observations,
        train_lengths,
        validation_observations,
        validation_lengths,
    )

    print_hmm_results(hmm_results)

    best_hmm, best_hmm_result = select_best_hmm(
        hmm_models,
        hmm_results,
    )

    print("\nBest HMM:")
    print(f"K = {best_hmm_result['k']}")
    print(
        "Validation average log-likelihood = "
        f"{best_hmm_result['validation_log_likelihood']:.4f}"
    )

    check_forward_algorithm(
        best_hmm,
        validation_observations,
        validation_lengths,
    )

    check_forward_backward_algorithm(
        best_hmm,
        validation_observations,
        validation_lengths,
    )
    check_viterbi_algorithm(
        best_hmm,
        validation_observations,
        validation_lengths,
    )

    hmm_state_means, hmm_state_stds = (
        get_hmm_parameters_original_scale(
            best_hmm,
            feature_means,
            feature_stds,
        )
    )

    print_hmm_states(
        best_hmm,
        hmm_state_means,
        hmm_state_stds,
    )

    print_hmm_transition_matrix(
        best_hmm
    )

    expected_durations = (
        compute_expected_state_durations(
            best_hmm
        )
    )

    print_expected_state_durations(
        best_hmm,
        expected_durations,
    )

    hmm_temporal_results = (
        analyze_hmm_temporal_behavior(
            best_hmm,
            validation_observations,
            validation_lengths,
        )
    )

    print_hmm_temporal_behavior(
        hmm_temporal_results
    )
    (
        hmm_train_score,
        hmm_validation_score,
        hmm_test_score,
    ) = evaluate_hmm(
        best_hmm,
        train_observations,
        train_lengths,
        validation_observations,
        validation_lengths,
        test_observations,
        test_lengths,
    )

    print_hmm_evaluation(
        hmm_train_score,
        hmm_validation_score,
        hmm_test_score,
    )

    hmm_uncertainty_results = (
        analyze_hmm_uncertainty(
            best_hmm,
            validation_observations,
            validation_lengths,
        )
    )

    print_hmm_uncertainty(
        hmm_uncertainty_results
    )

    gmm_models, gmm_results = train_gmm_candidates(
        train_observations,
        validation_observations,
    )

    print_gmm_results(gmm_results)

    best_model, best_result = select_best_gmm(
        gmm_models,
        gmm_results,
    )

    print("\nBest GMM:")
    print(f"K = {best_result['k']}")
    print(
        "Validation log-likelihood = "
        f"{best_result['validation_log_likelihood']:.4f}"
    )

    component_means, component_stds = (
        get_gmm_parameters_original_scale(
        best_model,
        feature_means,
        feature_stds,
        )
    )

    (
        gmm_train_score,
        gmm_validation_score,
        gmm_test_score,
    ) = evaluate_gmm(
        best_model,
        train_observations,
        validation_observations,
        test_observations,
    )

    print_gmm_components(
        best_model,
        component_means,
        component_stds,
    )
    print_gmm_evaluation(
        gmm_train_score,
        gmm_validation_score,
        gmm_test_score,
    )

    gmm_temporal_results = (
        analyze_gmm_temporal_behavior(
            best_model,
            validation_standardized,
        )
    )

    print_gmm_temporal_behavior(
        gmm_temporal_results
    )

    print_model_comparison(
        gmm_train_score,
        gmm_validation_score,
        gmm_test_score,
        hmm_train_score,
        hmm_validation_score,
        hmm_test_score,
        gmm_temporal_results,
        hmm_temporal_results,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_model_selection(
        gmm_results,
        hmm_results,
    )

    plot_log_likelihood_comparison(
        [
            gmm_train_score,
            gmm_validation_score,
            gmm_test_score,
        ],
        [
            hmm_train_score,
            hmm_validation_score,
            hmm_test_score,
        ],
    )

    plot_transition_matrix(
        best_hmm
    )

    plot_hmm_state_profiles(
        hmm_state_means
    )

    plot_temporal_comparison(
        gmm_temporal_results,
        hmm_temporal_results,
    )

    plot_hmm_timeline(
        best_hmm,
        validation_standardized,
        validation_data,
    )

    print("\nFigures saved in:")
    print(FIGURES_DIR)

    save_final_results(
        best_result,
        best_hmm_result,
        gmm_train_score,
        gmm_validation_score,
        gmm_test_score,
        hmm_train_score,
        hmm_validation_score,
        hmm_test_score,
        gmm_temporal_results,
        hmm_temporal_results,
        hmm_uncertainty_results,
    )

"""
import numpy as np


def safe_log(probabilities):
    """
    Compute the logarithm of probabilities, keeping
    zero probabilities equal to negative infinity.
    """

    log_probabilities = np.full_like(
        #crea un nuovo array della stessa forma di probabilities, riempiendolo tutto con -∞.
        probabilities,
        -np.inf,
        dtype=float,
    )

    positive = probabilities > 0

    log_probabilities[positive] = np.log(
        probabilities[positive]
        #seleziona solo i valori corrispondenti a True
    )

    return log_probabilities


def logsumexp(values):
    """
    Compute log(sum(exp(values))) in a numerically stable way.
    """
    #negli HMM i valori possono essere molto negativi tipo -1000, quindi se faccio exp(-1000) ottengo 0, e se faccio exp(-1000) + exp(-1000) ottengo ancora 0, quindi devo fare logsumexp per evitare questo problema.
    maximum = np.max(values)

    if np.isneginf(maximum):
        #np.isneginf(...) controlla se un numero è: -inf
        return -np.inf
        #ritorna -inf se tutti i valori sono -inf

    return maximum + np.log(
        np.exp(values - maximum).sum()
        #sono numeri perfettamente gestibili dal computer.
    )


def gaussian_log_probabilities(
    observations,
    #dati osservati:  Ogni riga è un'ora. Ogni colonna è una feature (global kitchen laundry water/heater AC)
    means,
    #Per ogni stato nascosto hai una media per ogni feature.
    variances,
    #Quindi ogni stato ha una varianza per ciascuna feature
):
    """
    Compute Gaussian log-probabilities for every
    observation and every hidden state.

    Diagonal covariance matrices are assumed.
    """

    number_of_observations, number_of_features = (
        observations.shape
        #ad esempio, 1000 osservazioni, 4 features
    )

    number_of_states = means.shape[0]
    # ad es 3 stati nascosti e, per ogni stato, 4 medie.

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
        #differences misura: quanto ogni osservazione è distante dalla media dello stato corrente.

        log_determinant = np.sum(
            np.log(variances[state])
        )
        #formula della Gaussiana multivariata. Il determinante di una matrice diagonale è semplicemente il prodotto degli elementi diagonali

        quadratic_term = np.sum(
            differences**2 / variances[state],
            axis=1,
            #somma sulle quattro feature della stessa osservazione.
            #versione della distanza di Mahalanobis.
        )

        log_probabilities[:, state] = (
            #prendi tutte le righe : della colonna corrispondente a state
            -0.5
            * (
                number_of_features
                * np.log(2 * np.pi)
                + log_determinant
                + quadratic_term
            )
        )
        #logaritmo della densità di una Gaussiana multivariata con covarianza diagonale.

    return log_probabilities


def forward_log(
    #Per ogni tempo t e per ogni stato k, calcoliamo la probabilità di aver osservato tutti i dati fino a t e di trovarci nello stato k.
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
            #quanto ogni osservazione è compatibile con ogni stato.
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
        #formula iniziale del Forward Algorithm
        log_initial_probabilities
        + emission_log_probabilities[0]
    )

    for time in range(
        #Dal secondo tempo in poi
        1,
        number_of_observations,
    ):

        for current_state in range(
            #consideri tutti gli stati possibili
            number_of_states
        ):

            previous_terms = (
                log_alpha[time - 1]
                #quanto erano probabili i diversi stati al tempo precedente.
                + log_transition_matrix[
                    :,
                    current_state,
                    #colonna dello stato di arrivo
                ]
            )

            log_alpha[
                #ricorsione del Forward Algorithm.
                time,
                current_state,
            ] = (
                emission_log_probabilities[
                    time,
                    current_state,
                ]
                + logsumexp(previous_terms)
                #somma la probabilità di tutti i possibili modi di arrivare nello stato corrente
            )

    log_likelihood = logsumexp(
        log_alpha[-1]
        #l'ultima riga: somma la probabilità di essere in qualsiasi stato all'ultimo tempo, dato tutto ciò che è stato osservato.
    )

    return log_alpha, log_likelihood
    #Una matrice e quanto è probabile l'intera sequenza sotto il modello.

def backward_log(
    #Il backward guarda invece ciò che viene dopo beta_t(i)=P(x_{t+1},...,x_T|z_t=i) e calcola la probabilità di osservare i dati successivi dato che ci troviamo nello stato i al tempo t.
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
        #Produce una matrice: ogni elemento è log P(x_t | z_t = k) per ogni tempo t e stato k.
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
    #ultima riga. nel dominio normale la condizione finale del backward è: βT(i)=1 per ogni stato.

    for time in range(
        number_of_observations - 2,
        #se ho 5 osseervazioni, i tempi sono 0,1,2,3,4. l'ultimo è gia settato a 0.
        -1,
        -1,
    ):

        for current_state in range(
            number_of_states
        ):

            next_terms = (
                #La domanda è: Se adesso sono nello stato 1, cosa può succedere al tempo successivo?
                log_transition_matrix[
                    #probabilità di partire dallo stato 1 e andare in ciascuno stato possibile al tempo successivo.
                    current_state,
                    :
                ]
                + emission_log_probabilities[
                    #Per ogni possibile stato futuro chiedi: quanto è probabile osservare xt+1 se al tempo t+1 sono in quello stato?
                    time + 1,
                    :
                ]
                + log_beta[
                    #la probabilità di osservare tutto quello che viene ancora dopo t+1.
                    time + 1,
                    :
                ]
            )

            log_beta[
                #sommare tutti i possibili stati successivi.
                time,
                current_state,
            ] = logsumexp(next_terms)

    return log_beta


def smoothed_state_probabilities(
    #usa insieme Forward + Backward per calcolare, per ogni tempo t, la probabilità che l’HMM si trovi in ciascuno stato nascosto dopo aver visto tutta la sequenza.
    observations,
    #matrice numerica che contiene i dati che l’HMM vede costruite prendendo solo MODEL_FEATURE_COLUMNS
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
        #log alpha: tutto quello che è successo fino al tempo t, assumendo di essere nello stato k. log_likelihood: log-probabilità dell’intera sequenza osservata.
        observations,
        initial_probabilities,
        transition_matrix,
        means,
        variances,
    )

    log_beta = backward_log(
        #log_beta[t,k] rappresenta: quanto è probabile tutto ciò che viene dopo il tempo t, se al tempo t sono nello stato k.
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
    #Contiene le probabilità degli stati nascosti.

    posterior = (
        #normalizzazione riga per riga
        posterior
        / posterior.sum(
            axis=1,
            keepdims=True,
        )
    )

    return posterior


def viterbi_log(
    #Non sommiamo tutti i percorsi.Manteniamo soltanto:il miglior percorso che arriva nello stato j.
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
        #se al tempo t fossi nello stato k, quanto sarebbe plausibile osservare xt
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
        #viterbi_scores[t, k] significa: log-probabilità del miglior percorso possibile che arriva allo stato k al tempo t.
        (
            number_of_observations,
            number_of_states,
        )
    )

    backpointers = np.zeros(
        #Per ogni: backpointers[t, k]salvi: da quale stato al tempo t−1 proveniva il miglior percorso che arriva allo stato k al tempo t?
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
                #viterbi_scores[time - 1] contiene: miglior percorso che arriva ai vari stati
                + log_transition_matrix[
                    #colonna delle transizioni verso lo stato corrente
                    :,
                    current_state,
                ]
            )

            best_previous_state = np.argmax(
                #l'indice del valore massimo.
                previous_scores
            )

            viterbi_scores[
                #Aggiorna il punteggio Viterbi
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
                #Salva da dove sei arrivato
                time,
                current_state,
            ] = best_previous_state

    best_last_state = np.argmax(
        #scegli il miglior stato finale
        viterbi_scores[-1]
    )

    best_log_probability = (
        #log-probabilità del singolo miglior percorso di stati
        viterbi_scores[
            -1,
            best_last_state,
        ]
    )

    best_path = np.empty(
        number_of_observations,
        #una posizione per ogni osservazione, cioe ora
        dtype=int,
    )

    best_path[-1] = best_last_state
    #Ora conosci l'ultimo stato. Ma devi ricostruire tutti quelli precedenti. Ed è qui che servono i backpointers.

    for time in range(
        number_of_observations - 1,
        0,
        -1,
    ):

        best_path[time - 1] = (
            #so qual è lo stato migliore al tempo time; guardo nel backpointer da quale stato precedente proveniva.
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
        #sommi orizzontalmente, cioè sugli stati della stessa ora.
    )

    normalized_entropy = (
        entropy / np.log(number_of_states)
    )

    return normalized_entropy
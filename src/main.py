from src.config import FIGURES_DIR

from src.data import (
    load_raw_data,
    create_datetime,
    aggregate_hourly_data,
    split_hourly_data,
    standardize_splits,
    prepare_model_data,
)

from src.models import (
    train_gmm_candidates,
    print_gmm_results,
    select_best_gmm,
    get_gmm_parameters_original_scale,
    print_gmm_components,
    evaluate_gmm,
    print_gmm_evaluation,
    analyze_gmm_temporal_behavior,
    print_gmm_temporal_behavior,

    train_hmm_candidates,
    print_hmm_results,
    select_best_hmm,
    check_forward_algorithm,
    check_forward_backward_algorithm,
    check_viterbi_algorithm,
    get_hmm_parameters_original_scale,
    print_hmm_states,
    print_hmm_transition_matrix,
    compute_expected_state_durations,
    print_expected_state_durations,
    analyze_hmm_temporal_behavior,
    print_hmm_temporal_behavior,
    evaluate_hmm,
    print_hmm_evaluation,
    analyze_hmm_uncertainty,
    print_hmm_uncertainty,
    print_model_comparison,
    save_final_results,
)

from src.plots import (
    plot_model_selection,
    plot_log_likelihood_comparison,
    plot_transition_matrix,
    plot_hmm_state_profiles,
    plot_temporal_comparison,
    plot_hmm_timeline,
)

import logging
logging.getLogger("hmmlearn.base").setLevel(logging.ERROR)

def main():
    """
    Run the complete energy-regime analysis pipeline.
    """

    raw_data = load_raw_data()

    time_data = create_datetime(
        raw_data
    )

    hourly_data = aggregate_hourly_data(
        time_data
    )

    train_data, validation_data, test_data = (
        split_hourly_data(
            hourly_data
        )
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
        prepare_model_data(
            train_standardized
        )
    )

    validation_observations, validation_lengths = (
        prepare_model_data(
            validation_standardized
        )
    )

    test_observations, test_lengths = (
        prepare_model_data(
            test_standardized
        )
    )

    gmm_models, gmm_results = (
        train_gmm_candidates(
            train_observations,
            validation_observations,
        )
    )

    print_gmm_results(
        gmm_results
    )

    best_gmm, best_gmm_result = (
        select_best_gmm(
            gmm_models,
            gmm_results,
        )
    )

    print("\nBest GMM:")
    print(
        f"K = {best_gmm_result['k']}"
    )

    component_means, component_stds = (
        get_gmm_parameters_original_scale(
            best_gmm,
            feature_means,
            feature_stds,
        )
    )

    print_gmm_components(
        best_gmm,
        component_means,
        component_stds,
    )

    (
        gmm_train_score,
        gmm_validation_score,
        gmm_test_score,
    ) = evaluate_gmm(
        best_gmm,
        train_observations,
        validation_observations,
        test_observations,
    )

    print_gmm_evaluation(
        gmm_train_score,
        gmm_validation_score,
        gmm_test_score,
    )

    gmm_temporal_results = (
        analyze_gmm_temporal_behavior(
            best_gmm,
            validation_standardized,
        )
    )

    print_gmm_temporal_behavior(
        gmm_temporal_results
    )

    hmm_models, hmm_results = (
        train_hmm_candidates(
            train_observations,
            train_lengths,
            validation_observations,
            validation_lengths,
        )
    )

    print_hmm_results(
        hmm_results
    )

    best_hmm, best_hmm_result = (
        select_best_hmm(
            hmm_models,
            hmm_results,
        )
    )

    print("\nBest HMM:")
    print(
        f"K = {best_hmm_result['k']}"
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

    save_final_results(
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
    )

if __name__ == "__main__":
    main()
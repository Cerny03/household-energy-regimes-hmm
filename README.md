# Discovering Household Energy Consumption Regimes with Gaussian HMMs

## Project overview

This project studies latent household energy-consumption regimes using probabilistic latent-variable models.

The main research question is:

- Does modelling temporal dependence produce more coherent and better-generalizing household energy-consumption regimes than an independent Gaussian mixture model?

A Gaussian Mixture Model (GMM) is used as a non-temporal baseline and compared with a Gaussian Hidden Markov Model (HMM).

## Dataset

The project uses the UCI Individual Household Electric Power Consumption dataset.

Minute-level observations are aggregated into hourly energy consumption.

The model uses four features:

- global energy consumption;
- kitchen sub-metering;
- laundry sub-metering;
- water-heater / air-conditioning sub-metering.

Only complete hours containing 60 valid minute-level measurements are retained for modelling.

Incomplete hours act as boundaries between continuous temporal sequences, so the HMM does not introduce transitions across missing periods.

The data are split chronologically into training, validation and test sets. Features are standardized using training-set statistics only.

## Models

### Gaussian Mixture Model

The Gaussian Mixture Model is used as a non-temporal baseline.

Each hourly observation is assigned independently to one of \(K\) Gaussian components.

Gaussian components use diagonal covariance matrices and their parameters are estimated by Expectation-Maximization with multiple random initializations.

### Gaussian Hidden Markov Model

The Gaussian HMM introduces a latent sequence of household consumption regimes and explicitly models transitions between consecutive hidden states.

Gaussian emissions use diagonal covariance matrices. The model is trained with Baum-Welch using multiple random initializations to reduce sensitivity to local optima.

## Model selection

Model complexity is selected using validation average log-likelihood.

The candidate values are:

K = [2, 5, 8, 9, 10, 11, 12]

The selected models are:

- GMM: K = 11
- HMM: K = 9

For every tested value of \(K\), the HMM achieves a higher validation average log-likelihood than the GMM.

The test set is used only after model selection.

## Inference

Forward, Forward-Backward smoothing and Viterbi inference were implemented explicitly in log-space.

The custom implementations were numerically validated against `hmmlearn`.

Forward-Backward is also used to compute posterior state probabilities and posterior uncertainty.

## Main results

Average log-likelihood:

| Dataset | GMM | HMM |
| --- | ---: | ---: |
| Train | 4.1733 | 4.3781 |
| Validation | 3.7876 | 4.0333 |
| Test | 3.9623 | 4.1362 |

Temporal behavior on validation data:

| Metric | GMM | HMM |
| --- | ---: | ---: |
| Switch rate | 47.54% | 40.47% |
| Mean run length | 2.10 h | 2.46 h |
| Median run length | 1.00 h | 2.00 h |

For the selected HMM:

- mean normalized posterior entropy: 0.0352;
- median normalized posterior entropy: 0.0060;
- mean maximum posterior probability: 0.9723;
- 5.01% of validation hours have maximum posterior probability below 0.80.

The baseline regime is the most persistent inferred HMM state, with a self-transition probability of approximately 0.78 and an expected duration of about 4.53 hours.

Overall, the selected HMM provides a better held-out probabilistic fit and produces more persistent state sequences than the selected GMM.

The temporal comparison should be interpreted descriptively rather than as a controlled same-\(K\) comparison, because the selected models use different numbers of latent regimes.

## Installation

Clone the project from GitHub and move into the repository directory:

```bash
git clone https://github.com/Cerny03/household-energy-regimes-hmm.git
cd household-energy-regimes-hmm
```

Create and activate the Conda environment, then install the required packages:

```bash
conda create -n pml26_env python=3.11
conda activate pml26_env
python -m pip install -r requirements.txt
```

## Run the project

```bash
python run_project.py
```

Generated figures are stored in `figures/` and numerical summaries in `results/`.
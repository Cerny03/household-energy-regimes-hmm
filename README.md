# Discovering Household Energy Consumption Regimes with Gaussian HMMs

## Project overview

This project studies latent household energy-consumption regimes using
probabilistic latent-variable models.

The main research question is:

- Does modelling temporal dependence produce more coherent and better-generalizing household energy-consumption regimes than an independent Gaussian mixture model?

A Gaussian Mixture Model (GMM) is used as a non-temporal baseline and
compared with a Gaussian Hidden Markov Model (HMM).

## Dataset

The project uses the UCI Individual Household Electric Power Consumption
dataset.

Minute-level observations are aggregated into hourly energy consumption.

The model uses four features:

- global energy consumption;
- kitchen sub-metering;
- laundry sub-metering;
- water-heater / air-conditioning sub-metering.

Hours with incomplete minute-level measurements are treated as missing.
Long missing intervals are used as boundaries between continuous
sequences.

## Models

### Gaussian Mixture Model

The GMM models each hourly observation independently.

### Gaussian Hidden Markov Model

The HMM introduces a latent consumption state and a transition matrix
between consecutive states.

Gaussian emissions with diagonal covariance matrices are used.

Models with 2, 3, 4 and 5 latent states are evaluated.

## Inference

Forward, backward/smoothing and Viterbi inference were implemented
explicitly in log-space.

The implementations were numerically validated against `hmmlearn`.

## Main results

The selected GMM and HMM both use 5 latent states.

Average log-likelihood:

| Dataset | GMM | HMM |
| --- | ---: | ---: |
| Train | 3.7888 | 4.0684 |
| Validation | 3.3576 | 3.6502 |
| Test | 3.6950 | 3.9414 |

On validation data:

- GMM switch rate: 31.47%
- HMM switch rate: 29.93%
- GMM median run length: 2 hours
- HMM median run length: 3 hours

The HMM posterior is generally highly concentrated, with a mean maximum
posterior state probability of 0.9847.


## Installation

Clone the project from GitHub and move into the repository directory:

```bash
git clone https://github.com/Cerny03/household-energy-regimes-hmm.git
cd household-energy-regimes-hmm
```

activate the conda environment and install the required packages

```bash
conda create -n pml26_env python=3.11
conda activate pml26_env
python -m pip install -r requirements.txt
```

## Run the project

```bash
python run_project.py
```

Generated figures are stored in figures/ and numerical summaries in
results/.
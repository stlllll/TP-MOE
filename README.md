# TP-MOE: Online Student-\(t\) Processes with an Overall-local Scale Structure for Modelling Non-stationary Data

This repository contains the official implementation of the paper:  
**Online Student-\(t\) Processes with an Overall-local Scale Structure for Modelling Non-stationary Data**  
by Taole Sha and Michael Minyi Zhang (University of Hong Kong).

Presented at the **28th International Conference on Artificial Intelligence and Statistics (AISTATS 2025)**, Mai Khao, Thailand.  
Proceedings of Machine Learning Research (PMLR), Volume 258.

Paper PDF: [https://raw.githubusercontent.com/mlresearch/v258/main/assets/sha25a/sha25a.pdf](https://raw.githubusercontent.com/mlresearch/v258/main/assets/sha25a/sha25a.pdf)

## Overview

Mixture-of-experts (MOE) models are powerful for modeling heterogeneous and non‑stationary data. This work proposes a **mixture of Student‑\(t\) processes (TP‑MOE)** that:

- Handles heavy‑tailed errors and non‑stationarity through a flexible Student‑\(t\) process prior.
- Uses an **overall‑local scale structure** to capture heteroscedastic noise.
- Employs a **Sequential Monte Carlo (SMC) sampler** for fully online inference as data arrive in real‑time.

The model outperforms Gaussian‑process‑based online methods (GP‑MOE, WISKI, OSVGP) on a wide range of synthetic and real‑world datasets, especially when data exhibit heavy tails, abrupt changes, or heteroscedasticity.

## Repository Structure

```
additional_experiments&code&data/
├── data/                          # All datasets used in the paper
│   ├── motorcycle.txt
│   ├── Europe_Brent_Spot_Price_FOB_Daily.csv
│   ├── Processed_DJI.csv
│   ├── co2_canada_diff.csv
│   ├── heart_rate.csv
│   ├── usd_exchange_rate.txt
│   ├── dow_jones_index.csv
│   ├── data_akbilgic.csv
│   ├── Foreign_Exchange_Rates.csv
│   ├── wind.csv
│   └── power.csv
├── utils.py                       # Helper functions (distance, kernel padding)
├── mvt.py                         # Multivariate t‑distribution utilities
├── tp_base_v.py                   # Core Student‑t process class (ESS, prediction)
└── pymp_TPMOE.py                  # Main SMC sampler for TP‑MOE (ParticleTPMOE class)
```

## Dependencies

- Python 3.7+
- NumPy
- SciPy
- autograd
- GPy (for some distance calculations)
- pymp (shared memory parallelism, or replace with `multiprocessing`)
- matplotlib (for plotting)
- CRPS (optional, for scoring)

Install via pip:

```bash
pip install numpy scipy autograd GPy matplotlib pymp-pypi
```

For the CRPS metric, you may need a separate package:

```bash
pip install crps
```

## Usage

### Basic Example (Motorcycle Dataset)

The script `pymp_TPMOE.py` contains a ready‑to‑run example for the motorcycle dataset. To run it:

```bash
python pymp_TPMOE.py
```

This will:
- Load and standardise the motorcycle data.
- Initialise a TP‑MOE model with 100 particles, 16 threads, minibatch size 50.
- Sequentially predict the next observation, update the model, and compute MSE and log‑likelihood.
- Plot the predictions with 95% credible intervals and cluster assignments.

### Custom Data

To use your own data:

```python
import numpy as np
from numpy.random import RandomState
from pymp_TPMOE import ParticleTPMOE

# Data preparation (N x D inputs, N x 1 outputs)
X = ...   # shape (N, D)
Y = ...   # shape (N, 1)
# Standardise recommended
X = (X - X.mean(axis=0)) / X.std(axis=0)
Y = (Y - Y.mean()) / Y.std()

rng = RandomState(42)

# Initialise model with first observation
tpmoe = ParticleTPMOE(
    rng=rng,
    num_threads=4,                 # number of CPU cores
    X=X[0:1], Y=Y[0:1],            # first observation
    J=100,                         # number of particles
    alpha=1.0,                     # initial CRP concentration
    X_mean=np.zeros(D),            # prior mean for inputs
    prior_obs=1.0,                 # lambda_0
    nu=D+1,                        # prior degrees of freedom ( > D-1 )
    psi=0.5*np.eye(D),             # prior covariance matrix
    alpha_a=10.0, alpha_b=1.0,     # Gamma prior on alpha
    mb_size=50                     # minibatch size (None = no minibatch)
)

# Sequential prediction and update
for i in range(1, N):
    m, v, df = tpmoe.predict(X[i:i+1])      # 1‑step ahead prediction
    tpmoe.particle_update(X[i:i+1], Y[i:i+1]) # update model with true observation
```

### Prediction

The `predict` method returns the predictive mean, variance, and degrees of freedom (for the predictive Student‑\(t\) distribution). To obtain credible intervals:

```python
mean, var, df = tpmoe.predict(X_test)
std = np.sqrt(var)
lower = mean - 1.96 * std
upper = mean + 1.96 * std
```

### Saving / Loading

The `ParticleTPMOE` object holds all state (particles, weights, hyperparameters). You can serialise it with `pickle` (note: threading state may not be picklable – save only relevant attributes).

## Datasets

The `data/` folder contains the 14 datasets used in the experiments:

| Dataset | Description | Size |
|---------|-------------|------|
| `motorcycle.txt` | Accelerometer measurement of a motorcycle crash | 94 |
| `Europe_Brent_Spot_Price_FOB_Daily.csv` | Brent crude oil prices | 1025 |
| `Processed_DJI.csv` | Daily features of Dow Jones Industrial Average | 112 |
| `co2_canada_diff.csv` | Annual CO₂ output in Canada | 215 |
| `heart_rate.csv` | MIMIC‑III patient heart rate | 10000 |
| `usd_exchange_rate.txt` | EUR/USD exchange rate | 3139 |
| `dow_jones_index.csv` | 30 stocks (first stock return as output) | 25 |
| `data_akbilgic.csv` | Istanbul Stock Exchange returns | 536 |
| `Foreign_Exchange_Rates.csv` | 22 exchange rates | 101 |
| `wind.csv` | Wind power generation (9 features) | 10950 |
| `power.csv` | Electric power consumption (6 features) | 10484 |

(Plus synthetic datasets that can be generated as in the paper’s code.)

## Reproducing Paper Results

The paper’s tables (MSE, log‑likelihood, CRPS, CPU time) can be reproduced by running the code on each dataset with the settings described in Section 4 (J=100, inducing points=50 for sparse methods, etc.). The provided `pymp_TPMOE.py` includes commented code for loading each dataset.

For comparisons with GP‑MOE, WISKI, and OSVGP, refer to the original implementations:

- [GP‑MOE](https://github.com/michaelzhang01/GPMOE)
- [WISKI](https://github.com/stanfordmlgroup/wiski)
- [OSVGP](https://github.com/trungngv/streaming-sparse-gps)

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@inproceedings{sha2025onlinestudentt,
  title={Online Student-\(t\) Processes with an Overall-local Scale Structure for Modelling Non-stationary Data},
  author={Sha, Taole and Zhang, Michael Minyi},
  booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)},
  pages={1108--1116},
  year={2025},
  volume={258},
  series={Proceedings of Machine Learning Research},
  publisher={PMLR}
}
```

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Acknowledgements

The authors thank the HKU Summer Research Fellowship and the HKU‑URC Seed Fund for Basic Research for New Staff. This code builds upon the [GPMOE](https://github.com/michaelzhang01/GPMOE) repository.

## Contact

For questions or issues, please open an issue on GitHub or contact Taole Sha (u3577089@connect.hku.hk).

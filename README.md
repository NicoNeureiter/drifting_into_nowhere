# Can Bayesian phylogeography reconstruct migrations and expansions in human history?
This repository contains the scripts for the simulation, reconstruction and evaluation presented in our paper "Can Bayesian phylogeography reconstruct migrations and expansions in human history?" (Neureiter N., Ranacher P., Van Gijn R., Bickel B., Weibel R., forthcoming). Below we briefly describe the goals of the study and the necessary steps to reproduce the experiments.

## Study summary
In this study we want to evaluate the performance of Bayesian phylogeographic methods under two movement scenarios (i.e. spatio-temporal processes). In the paper we call these processes migration and expansion, which are implemented here in the form of directional random walks and a grid-based region-growing process respectively (to be found in `src/simulation/migration_simulation.py` and `src/simulation/expansion_simulation.py`). We simulate migrations and expansions under varying degrees of directional trends, attempt to reconstruct the root location based on the simulated phylogeny and tip locations and evaluate the reconstructed root.

## Requirements
The required python packages are listed in the requirements.txt (and can be installed via `pip3 install -r requirements.txt`). This should be sufficient to run the simulations provided in this package. In order to perform and evaluate a phylogenetic reconstruction (as we do in the experiments), an installation of BEAST 1 is required. You can find the free download and set-up instructions on the BEAST 1 website: [https://beast.community/]. In order to run our experiment scripts, the `beast` command and the `treeannotator` command need to be defined in the environment variable (alternatively, you can adapt the scripts in `src/beast_scripts/`).

## Experiment instructions


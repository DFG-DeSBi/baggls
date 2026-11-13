# BaGGLS

## Description
We introduce BaGGLS, a flexible and interpretable probabilistic binary regression model designed for high-dimensional biological inference involving feature interactions. BaGGLS incorporates a Bayesian group global-local shrinkage prior, aligned with the group structure introduced by interaction terms.
This prior encourages sparsity while retaining interpretability, helping to isolate meaningful signals and suppress noise. To enable scalable inference, we employ a partially factorized variational approximation that captures posterior skewness and supports efficient learning even in large feature spaces.
In extensive simulations, we compare BaGGLS to frequentist probit regressions (unconstrained and with L1-penalty) as well as a probit model with Markov Chain Monte Carlo (MCMC) sampling under a horseshoe prior. 
We can show that BaGGLS outperforms the other methods with regard to interaction
detection and is many times faster than MCMC sampling under the horseshoe prior (see simulation). We also demonstrate the usefulness of BaGGLS in the context of interaction discovery from motif scanner outputs (e.g., Find Individual Motif Occurrences (FIMO)) and noisy attribution scores from deep learning models (see application). This shows that BaGGLS is a promising approach for uncovering biologically relevant interaction patterns, with potential applicability across a range of high-dimensional tasks in computational biology.

## Installation

```bash
# Clone the repository
git clone https://gitlab.com/dacs-hpi/baggls.git
cd baggls
```

## Requirements
- Tested on Python 3.12
- Required packages can be found in environment.yml
- **GPU Support (Optional)**: For CNN training and evaluation with GPU acceleration, edit `environment.yml` to uncomment and adjust the CUDA version lines according to your system's CUDA installation.

```bash
# Create conda environment from environment.yml
# Note: Edit environment.yml first if you need GPU support
conda env create -f environment.yml

# Activate the environment
conda activate baggls
```

## Repository Structure
```
baggls/
├── configs/            # Configuration files for models and experiments
├── data/               # Dataset files or scripts to download data
├── src/                # Source code for the method
│   ├── baggls.py      # BaGGLS model implementation
│   ├── application/   # Application to attribution maps
│   └── simulation/    # Simulation experiments
├── results/            # Experimental results and figures
├── environment.yml     # Conda environment specification
└── README.md           # This file
```


## Data

**Experimental dataset** for application available on Zenodo:  
https://doi.org/10.5281/zenodo.17675631

Follow instructions in the README.md in the data folder.
Download and extract to the repository root, or configure custom paths in `configs/paths.yml`.

## Experiments

**Simulation:** Run `src/simulation/simulation_paper.ipynb` or uncomment the main block in `src/simulation/simulation.py` and execute the script.

**Application:** See `src/application/application_eval.ipynb` for analysis notebooks.

## Citation
If you use BaGGLS in your research, please cite:

```bibtex
@article{baggls2025,
  title={BaGGLS: A Bayesian Shrinkage Framework for Interpretable Modeling of Interactions in High-Dimensional Biological Data},
  author={Marta S. Lemanczyk, Lucas Kock, Johanna Schlimme, Nadja Klein, Bernhard Y. Renard},
  journal={arXiv},
  year={2025},
  url={https://arxiv.org/abs/2511.15330}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please contact lucas.kock(at)nus.edu.sg regarding the method or marta.lemanczyk(at)hpi.de regarding the evaluation and code or open an issue on this repository.

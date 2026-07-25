# STNMF with AF-HALS

[![CI](https://github.com/gollischlab/STNMF_with_AFHALS/actions/workflows/ci.yml/badge.svg)](https://github.com/gollischlab/STNMF_with_AFHALS/actions/workflows/ci.yml)
[![Coverage](https://raw.githubusercontent.com/gollischlab/STNMF_with_AFHALS/main/docs/coverage.svg)](https://github.com/gollischlab/STNMF_with_AFHALS/actions/workflows/ci.yml)
[![Documentation status](https://readthedocs.org/projects/stnmf/badge/?version=latest)](https://stnmf.readthedocs.io/en/latest)
[![PyPI version](https://img.shields.io/pypi/v/stnmf.svg)](https://pypi.python.org/pypi/stnmf)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2024.04.22.590506-007ec6)](https://doi.org/10.1101/2024.04.22.590506)

A fast and versatile implementation of spike-triggered non-negative matrix factorization (STNMF) based on accelerated fast hierarchical alternating least squares (AF-HALS) algorithms.

This Python package allows fast inference of receptive-field subunits from the spiking responses of retinal ganglion cells including methods of hyperparameter tuning.

Described in the paper:

> **Zapp SJ, Khani MH, Schreyer HM, Sridhar S, Ramakrishna V, Krüppel S, Protti DA, Mietsch M, Karamanlis D, Gollisch T (2024). Accelerated spike-triggered non-negative matrix factorization reveals coordinated ganglion cell subunit mosaics in the primate retina. *bioRxiv*, 590506.** <https://doi.org/10.1101/2024.04.22.590506>

## Documentation

The documentation is available at <https://stnmf.readthedocs.io>.

## Installation

Install using `pip` from command-line:

```bash
pip install stnmf
```

## Contact

For feedback and bug reports, please use the [Github issue tracker](https://github.com/gollischlab/STNMF_with_AFHALS/issues).

## Development

This project uses [uv] as Python package manager and build backend and orchestrates tasks like linting and tests with [just].

Run `just` to view available commands.

[uv]: https://docs.astral.sh/uv
[just]: https://just.systems

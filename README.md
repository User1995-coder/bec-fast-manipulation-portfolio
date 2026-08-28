# BEC Fast Manipulation

Scientific Python library for modelling and controlling the dynamics of Bose-Einstein condensates.

The project implements reusable numerical models for condensate expansion, shortcut-to-adiabaticity protocols and optical trapping, with an emphasis on modularity, physical validation and reproducibility.

## Core capabilities

### Bose-Einstein condensate dynamics

* Castin-Dum scaling dynamics
* Thomas-Fermi approximation
* Free and thermal expansion
* Numerical integration of time-dependent trapping dynamics

### Shortcut to adiabaticity

* Reverse-engineered STA trajectories
* Retro-sinusoidal scaling protocols
* Numerical search and optimization of admissible control trajectories
* Conversion between scaling dynamics and time-dependent trap frequencies

### Painted optical potentials

* Crossed painted dipole-potential modelling
* Finite-Gaussian beam model
* Mapping between trapping frequencies and experimental control parameters
* Validation of harmonic approximations against the full optical potential

### Robustness and numerical analysis

* Monte Carlo perturbation generation
* Statistical analysis utilities
* Scientific plotting infrastructure
* Reproducible numerical validation tools

## Architecture

The project follows a `src`-based Python package structure:

```text
src/bec_fast_manipulation/
├── analysis/
├── castin_dum/
├── constants/
├── monte_carlo/
├── painted_potential/
├── retro_sinusoidal/
├── sta_optimizer/
├── thermal_expansion/
└── thomas_fermi/
```

The physical models are designed as independent reusable components rather than experiment-specific scripts.

## Physical modelling

A typical modelling chain is:

```text
Initial trapped condensate
        ↓
Thomas-Fermi equilibrium
        ↓
Castin-Dum scaling dynamics
        ↓
Shortcut-to-adiabaticity trajectory
        ↓
Time-dependent trap frequencies
        ↓
Painted optical-potential controls
        ↓
Robustness and numerical validation
```

The code separates:

* physical models;
* numerical integration;
* optimization;
* statistical analysis;
* visualization and reporting.

This separation makes the different components independently testable and reusable.

## Installation

Clone the repository and install the package in editable mode:

```bash
pip install -e ".[dev]"
```

## Tests

Run the complete test suite with:

```bash
python -m pytest
```

The tests cover the numerical models and physical consistency of the implemented equations.

## Technical stack

* Python
* NumPy
* SciPy
* Matplotlib
* pandas
* pytest
* Git

## Project focus

This repository is a cleaned public version of a scientific computing project developed around the fast manipulation of trapped Bose-Einstein condensates.

The public repository focuses on the reusable physics and numerical modelling components.


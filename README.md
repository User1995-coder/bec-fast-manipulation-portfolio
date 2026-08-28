# BEC Fast Manipulation

Numerical tools and reproducible simulations for the fast manipulation of Bose-Einstein condensates.

This project focuses on the modelling and control of Bose-Einstein condensate dynamics, with particular emphasis on fast decompression protocols and painted optical potentials.

## Features

- Castin-Dum scaling dynamics
- Thomas-Fermi modelling
- Thermal and free expansion
- Delta-kick cooling
- Shortcut-to-adiabaticity protocols
- Painted optical potential modelling
- Monte Carlo robustness analysis

## Repository structure

`src/bec_fast_manipulation/` contains the reusable physical and numerical models.

`experiments/` contains the reproducible simulation workflows, figures and numerical results.

`tests/` contains the test suite.

`docs/` and `scripts/physics/` contain additional documentation and physical validation tools.

## Installation

```bash
pip install -e ".[dev]"
```

## Running the tests

```bash
python -m pytest
```

## Scientific workflow

The main modelling workflow is:

```text
BEC equilibrium
    ↓
Thomas-Fermi model
    ↓
Castin-Dum scaling dynamics
    ↓
Shortcut to adiabaticity
    ↓
Time-dependent trap frequencies
    ↓
Painted-potential controls
    ↓
Robustness analysis
```

More detailed information about the physical models, assumptions and numerical parameters can be found in the individual experiment directories.
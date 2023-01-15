A library for checking exact conditions in density functional theory (DFT) approximations.
===============================

## Installation (Linux)

Before continuing, a python virtual environment is recommended. To install, use the following example command:
```bash
git clone https://github.com/pedersor/dft_exconditions.git
cd dft_exconditions
pip install .
```

## Background and usage
Each DFT exact condition (constraint) that we consider has a corresponding local condition (further details will be provided in a forthcoming paper). Such local conditions can be assessed for a given exchange-correlation (XC) approximation using a grid search over applicable variables. For instance, using the PBE functional we can check our default set of exact conditions using

```bash
cd examples/local_conditions
python3 local_conditions_runs.py --func_id gga_c_pbe --condition_string negativity_check 
```
Running the above code produces an output `out/gga_c_pbe.csv`. Here, by default, our grid search is over density variables $r_s \in [0.0001, 5], s \in [0, 5], \zeta \in [0, 1]$. Further details and customization can be found in `local_condition_checks.default_search_variables`. In the output csv we see that PBE correlation is always negative with these density variables.

Exact conditions can also be assessed for given system densities. For instance, we may check whether various atomic self-consistent densities violate exact conditions in approximations. In the following we test neutral atoms H-Ar and their cations and also report the error in ionization energies (IEs). 

```bash
cd examples/exact_conditions
python3 ie_runs.py --xc sogga11
```
Running the above code produces the outputs `ie_checks_SOGGA11.csv` and `ie_errs_SOGGA11.csv`. The former enumerates over the systems and checks whether various exact conditions are satisfied and the latter provides the IE error in the DFT approximation.  

Analysis and visualization over obtained results can be found in `analysis/`. Other code such as `examples/densities/examples.py` is used to reproduce figures that will appear in a forthcoming paper.

## Disclaimer
This library is in *alpha*.

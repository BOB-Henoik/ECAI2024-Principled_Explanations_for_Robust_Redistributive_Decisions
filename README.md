# Supplementaries
This folder contains the developped package and main examples of the article "Principled Explanations for Robust Redistributive Decisions" of the conference ECAI2024.

# How to install
## Software
This package was developped and tested on Python 3.8.6
Gurobi solver is necessary to run the Mixed Integer Linear Optimisation and was tested on Gurobi version 9.1.1 (see https://www.gurobi.com/downloads/gurobi-software/ )
Guroby requires a licence, free for academics, using the command ``grbgetket`` with the licence to set up.
The benchmark was performed on Windows 10 Entreprise with a Intel(R) Core(TM) i9-10885H processor and 32Go of RAM.

## Installing dependencies
#### Build the virtual environment
```shell
python -m venv .\venv
```
#### Activate the virtual environment
For Powershell :
```shell
.\venv\Scripts\Activate.ps1
```

For bash :
```shell
./venv/bin/activate
```

#### Installing dependencies using pip
```shell
pip install -r .\package_requirements.txt
```

# How to reproduce the experiments in the paper

If you run the file ``ECAI.py``, it will first generate the data from Example 1 and apply all the methods on restricted and generalized Lorenz and robust redistributive OWA dominances.
It will also run the computations for Table 1 for 5 criteria. It should run in a few seconds. A commented line contains the computations for n=5,10,20,50 but it will take around 15 hours in total.

# How to experiment with the package

Three other main files are provided :
- ``generation.py`` : generates and saves data, either integers with a fixed sum for studying restricted Lorenz dominance, or general integers to study generalized Lorenz dominance or floats to study robust redistributive OWA.
- ``explanation.py`` : computes the explanations from some previously generated data and saves its length, compute time and number of statements congruent to preferential information.
- ``comparison.py`` : compares the different methods on saved explanations. First computes indicators of length and compute time, then compares the methods by pairs, also on length and compute time.

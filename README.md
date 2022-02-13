# HydrogenMoleculeWaveFunction
The two electron wave function is calculated as in ["The Ground State of the Hydrogen Molecule" (Hubert M. James and Albert Sprague Coolidge) J. Chem. Phys. 1, 825 (1933)](https://home.uni-leipzig.de/pwm/teaching/ep5_ws1213/literature/James_Coolidge_1933_ground_state_hydrogen_molecule.pdf)

Easiest way is to use a conda environment and install the dependencies sympy, vegas, numba and optionally mpi4py.

python WF.py

calculates the five term version of the paper:

final parameters:  [0.675, 0.7, 0.19843750000000002, -0.07500000000000001, -0.0125, 0.17500000000000002]  with energy:  -1.17191(22)

It results in an energy of -1.1719 Hartree, which corresponds to a binding energy of 0.1719 * 2 * 13.6eV = 4.676eV (slightly lower than the experimental value of 4,74eV)



```
python WF.py  --help
usage: WF.py [-h] [--heitler_london] [--only_C00000] [--optimize_internuclear_distance] [--monte_carlo_evals MONTE_CARLO_EVALS]

Hydrogen molecule ground state calculation, if mpi4py is installed you can start it with mpirun for faster calculation

optional arguments:
  -h, --help            show this help message and exit
  --heitler_london      Use Heitler London ansatz instead of J. Chem. Phys. 1, 825 (1933)
  --only_C00000         Only one term of J. Chem. Phys. 1, 825 (1933) ansatz used instead of the default 5 terms (Table I and II)
  --optimize_internuclear_distance
                        Optimize internuclear distance, otherwise 1.4 a.u. is used (2*0.7)
  --monte_carlo_evals MONTE_CARLO_EVALS
                        Number of Monte Carlo evaluations, reduce for faster calculation (default 1e6)
```

mpirun -n 4 python WF.py

will run it with mpi using 4 processes, if mpi4py is installed

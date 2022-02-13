# Copyright (c) 2022 Detlef Schmicker, see also the LICENSE file.

from sympy import symbols, exp, sqrt, lambdify, simplify, diff

import numba

import vegas
import time
import sys

import argparse

# local comments only: using envirenment huggingface at the moment

vars = symbols('x1 x2 x3 x4 x5 x6 a b c00020 c00110 c10000 c00001')

x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001 = vars

parser = argparse.ArgumentParser(description='Hydrogen molecule ground state calculation, if mpi4py is installed you can start it with mpirun for faster calculation')
parser.add_argument('--heitler_london', dest='heitler_london', action='store_true', help='Use Heitler London ansatz instead of J. Chem. Phys. 1, 825 (1933)')
parser.add_argument('--only_C00000', dest='only_C00000', action='store_true', help='Only one term of J. Chem. Phys. 1, 825 (1933) ansatz used instead of the default 5 terms (Table I and II)')
parser.add_argument('--optimize_internuclear_distance', dest='optimize_internuclear_distance', action='store_true', help='Optimize internuclear distance, otherwise 1.4 a.u. is used (2*0.7)')
parser.add_argument('--monte_carlo_evals', dest='monte_carlo_evals', type=float, default=1e6, help='Number of Monte Carlo evaluations, reduce for faster calculation (default 1e6)')
args = parser.parse_args()

optimize_internuclear_distance_step = 0
if args.optimize_internuclear_distance:
    optimize_internuclear_distance_step = 0.1

start_search = [0.7, 0.7, 0, 0, 0, 0]
step_search  = [-0.1, optimize_internuclear_distance_step, 0.1, 0.1, 0.1, 0.1]
if args.only_C00000:
    step_search  = [-0.1, optimize_internuclear_distance_step, 0, 0, 0, 0]


def v(x1,x2,x3):
    return 1/sqrt(x1**2+x2**2+x3**2)

def wf(x1,x2,x3, a ,b ,c):
    return exp(-a*sqrt((x1-b)**2+x2**2+x3**2)) +exp(-a*sqrt((x1+b)**2+x2**2+x3**2))


# The Ground State of the Hydrogen Molecule
# Hubert M. James and Albert Sprague Coolidge
# J. Chem. Phys. 1, 825 (1933)
# As in https://home.uni-leipzig.de/pwm/teaching/ep5_ws1213/literature/James_Coolidge_1933_ground_state_hydrogen_molecule.pdf

# b is half the internuclear distance in our calculation 0.7 in our calculation is 1.4 in the paper
# atomic units give Energy in Hartree (atomic energy unit back in 1933 seems to be the ionization energy of hydrogen atom 13.6 eV, which is 0.5 Hartree, therefore we get halve the energy values from the paper)

ra1 = sqrt((x1-b)**2+x2**2+x3**2)
rb1 = sqrt((x1+b)**2+x2**2+x3**2)

ra2 = sqrt((x4-b)**2+x5**2+x6**2)
rb2 = sqrt((x4+b)**2+x5**2+x6**2)

r12 = sqrt((x1-x4)**2+(x2-x5)**2+(x3-x6)**2)

lambda1=ra1+rb1
lambda2=ra2+rb2
mu1 = ra1-rb1
mu2 = ra2-rb2

def wf2(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001):
    return exp(-a*(lambda1+lambda2)) * (1 + c00020*(mu2**2+mu1**2) + c00110*(2*mu1*mu2) + c10000*(lambda1+lambda2) + c00001*2*r12) 


#Heitler London
def wf_hl(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001):
   return exp(-a*(ra1+rb2)) + exp(-a*(rb1+ra2))



if args.heitler_london:
    print("Heitler London calculation!!")
    wf2 = wf_hl
    step_search = [-0.1, optimize_internuclear_distance_step, 0, 0, 0, 0]

def v(x1,x2,x3):
    return 1/sqrt(x1**2+x2**2+x3**2)


lap =  (   diff(wf2(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001), x1, 2)
           +diff(wf2(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001), x2, 2)
           +diff(wf2(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001), x3, 2)
           +diff(wf2(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001), x4, 2)
           +diff(wf2(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001), x5, 2)
           +diff(wf2(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001), x6, 2)
        )

integrand_wf = lambdify(vars,simplify(wf2(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001)*wf2(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001)),'numpy')

integrand_wf_numba = numba.jit(integrand_wf)

integrand_full = lambdify(vars, wf2(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001)* (
         -0.5 * lap +
        ( -v(x1-b,x2,x3) -v(x1+b,x2,x3) -v(x4-b,x5,x6) -v(x4+b,x5,x6) +v(x1-x4,x2-x5,x3-x6) 
        ) * wf2(x1,x2,x3,x4,x5,x6,a,b,c00020, c00110, c10000, c00001) 
        ),'numpy'
    )

integrand_full_numba = numba.jit(integrand_full)

integrator = vegas.Integrator(6*[[-10,10]])
if integrator.mpi_rank == 0:
    print("starting")
    sys.stdout.flush()
def Energy(y):
    a = y[0]
    b = y[1]
    c00020 = y[2]
    c00110 = y[3]
    c10000 = y[4]
    c00001 = y[5]
    if integrator.mpi_rank == 0:
        start = time.time()
    result1 = integrator(lambda x: integrand_wf_numba(x[0],x[1],x[2],x[3],x[4],x[5],a,b,c00020, c00110, c10000, c00001), nitn=10, neval=args.monte_carlo_evals)
    if integrator.mpi_rank == 0:
        end = time.time()
        print('elapsed time: %.2f s' % (end - start))
        print(result1.summary())
        print('result1 = %s    Q = %.2f' % (result1, result1.Q))
        sys.stdout.flush()
    result2 = integrator(lambda x: integrand_full_numba(x[0],x[1],x[2],x[3],x[4],x[5],a,b,c00020, c00110, c10000, c00001), nitn=10, neval=args.monte_carlo_evals)
    if integrator.mpi_rank == 0:
        end = time.time()
        print('elapsed time: %.2f s' % (end - start))
        print(result2.summary())
        print('result2 = %s    Q = %.2f' % (result2, result2.Q))
        print("Energy calculated: ", result2 / result1 + 1 / abs(2*b), "at ", y)
        sys.stdout.flush()
    return result2 / result1 + 1 / abs(2*b)


def optimize(f, point, steps, iter=5):
    bestvalue = f(point)
    bestpoint = point[:]
    for r in range(iter):
        for d in range(len(point)):
            if steps[d] != 0:
                if integrator.mpi_rank == 0:
                    print("iteration: ", r, " direction ", d, "best: ", bestvalue, "point", bestpoint)
                point[d] += steps[d]
                tmpvalue = f(point)
                if tmpvalue < bestvalue:
                    bestvalue = tmpvalue
                    bestpoint = point[:]
                    if integrator.mpi_rank == 0:
                        print("new best: ", bestvalue, "point", bestpoint)
                        sys.stdout.flush()
                else:
                    steps[d] = -steps[d]
                    point[d] += steps[d]
                ok = True
                while ok:
                    point[d] += steps[d]
                    tmpvalue = f(point)
                    if tmpvalue < bestvalue:
                        bestvalue = tmpvalue
                        bestpoint = point[:]
                        if integrator.mpi_rank == 0:
                            print("new best: ", bestvalue, "point", bestpoint)
                            sys.stdout.flush()
                    else:
                        ok = False
                        point[d] -= steps[d]
        for d in range(len(steps)):
            steps[d] *= 0.5
    return bestpoint, bestvalue




def main():
    position, energy = optimize(Energy, start_search, step_search, iter=7)
    if integrator.mpi_rank == 0:
        print("final parameters: ", position, " with energy: ", energy)
        sys.stdout.flush()

if __name__ == '__main__':
    main()
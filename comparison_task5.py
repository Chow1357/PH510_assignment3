#!/opt/software/anaconda/python-3.10.9/bin/python
import numpy as np 
from mpi4py import MPI 

#importing functiosn from other scripts to save rewriting
from Task2_optimised import estimate_greens_function
from Task4 import (
    physical_to_grid,
    make_boundary_array,
    make_zero_charge,
    make_uniform_charge,
    make_gradient_charge,
    make_exponential_charge,
    potential_from_greens,
)
from part1 import solve_poisson_sor   # change filename if needed

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if __name__ == "__main__":
    # defining the main parameters 
    N = 50
    L = 100.0 # cm
    nwalkers = 200000
    seed = 1234

    # three points asked to test first (from task 3) 
    points = {
        "centre": (50.0, 50.0),
        "corner": (2.0, 2.0),
        "face": (2.0, 50.0),
    }

    #The three boundary condition cases stated in the first part of task 4 
    boundary_cases = {
        "all_plus_100": (100.0, 100.0, 100.0, 100.0),
        "tb_plus100_lr_minus100": (100.0, 100.0, -100.0, -100.0),
        "top_left_200_bottom_0_right_minus400": (200.0, 0.0, 200.0, -400.0),
    }

    # charge cases required
    charge_cases = {
        "zero_charge": make_zero_charge(N),
        "uniform_10C": make_uniform_charge(N, rho=10.0),
        "gradient_top_to_bottom":make_gradient_charge(N),
        "exponential_centre": make_exponential_charge(N, L_m=1.0),
    }
    # storage for output rows
    rows = []
    
    # loop over the points
    for point_name, (x_cm, y_cm) in points.items():
        start_i, start_j = physical_to_grid(x_cm, y_cm, L_cm, N)

        g_charge, g_charge_std, g_charge_stderr, mean_visits, std_visits, boundary_prob = estimate_greens_function(start_i, start_j, N, nwalkers, seed=seed, chunk_size=2500
        )
    
        if rank == 0:
            for charge_name, f in charge_cases.items():
                for boundary_name, (V_top, V_bottom, V_left, V_right) in boundary_cases.items():
                boundary_values = make_boundary_array(N, V_top, V_bottom, V_left, V_right)
                # reconstruct the stochastic potential from the Green's function 
                phi_green, phi_boundary, phi_charge, sigma_green = potential_from_greens(
                    g_charge, g_charge_stderr, boundary_prob, boundary_values, f)

                phi_det_grid, iterations, omega, delta = solve_poisson_sor(N, f, V_top, V_bottom, V_left, V_right, target=target)

                # compute deterministic potential 
                phi_det = phi_det_grid[start_i, start_j]
                # compute absolute difference between stochastic and deterministic potentials
                difference = abs(phi_green - phi_det)
                # this become row of the output table
                rows.append([
                    point_name,
                    boundary_name,
                    charge_name,
                    phi_green,
                    sigma_green,
                    phi_det,
                    difference,
                ])
                
                


        

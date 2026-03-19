#!/opt/software/anaconda/python-3.10.9/bin/python
import numpy as np
from mpi4py import MPI
from Task2_optimised import estimate_greens_function

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#converting physcal points to grid points
def physical_to_grid(x_cm, y_cm, L_cm, N):
    h = L_cm / (N + 1)
    i = int(round(y_cm / h))
    j = int(round(x_cm / h))
    i = max(1, min(N, i))
    j = max(1, min(N, j))
    return i, j

# building a new boundary potential array 
def make_boundary_array(N, V_top, V_bottom, V_left, V_right):
    B = np.zeros((N + 2, N + 2), dtype=float)

    B[0, :] = V_bottom
    B[N + 1, :] = V_top
    B[:, 0] = V_left
    B[:, N + 1] = V_right

    B[0, 0] = 0.5 * (V_bottom + V_left)
    B[0, N + 1] = 0.5 * (V_bottom + V_right)
    B[N + 1, 0] = 0.5 * (V_top + V_left)
    B[N + 1, N + 1] = 0.5 * (V_top + V_right)

    return B
# Zero-charge array 
#creating an interior charge-density array with zero everywhere
def make_zero_charge(N):
    return np.zeros((N + 2, N + 2), dtype=float)

# Turning Greens functions into a potential 
def potential_from_greens(G, G_stderr, boundary_prob, B, f):
    phi_boundary = np.sum(boundary_prob * B)
    phi_charge = np.sum(G * f)
    phi_total = phi_boundary + phi_charge
    sigma_charge = np.sqrt(np.sum((G_stderr * f) ** 2))
    return phi_total, phi_boundary, phi_charge, sigma_charge

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

# zero charge array for the first stage
# no interior charge 
f = make_zero_charge(N)

# loop over the three points 
for point_name, (x_cm, y_cm) in point.items()
    start_i, start_j = physical_to_grid(x_cm, y_cm, L, N)
# evaluate greens function at the point stated
G, G_std, G_stderr, mean_visits, std_visits, boundary_prob = estimate_greens_function(start_i, start_j, N, nwalkers, seed=seed, chunk_size=2500)


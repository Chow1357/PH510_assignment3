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

# uniform charge density over the interior
def make_uniform_charge(N, rho=10.0):
    """
    Uniform charge density over the interior grid.
    Since the square is 1 m x 1 m, using rho=10.0 corresponds
    to 10 C spread uniformly over the whole grid.
    """
    f = np.zeros((N + 2, N + 2), dtype=float)
    f[1:N+1, 1:N+1] = rho
    return f

# charge gradient from bottom (0) to top (1 C m^-2)
def make_gradient_charge(N):
    """
    charge density varies linearly from 0 at the bottom 
    to 1 C m^-2 at the top.
    """
    f = np.zeros((N + 2, N + 2), dtype=float)

    for i in range(1, N + 1):
        yfrac = (i - 1) / (N - 1)
        f[i, 1:N+1] = yfrac

    return f

# exponentially decaying charge distribution at the centre 
def make_exponential_charge(N, L_m=1.0):
    """
    Exponentially decaying charge distribution exp(-10 r)
    centred in the middle of the grid
    """
    f = np.zeros((N + 2, N + 2), dtype=float)
    h = L_m / (N + 1)

    xc = 0.5 * L_m
    yc = 0.5 * L_m

    for i in range(1, N + 1):
        y = i * h
        for j in range (1, N + 1):
            x = j * h
            r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
            f[i, j] = np.exp(-10.0 * r)

    return f

# Turning Greens functions into a potential 
def potential_from_greens(G, G_stderr, boundary_prob, B, f):
    phi_boundary = np.sum(boundary_prob * B)
    phi_charge = np.sum(G * f)
    phi_total = phi_boundary + phi_charge
    sigma_charge = np.sqrt(np.sum((G_stderr * f) ** 2))
    return phi_total, phi_boundary, phi_charge, sigma_charge

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

    # loop over the three points 
    for point_name, (x_cm, y_cm) in points.items():
        start_i, start_j = physical_to_grid(x_cm, y_cm, L, N)
        # evaluate greens function at the point stated
        G, G_std, G_stderr, mean_visits, std_visits, boundary_prob = estimate_greens_function(start_i, start_j, N, nwalkers, seed=seed, chunk_size=2500)

        #ensure only one MPI process prints the results
        if rank ==0:
            print()
            print(f"Point: {point_name}")
            print(f"Physical coordinates: ({x_cm:.1f} cm, {y_cm:.1f} cm)")
            print(f"Grid coordinates: ({start_i}, {start_j})")
            print(f"Boundary probability sum: {np.sum(boundary_prob):.6f}")

            # loop over al charge cases 
            for charge_name, f in charge_cases.items():
                print(f" Charge case: {charge_name}")

                # loop over the three boundary cases 
                for case_name, (V_top, V_bottom, V_left, V_right) in  boundary_cases.items():
                    B = make_boundary_array(N, V_top, V_bottom, V_left, V_right)
                    # compute the potential 
                    phi_total, phi_boundary, phi_charge, sigma_charge = potential_from_greens(G, G_stderr, boundary_prob, B, f)

                    # print functions 
                    print(f"  Case: {case_name}")
                    print(f"  phi_total    = {phi_total:.6f} V")
                    print(f"  phi_boundary = {phi_boundary:.6f} V")
                    print(f"  phi_charge   = {phi_charge:.6f} V")
                    print(f"  sigma_charge = {sigma_charge:.6f} V")

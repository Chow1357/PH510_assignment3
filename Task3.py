#!/opt/software/anaconda/python-3.10.9/bin/python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI
from Task2_optimised import estimate_greens_function

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#--------------------------
# TASK 3
#--------------------------
# to a corresponding position on the grid
def physical_to_grid(x_cm, y_cm, L_cm, N):
    """
    Convert physical coordinates in cm to interior grid indices.
    """

    h = L_cm / (N+1)

    i = int(round(y_cm / h))
    j = int(round(x_cm / h))

    # clamping values so they stay within the grid
    i = max(1, min(N, i))
    j = max(1, min(N, j))

    return i, j

if __name__ == "__main__":
    N = 50
    nwalkers = 200000
    L = 100  # cm

    # defining the different points to test on the grid in length scale
    points = {"centre": (50, 50),"corner": (2, 2),"face": (2, 50)}

    for name, (x_cm, y_cm) in points.items():
        start_i, start_j = physical_to_grid(x_cm, y_cm, L, N)

        G, G_std, G_stderr, mean_visits, std_visits, boundary_prob = estimate_greens_function(start_i, start_j, N, nwalkers, seed=1234, chunk_size=2500)

        if rank == 0:
            print()
            print(f"Number of walkers: {nwalkers}")
            print(f"Physical point: ({x_cm} cm, {y_cm} cm)")
            print(f"Grid index: ({start_i}, {start_j})")
            print("Green's function:", G[start_i, start_j])
            print("Standard deviation:", G_std[start_i, start_j])
            print("Standard error:", G_stderr[start_i, start_j])
            print(f"Sum of boundary probabilities: {np.sum(boundary_prob):.6f}")
            plt.figure(figsize=(6, 5))
            plt.imshow(G[1:N+1, 1:N+1], origin="lower", extent=[0, L, 0, L])
            plt.colorbar(label="Green's function")
            plt.scatter(x_cm, y_cm, marker="x")
            plt.xlabel("x (cm)")
            plt.ylabel("y (cm)")
            plt.title(f"Green's function from start point ({x_cm} cm, {y_cm} cm)")
            plt.tight_layout()
            plt.savefig(f"greens_function_{x_cm}_{y_cm}.png", dpi=300)
            plt.close()


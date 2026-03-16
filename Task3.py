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

#
def is_boundary(i, j, N):
    """
    function determines when the walker has
    reached the absorbing boundary
    """
    return i == 0 or i == N+1 or j == 0 or j == N+1
# One random walk test
def single_walk(start_i, start_j, N, rng):
    """
    performs a random walk from a start point on the grid (start_i, start_j)
    and creates an array and records how many times this walker visits each point.
    """
    visits = np.zeros((N+2, N+2), dtype=int)

    i, j = start_i, start_j

    while not is_boundary(i, j, N):
        visits[i, j] += 1
        # this matches 
        step = rng.integers(4) # random number generator with equal probability (directions) 
        if step == 0:
            i += 1
        elif step == 1:
            i -= 1
        elif step == 2:
            j += 1
        else:
            j -= 1
    return visits

#--------------------------
# TASK 3
#--------------------------
if __name__ == "__main__":

    L = 100 

    # defining the different points to test on the grid in length scale
    points = [(50, 50), (2, 2), (2, 50)] # in cm 

    # function that converts a physical position in cm 
    # to a corresponding position on the grid
    def physical_to_grid(x_cm, y_cm, L_cm, N):
        """
        """

        h = L_cm / (N+1)
    
        i = int(round(x_cm / h))
        j = int(round(y_cm / h))

        # clamping values so they stay within the grid
        i = max(1, min(N, i))
        j = max(1, min(N, j))
        return i, j

    for x_cm, y_cm in points: 
 
        start_i, start_j = physical_to_grid(x_cm, y_cm, L, N)
 
        G, G_std, G_stderr, mean_visits, std_visits = estimate_greens_function(start_i, start_j, N, nwalkers, factor=0.25, seed=1234)

        if rank == 0:
            print()
            print(f"Physical point: ({x_cm} cm, {y_cm} cm)")
            print(f"Grid index: ({start_i}, {start_j})")
            print("Green's function:", G[start_i, start_j])
            print("Standard deviation:", G_std[start_i, start_j])
            print("Standard error:", G_stderr[start_i, start_j])

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
    

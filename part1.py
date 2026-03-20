#!/opt/software/anaconda/python-3.10.9/bin/python
"""
thsi code is run with python3.10.9
Solve the 2D Poisson equation on a square grid using SOR 

MIT License

Copyright (c) 2026 Chow1357

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def make_boundary_array(n, top_bc, bottom_bc, left_bc, right_bc):
    """
    create boundary values on a halo grid.
    Interior points are 1..N and boundaries are 0 and N+1
    """

    phi_grid = np.zeros((n + 2, n + 2), dtype=float)

    phi_grid[0, :] = bottom_bc
    phi_grid[n + 1, :] = top_bc
    phi_grid[:, 0] = left_bc
    phi_grid[:, n +1] = right_bc

    #corner values
    phi_grid[0, 0] = 0.5 * (bottom_bc + left_bc)
    phi_grid[0, n + 1] = 0.5 * (bottom_bc + right_bc)
    phi_grid[n + 1, 0] = 0.5 * (top_bc + left_bc)
    phi_grid[n + 1, n + 1] = 0.5 * (top_bc + right_bc)

    return phi_grid

# defining the poisson over relaxation method
def poisson_sor_step(phi_grid, source_term, n, h, omega_value):
    """
    perform one SOR sweep and return the maximum change.
    """
    max_change = 0.0

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            old_value = phi_grid[i, j]

            # Poisson update from the neighbours and the source term f
            phi_star = 0.25 * (
                phi_grid[i + 1, j] + phi_grid[i - 1, j] +
                phi_grid[i, j + 1] + phi_grid[i, j - 1] -
                h * h * source_term[i, j]
            )
            # SOR update
            phi_grid[i, j] = (1.0 - omega_value) * old_value + omega_value * phi_star

            change = abs(phi_grid[i, j] - old_value)
            if change > max_change:
                max_change = change
    return max_change

def solve_poisson_sor(n, source_term, boundaries, target=1e-8):
    """
    Solve Poisson's equation on a halo grid using SOR.
    """
    top_bc, bottom_bc, left_bc, right_bc = boundaries
    # grid spacing
    h = 1.0 / (n + 1)
    # SOR parameter for square grid
    omega_value = 2.0 / (1.0 + np.sin(np.pi / n))

    # potential array
    phi_grid = make_boundary_array(n, top_bc, bottom_bc, left_bc, right_bc)
    #starting the loop which repeats the SOR sweep until convergence is acheived
    #i.e tolerance is exceeded
    delta_value = 1.0
    iteration_count = 0

    while delta_value > target:
        delta_value = poisson_sor_step(phi_grid, source_term, n, h, omega_value)
        iteration_count += 1

    return phi_grid, iteration_count, omega_value, delta_value

def make_example_charge(n):
    """
    Example source term for standalone testing.
    """
    # source term for poisson
    source_term = np.zeros([n + 2, n + 2], dtype=float)

    # example: positive and negative source
    source_term[n // 4 + 1, n // 4 + 1] = 100.0
    source_term[3 * n //4, 3 * n // 4 ] = -100.0

    return source_term


if __name__ == "__main__":

    # defining the grid on which we can solve poisson's equation
    N = 50

    # target value for change in solution
    TARGET = 1e-8

    # defining the boundary conditions and introducing a potential difference
    V_TOP = 100.0
    V_BOTTOM = 0.0
    V_LEFT = 0.0
    V_RIGHT = 0.0

    boundaries = (V_TOP, V_BOTTOM, V_LEFT, V_RIGHT)
    source_term = make_example_charge(N)

    solution_grid, n_iterations, sor_omega, final_delta = solve_poisson_sor(
        N, source_term, boundaries, target=TARGET
    )

    #print functions for important parameters
    print("Iterations =", n_iterations)
    print("Omega =", sor_omega)
    print("Final max change =", final_delta)

    # plotting phi
    plt.figure()
    # plotting only interior points instead of the full halo grid
    plt.imshow(solution_grid[1:N+1, 1:N+1], origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label='Potential (phi)')
    plt.title('Solution of Poisson Equation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("phi.png", dpi=300)
    plt.close()







import numpy as np 
import matplotlib.pyplot as plt

#------------------------------
# Boundary test 
#stops walk when it has reached the boundary
#------------------------------
def is_boundary(i, j, N):
    """
    """
    return i == 0 or i == N+1 or j == 0 or j == N+1


#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np


def initialize_particles(N, L_x, L_y, sigma):
    """
    Generate initial positions and orientations
    with a minimum distance condition.
    """

    x = np.zeros(N)
    y = np.zeros(N)

    min_dist = 1.05 * sigma


    for i in range(N):

        while True:

            xi = np.random.uniform(-L_x/2, L_x/2)
            yi = np.random.uniform(-L_y/2, L_y/2)

            valid = True

            for j in range(i):

                dx = xi - x[j]
                dy = yi - y[j]

                # Minimum image convention
                dx -= L_x * np.round(dx / L_x)
                dy -= L_y * np.round(dy / L_y)

                r = np.sqrt(dx**2 + dy**2)

                if r < min_dist:
                    valid = False
                    break


            if valid:
                x[i] = xi
                y[i] = yi
                break


    # Initial orientations

    theta = np.random.uniform( 0, 2*np.pi, N )


    return x, y, theta







#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


def mean_active_force(Fa_history):
    """
    Mean active force over particles.
    """

    return np.mean(Fa_history, axis=0)



def mean_score(q_history):
    """
    Mean score over particles.
    """

    return np.mean(q_history, axis=0)



def variance_active_force(Fa_history):
    """
    Variance of active force.
    """

    return np.var(Fa_history, axis=1)



def density_profile_x(x_history, L_x, bins=50):
    """
    Density profile along x direction.
    """

    density = []

    for x in x_history:

        hist, edges = np.histogram(
            x,
            bins=bins,
            range=(-L_x/2, L_x/2),
            density=True
        )

        density.append(hist)


    return np.array(density)






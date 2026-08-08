#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =========================
# Simulation parameters
# =========================

N = 60          # number of particles
T = 2000000      # number of time steps
dt = 0.001          # time step

L_x = 32        # box size x
L_y = 18          # box size y


# =========================
# Active particle parameters
# =========================

Dr = 0.5            # rotational diffusion
mu = 1.0            # mobility


# =========================
# WCA parameters
# =========================

sigma = 1.0
epsilon = 1.0

rc = 2**(1/6)*sigma


# =========================
# Learning parameters
# =========================

alpha = 1000        # social learning rate
alpha_q = 0.1       # environmental learning rate

q_d = 0.25          # dark region score
q_l = 0.75          # light region score


# =========================
# Saving parameters
# =========================

save_interval = 1000

n_runs = 1


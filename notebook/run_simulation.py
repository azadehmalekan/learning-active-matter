#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import HTML
import time


# In[3]:


project_root = Path.cwd().parent
sys.path.append(str(project_root))

from src.parameters import *
from src.initialization import initialize_particles
from src.learning import initialize_learning, social_learning_numba, environmental_learning
from src.simulation import run_simulation, run_multiple_simulations
from src.io  import save_multiple_runs


# In[4]:


print(N)
print(T)


# In[4]:


start = time.time()
results = run_multiple_simulations( n_runs= n_runs , N = N,T = T, dt = dt, L_x=L_x, L_y=L_y, sigma=sigma, epsilon=epsilon, mu=mu,Dr=Dr, alpha=alpha, alpha_q=alpha_q, q_d=q_d, q_l=q_l, save_interval=save_interval)
print("Elapsed time:", time.time()-start)


# In[7]:


save_multiple_runs(results)











#!/usr/bin/env python
# coding: utf-8

# In[7]:


from src.io import load_multiple_runs

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import sys


project_root = Path.cwd().parent
sys.path.append(str(project_root))

from src.parameters import *
from src.io import load_multiple_runs
from analysis.observables import ( mean_active_force, mean_score, variance_active_force, variance_score, density_profile_x )


# In[2]:


results = load_multiple_runs()

print("Number of runs:", len(results))


# In[3]:


mean_Fa_all, std_Fa_all  = mean_active_force(results)
mean_q_all, std_q_all  = mean_score(results)


# In[4]:


variance_Fa, error_variance_Fa = variance_active_force(results)


# In[9]:


variance_q, error_variance_q = variance_score(results)







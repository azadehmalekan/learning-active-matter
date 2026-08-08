#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path.cwd().parent
sys.path.append(str(project_root))


from src.io import load_multiple_runs
from src.parameters import *


from analysis.animation import (
    animate_particles_q_light,
    animate_particles_Fa,
    animate_particles_q_Fa_light,animate_all_runs_concatenated
)


# In[14]:


results = load_multiple_runs()

print("Number of runs:", len(results))


# In[15]:


ani = animate_particles_q_Fa_light(
    x_hist,
    y_hist,
    q_hist,
    Fa_hist,
    L_x,
    L_y,
    dt,
    save_interval
)


HTML(ani.to_jshtml())


# In[16]:


ani = animate_all_runs_concatenated(
    results,
    L_x,
    L_y,
    dt,
    save_interval,
    color_by="q"
)

HTML(ani.to_jshtml())


# In[17]:


ani = animate_all_runs_concatenated(
    results,
    L_x,
    L_y,
    dt,
    save_interval,
    color_by="Fa"
)

HTML(ani.to_jshtml())


# In[22]:


Path("../figures").mkdir(exist_ok=True)

ani.save(
    "../figures/all_runs_q.mp4",
    fps=10
)


# In[20]:


ani.save(
    "figures/all_runs_q.mp4",
    fps=10
)


# In[ ]:





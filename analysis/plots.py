#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


def plot_particles(x, y, L_x, L_y, step=-1):

    plt.figure(figsize=(6,4))

    plt.scatter(
        x[step],
        y[step],
        s=10
    )

    plt.xlim(-L_x/2, L_x/2)
    plt.ylim(-L_y/2, L_y/2)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.axis("equal")
    plt.title(f"Particles at step {step}")

    plt.show()



def plot_learning(Fa, q, particle=0):

    fig, ax = plt.subplots(1,2, figsize=(10,4))


    ax[0].plot(Fa[:,particle])
    ax[0].set_xlabel("saved step")
    ax[0].set_ylabel("$F_a$")


    ax[1].plot(q[:,particle])
    ax[1].set_xlabel("saved step")
    ax[1].set_ylabel("$q$")


    plt.show()







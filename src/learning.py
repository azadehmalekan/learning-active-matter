import numpy as np
from numba import njit


def initialize_learning(N):
    """
    Initialize active force and quality score.
    """

    Fa = np.random.rand(N)
    q = np.random.rand(N)

    return Fa, q


@njit
def social_learning_numba(
    Fa,
    q,
    neighbors,
    alpha,
    dt
):
    """
    Numba-optimized social learning
    between neighboring particles.
    """

    for k in range(len(neighbors)):

        i = neighbors[k, 0]
        j = neighbors[k, 1]

        if q[i] > q[j]:

            Fa[j] += (
                alpha
                * (Fa[i] - Fa[j])
                * dt
            )

            q[j] += (
                alpha
                * (q[i] - q[j])
                * dt
            )

        elif q[j] > q[i]:

            Fa[i] += (
                alpha
                * (Fa[j] - Fa[i])
                * dt
            )

            q[i] += (
                alpha
                * (q[j] - q[i])
                * dt
            )

    return Fa, q


def environmental_learning(
    x,
    q,
    L_x,
    alpha_q,
    dt,
    q_d,
    q_l
):
    """
    Update score according to environment.
    """

    light_x = np.abs(L_x) / 4

    q_goal = np.where(
        np.abs(x) < light_x,
        q_l,
        q_d
    )

    q += (
        alpha_q
        * (q_goal - q)
        * dt
    )

    return q
import numpy as np

from src.initialization import initialize_particles

from src.forces import (
    compute_forces_numba,
    get_neighbors_cell
)

from src.learning import (
    initialize_learning,
    social_learning_numba,
    environmental_learning
)


def run_simulation(
    N,
    T,
    dt,
    L_x,
    L_y,
    sigma,
    epsilon,
    mu,
    Dr,
    alpha,
    alpha_q,
    q_d,
    q_l,
    save_interval
):

    # -------------------------
    # Initialize particles
    # -------------------------

    x = np.zeros((T, N))
    y = np.zeros((T, N))
    theta = np.zeros((T, N))

    rc = 2**(1/6) * sigma


    x[0], y[0], theta[0] = initialize_particles(
        N,
        L_x,
        L_y,
        sigma
    )


    # -------------------------
    # Initialize learning
    # -------------------------

    Fa, q = initialize_learning(N)


    # -------------------------
    # Storage
    # -------------------------

    n_save = T // save_interval + 1

    x_history = np.zeros((n_save, N))
    y_history = np.zeros((n_save, N))
    theta_history = np.zeros((n_save, N))

    Fa_history = np.zeros((n_save, N))
    q_history = np.zeros((n_save, N))

    save_id = 0


    # -------------------------
    # Save initial state
    # -------------------------

    x_history[save_id] = x[0]
    y_history[save_id] = y[0]
    theta_history[save_id] = theta[0]

    Fa_history[save_id] = Fa
    q_history[save_id] = q

    save_id += 1


    # -------------------------
    # Verlet neighbor update
    # -------------------------

    neighbor_update = 50

    neighbors = None


    # -------------------------
    # Main simulation loop
    # -------------------------

    for t in range(T - 1):


        # ---------------------
        # Update neighbor list
        # ---------------------

        if t % neighbor_update == 0:

            neighbors = get_neighbors_cell(
                x[t],
                y[t],
                L_x,
                L_y,
                rc
            )

            # Convert Python list to NumPy array
            # for Numba

          
            neighbors = np.asarray(
                neighbors,
                dtype=np.int64).reshape(-1, 2)


        # ---------------------
        # Particle forces
        # ---------------------

        Fx, Fy = compute_forces_numba(
            x[t],
            y[t],
            L_x,
            L_y,
            epsilon,
            sigma,
            neighbors
        )


        # ---------------------
        # Social learning
        # ---------------------

        Fa, q = social_learning_numba(
            Fa,
            q,
            neighbors,
            alpha,
            dt
        )


        # ---------------------
        # Environmental learning
        # ---------------------

        q = environmental_learning(
            x[t],
            q,
            L_x,
            alpha_q,
            dt,
            q_d,
            q_l
        )


        # ---------------------
        # Update position
        # ---------------------

        x[t + 1] = (
            x[t]
            +
            (
                Fa * np.cos(theta[t])
                +
                mu * Fx
            ) * dt
        )


        y[t + 1] = (
            y[t]
            +
            (
                Fa * np.sin(theta[t])
                +
                mu * Fy
            ) * dt
        )


        # ---------------------
        # Periodic boundary
        # ---------------------

        x[t + 1] = (
            (x[t + 1] + L_x / 2) % L_x
            -
            L_x / 2
        )

        y[t + 1] = (
            (y[t + 1] + L_y / 2) % L_y
            -
            L_y / 2
        )


        # ---------------------
        # Update orientation
        # ---------------------

        theta[t + 1] = (
            theta[t]
            +
            np.sqrt(2 * Dr * dt)
            * np.random.randn(N)
        )


        # ---------------------
        # Save data
        # ---------------------

        if (t + 1) % save_interval == 0:

            x_history[save_id] = x[t + 1]
            y_history[save_id] = y[t + 1]
            theta_history[save_id] = theta[t + 1]

            Fa_history[save_id] = Fa
            q_history[save_id] = q

            save_id += 1


    # -------------------------
    # Final save if needed
    # -------------------------

    if save_id < n_save:

        x_history[save_id] = x[-1]
        y_history[save_id] = y[-1]
        theta_history[save_id] = theta[-1]

        Fa_history[save_id] = Fa
        q_history[save_id] = q


    return (
        x_history,
        y_history,
        theta_history,
        Fa_history,
        q_history
    )


def run_multiple_simulations(
    n_runs,
    **kwargs
):

    results = []


    for i in range(n_runs):

        print(
            f"Running simulation {i + 1}/{n_runs}"
        )


        result = run_simulation(
            **kwargs
        )


        results.append(result)


    print("All simulations finished")


    return results
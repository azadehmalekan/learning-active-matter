import numpy as np


def mean_active_force(results):
    """
    Mean active force over particles and runs.
    """

    mean_Fa_runs = []

    for run in results:

        Fa_hist = run[3]

        # average over particles
        mean_Fa_t = np.mean(
            Fa_hist,
            axis=1
        )

        mean_Fa_runs.append(mean_Fa_t)


    mean_Fa_runs = np.array(mean_Fa_runs)


    # average over runs
    mean_Fa_all = np.mean(
        mean_Fa_runs,
        axis=0
    )

    std_Fa_all = np.std(
        mean_Fa_runs,
        axis=0
    )


    return mean_Fa_all, std_Fa_all



def mean_score(results):
    """
    Mean score over particles and runs.
    """

    mean_q_runs = []

    for run in results:

        q_hist = run[4]

        # average over particles
        mean_q_t = np.mean(
            q_hist,
            axis=1
        )

        mean_q_runs.append(mean_q_t)


    mean_q_runs = np.array(mean_q_runs)


    mean_q_all = np.mean(
        mean_q_runs,
        axis=0
    )

    std_q_all = np.std(
        mean_q_runs,
        axis=0
    )


    return mean_q_all, std_q_all



def variance_active_force(results):
    """
    Variance of active force over particles,
    averaged over independent runs.
    """

    var_runs = []

    for run in results:

        Fa_hist = run[3]

        # variance over particles at each time
        var_t = np.var(
            Fa_hist,
            axis=1
        )

        var_runs.append(var_t)


    var_runs = np.array(var_runs)


    variance_Fa_all = np.mean(
        var_runs,
        axis=0
    )

    std_variance_Fa = np.std(
        var_runs,
        axis=0
    )


    return variance_Fa_all, std_variance_Fa

def variance_score(results):
    """
    Variance of score over particles,
    averaged over independent runs.
    """

    var_q_runs = []

    for run in results:

        q_hist = run[4]

        # variance over particles at each time
        var_t = np.var(
            q_hist,
            axis=1
        )

        var_q_runs.append(var_t)


    var_q_runs = np.array(var_q_runs)


    variance_q_all = np.mean(
        var_q_runs,
        axis=0
    )

    std_variance_q = np.std(
        var_q_runs,
        axis=0
    )


    return variance_q_all, std_variance_q

def density_profile_x(results, L_x, bins=50):
    """
    Density profile along x direction.
    """

    density_runs = []

    for run in results:

        x_history = run[0]

        density_time = []

        for x in x_history:

            hist, edges = np.histogram(
                x,
                bins=bins,
                range=(-L_x/2, L_x/2),
                density=True
            )

            density_time.append(hist)


        density_runs.append(density_time)


    return np.array(density_runs)
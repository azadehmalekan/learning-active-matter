import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.patches import Rectangle



def animate_particles_q_light(
        x_history,
        y_history,
        q_history,
        L_x,
        L_y,
        dt,
        save_interval,
        interval=100
):

    fig, ax = plt.subplots(figsize=(6,4))


    # light region

    light_width = L_x/2

    light = plt.Rectangle(
        (-light_width/2, -L_y/2),
        light_width,
        L_y,
        alpha=0.2
    )

    ax.add_patch(light)


    scat = ax.scatter(
        [],
        [],
        c=[],
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=20
    )


    time_text = ax.text(
        0.05,
        0.95,
        "",
        transform=ax.transAxes
    )


    ax.set_xlim(-L_x/2, L_x/2)
    ax.set_ylim(-L_y/2, L_y/2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect("equal")


    cbar = plt.colorbar(
        scat,
        ax=ax
    )

    cbar.set_label("q score")


    def update(frame):

        positions = np.column_stack(
            (
                x_history[frame],
                y_history[frame]
            )
        )


        scat.set_offsets(
            positions
        )


        scat.set_array(
            q_history[frame]
        )


        sim_time = (
            frame
            *
            save_interval
            *
            dt
        )


        time_text.set_text(
            f"t = {sim_time:.2f}"
        )


        return scat, time_text


    ani = FuncAnimation(
        fig,
        update,
        frames=len(x_history),
        interval=interval,
        blit=True
    )


    return ani


def animate_particles_Fa(
        x_history,
        y_history,
        Fa_history,
        L_x,
        L_y,
        dt,
        save_interval,
        interval=100
):

    fig, ax = plt.subplots(figsize=(6,4))


    scat = ax.scatter(
        [],
        [],
        c=[],
        cmap="plasma",
        s=20
    )


    time_text = ax.text(
        0.05,
        0.95,
        "",
        transform=ax.transAxes
    )


    ax.set_xlim(-L_x/2, L_x/2)
    ax.set_ylim(-L_y/2, L_y/2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect("equal")


    cbar = plt.colorbar(
        scat,
        ax=ax
    )

    cbar.set_label("$F_a$")


    def update(frame):

        positions = np.column_stack(
            (
                x_history[frame],
                y_history[frame]
            )
        )


        scat.set_offsets(
            positions
        )


        # color by active force

        scat.set_array(
            Fa_history[frame]
        )


        sim_time = (
            frame
            *
            save_interval
            *
            dt
        )


        time_text.set_text(
            f"t = {sim_time:.2f}"
        )


        return scat, time_text



    ani = FuncAnimation(
        fig,
        update,
        frames=len(x_history),
        interval=interval,
        blit=True
    )


    return ani  




def animate_particles_q_Fa_light(
        x_history,
        y_history,
        q_history,
        Fa_history,
        L_x,
        L_y,
        dt,
        save_interval,
        interval=100
):

    fig, ax = plt.subplots(figsize=(6,4))


    # -------------------------
    # Light region
    # -------------------------

    light_width = L_x/2

    light = plt.Rectangle(
        (-light_width/2, -L_y/2),
        light_width,
        L_y,
        alpha=0.2
    )

    ax.add_patch(light)


    # -------------------------
    # Initial scatter
    # -------------------------

    scat = ax.scatter(
        [],
        [],
        c=[],
        s=[],
        cmap="viridis",
        vmin=0,
        vmax=1
    )


    time_text = ax.text(
        0.05,
        0.95,
        "",
        transform=ax.transAxes
    )


    ax.set_xlim(-L_x/2, L_x/2)
    ax.set_ylim(-L_y/2, L_y/2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect("equal")


    cbar = plt.colorbar(
        scat,
        ax=ax
    )

    cbar.set_label("q score")


    # -------------------------
    # Animation update
    # -------------------------

    def update(frame):

        positions = np.column_stack(
            (
                x_history[frame],
                y_history[frame]
            )
        )


        scat.set_offsets(
            positions
        )


        # color = q

        scat.set_array(
            q_history[frame]
        )


        # size = Fa

        sizes = (
            20
            +
            80 *
            (Fa_history[frame] /
             np.max(Fa_history))
        )


        scat.set_sizes(
            sizes
        )


        sim_time = (
            frame
            *
            save_interval
            *
            dt
        )


        time_text.set_text(
            f"t = {sim_time:.2f}"
        )


        return scat, time_text



    ani = FuncAnimation(
        fig,
        update,
        frames=len(x_history),
        interval=interval,
        blit=True
    )


    return ani





def animate_all_runs_concatenated(
        results,
        L_x,
        L_y,
        dt,
        save_interval,
        color_by="q",
        interval=100
):

    fig, ax = plt.subplots(figsize=(6,4))

    # light region
    light_width = L_x / 2
    light = Rectangle(
        (-light_width/2, -L_y/2),
        light_width,
        L_y,
        alpha=0.2)
    ax.add_patch(light)


    scat = ax.scatter([], [], s=8)

    time_text = ax.text(
        0.05,
        0.95,
        "",
        transform=ax.transAxes
    )


    ax.set_xlim(-L_x/2, L_x/2)
    ax.set_ylim(-L_y/2, L_y/2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_aspect("equal")


    n_frames = results[0][0].shape[0]


    def update(frame):

        x_all = []
        y_all = []
        q_all = []
        Fa_all = []


        for run in results:

            x_hist = run[0]
            y_hist = run[1]
            Fa_hist = run[3]
            q_hist = run[4]


            x_all.append(
                x_hist[frame]
            )

            y_all.append(
                y_hist[frame]
            )

            q_all.append(
                q_hist[frame]
            )

            Fa_all.append(
                Fa_hist[frame]
            )


        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)

        q_all = np.concatenate(q_all)
        Fa_all = np.concatenate(Fa_all)


        scat.set_offsets(
            np.column_stack(
                (x_all,y_all)
            )
        )


        if color_by=="q":

            scat.set_array(q_all)


        elif color_by=="Fa":

            scat.set_array(Fa_all)


        else:

            scat.set_array(None)


        sim_time = frame * save_interval * dt

        time_text.set_text(
            f"t = {sim_time:.2f}"
        )


        return scat, time_text



    ani = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=False
    )


    return ani
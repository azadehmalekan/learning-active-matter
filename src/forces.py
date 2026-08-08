import numpy as np
from numba import njit


def get_neighbors_bruteforce(x, y, L_x, L_y, rc):
    """
    Find particle pairs within interaction distance.
    """

    N = len(x)

    neighbors = []

    rc2 = rc * rc

    for i in range(N):

        for j in range(i + 1, N):

            dx = x[i] - x[j]
            dy = y[i] - y[j]

            # Minimum image convention
            dx -= L_x * np.round(dx / L_x)
            dy -= L_y * np.round(dy / L_y)

            r2 = dx * dx + dy * dy

            if r2 < rc2 and r2 > 1e-16:

                neighbors.append((i, j))

    return neighbors


def build_cell_list(x, y, L_x, L_y, cell_size):

    N_cells_x = int(L_x / cell_size)
    N_cells_y = int(L_y / cell_size)

    cells = [
        [
            []
            for j in range(N_cells_y)
        ]
        for i in range(N_cells_x)
    ]

    x_min = -L_x / 2
    y_min = -L_y / 2

    for i in range(len(x)):

        cell_x = int((x[i] - x_min) / cell_size)
        cell_y = int((y[i] - y_min) / cell_size)

        cell_x %= N_cells_x
        cell_y %= N_cells_y

        cells[cell_x][cell_y].append(i)

    return cells, N_cells_x, N_cells_y


def get_neighbor_cells(
        cell_x,
        cell_y,
        N_cells_x,
        N_cells_y
):

    cells = []

    for dx in [-1, 0, 1]:

        for dy in [-1, 0, 1]:

            nx = (cell_x + dx) % N_cells_x
            ny = (cell_y + dy) % N_cells_y

            cells.append((nx, ny))

    return cells


def get_neighbors_cell(x, y, L_x, L_y, rc):

    cells, N_cells_x, N_cells_y = build_cell_list(
        x,
        y,
        L_x,
        L_y,
        rc
    )

    neighbors = []

    checked = set()

    for cx in range(N_cells_x):

        for cy in range(N_cells_y):

            for i in cells[cx][cy]:

                near_cells = get_neighbor_cells(
                    cx,
                    cy,
                    N_cells_x,
                    N_cells_y
                )

                for nx, ny in near_cells:

                    for j in cells[nx][ny]:

                        if i >= j:
                            continue

                        pair = (i, j)

                        if pair in checked:
                            continue

                        checked.add(pair)

                        dx = x[i] - x[j]
                        dy = y[i] - y[j]

                        dx -= L_x * np.round(dx / L_x)
                        dy -= L_y * np.round(dy / L_y)

                        r2 = dx * dx + dy * dy

                        if r2 < rc * rc:

                            neighbors.append(pair)

    return neighbors


# ============================================================
# NUMBA FORCE CALCULATION
# ============================================================

@njit
def compute_forces_numba(
    x,
    y,
    L_x,
    L_y,
    epsilon,
    sigma,
    neighbors
):

    N = len(x)

    Fx = np.zeros(N)
    Fy = np.zeros(N)

    rc = 2.0 ** (1.0 / 6.0) * sigma
    rc2 = rc * rc

    for k in range(len(neighbors)):

        i = neighbors[k, 0]
        j = neighbors[k, 1]

        dx = x[i] - x[j]
        dy = y[i] - y[j]

        # Minimum image convention

        dx -= L_x * np.round(dx / L_x)
        dy -= L_y * np.round(dy / L_y)

        r2 = dx * dx + dy * dy

        if r2 < rc2 and r2 > 1e-16:

            r = np.sqrt(r2)

            sr = sigma / r

            sr6 = sr ** 6
            sr12 = sr6 * sr6

            F = (
                24.0
                * epsilon
                / r
                * (2.0 * sr12 - sr6)
            )

            fx = F * dx / r
            fy = F * dy / r

            Fx[i] += fx
            Fy[i] += fy

            Fx[j] -= fx
            Fy[j] -= fy

    return Fx, Fy
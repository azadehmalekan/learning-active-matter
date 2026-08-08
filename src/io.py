import numpy as np
import os


def save_multiple_runs(results, folder="data"):

    os.makedirs(folder, exist_ok=True)

    for i, result in enumerate(results):

        x_history, y_history, theta_history, Fa_history, q_history = result

        filename = os.path.join(
            folder,
            f"run_{i}.npz"
        )

        np.savez(
            filename,
            x=x_history,
            y=y_history,
            theta=theta_history,
            Fa=Fa_history,
            q=q_history
        )

        print(f"Saved {filename}")



def load_multiple_runs(folder="data"):

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Folder {folder} does not exist"
        )

    results = []

    files = sorted(
        [
            f for f in os.listdir(folder)
            if f.startswith("run_") and f.endswith(".npz")
        ]
    )


    for file in files:

        path = os.path.join(folder, file)

        data = np.load(path)

        results.append(
            (
                data["x"],
                data["y"],
                data["theta"],
                data["Fa"],
                data["q"]
            )
        )

        print(f"Loaded {file}")


    return results
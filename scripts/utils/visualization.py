import os
import numpy as np
import matplotlib.pyplot as plt


def plot_2d_sim(path, obstacles, goal, title="2D CLF-CBF", save_path=None):
    plt.figure(figsize=(8, 8))
    for pos, r in obstacles:
        plt.gca().add_patch(plt.Circle(pos, r, color="r", alpha=0.3))
    plt.plot(path[:, 0], path[:, 1], "b-", linewidth=2, label="Trajectory")
    plt.scatter(*goal, c="g", marker="*", s=200, label="Goal")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(os.path.join(save_path, "2d_plot.png"))
    plt.show()


def plot_3d_sim(path, obstacles, goal, title="3D Drone ECBF", save_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for pos, r in obstacles:
        u, v = np.mgrid[0 : 2 * np.pi : 10j, 0 : np.pi : 10j]
        ax.plot_wireframe(
            pos[0] + r * np.cos(u) * np.sin(v),
            pos[1] + r * np.sin(u) * np.sin(v),
            pos[2] + r * np.cos(v),
            color="r",
            alpha=0.04,
        )
    ax.plot(path[:, 0], path[:, 1], path[:, 2], "b-", linewidth=2)
    ax.scatter(*goal, color="g", s=200, marker="*", label="Goal")
    ax.set_title(title)
    if save_path:
        plt.savefig(os.path.join(save_path, "3d_plot.png"))
    plt.show()

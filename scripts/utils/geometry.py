import numpy as np


def as3(x):
    """
    Ensure x is a 3D float vector (shape (3,)).
    If 2D (shape (2,)), append 0.0.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 2:
        return np.array([x[0], x[1], 0.0], dtype=float)
    if x.size >= 3:
        return x[:3]
    raise ValueError(f"Input vector must be at least 2D (got size {x.size})")


def get_tangent_2d(rel_pos, u_nom):
    tangent = np.array([-rel_pos[1], rel_pos[0]])
    tangent /= np.linalg.norm(tangent) + 1e-6
    if np.dot(tangent, u_nom) < 0:
        tangent = -tangent
    return tangent


def project_to_tangent_plane_3d(rel_pos, u_nom):
    dist = np.linalg.norm(rel_pos)
    normal = rel_pos / (dist + 1e-6)
    u_tangent = u_nom - np.dot(u_nom, normal) * normal
    if np.linalg.norm(u_tangent) < 1e-3:
        u_tangent = np.array([normal[1], -normal[0], 0.1])
    return u_tangent / (np.linalg.norm(u_tangent) + 1e-6)

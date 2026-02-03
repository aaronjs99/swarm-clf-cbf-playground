import numpy as np


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

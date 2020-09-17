#!/usr/bin/env python
# coding: utf-8

from enum import Enum
import numpy as np


N_POINTS = 1000
N_OBJ = 100

np.random.seed(seed=3121989668)


class NoiseLevel(Enum):
    NoNoise = "NoNoise"
    LowNoise = "LowNoise"
    MedNoise = "MedNoise"
    HighNoise = "HighNoise"


def spherical_to_euclidean(pts):
    r"""Converts a set of coordinates from spherical to cartesian coordinates.

    Parameters
    ----------
    pts: numpy.array
        List of points in spherical coordinates. Matrix dimension should be
        Nx3, where N is the number of points.

    Returns
    -------
    numpy.array
        A numpy array of size Nx3 with the euclidean coordinates of the input
        points.
    """
    euc_pts = np.zeros_like(pts)
    sin_theta, cos_theta = np.sin(pts[:, 1]), np.cos(pts[:, 1])
    sin_phi, cos_phi = np.sin(pts[:, 2]), np.cos(pts[:, 2])

    for i in range(pts.shape[0]):
        r = pts[i, 0]
        euc_pts[i, 0] = r * sin_theta[i] * cos_phi[i]
        euc_pts[i, 1] = r * sin_theta[i] * sin_phi[i]
        euc_pts[i, 2] = r * cos_theta[i]

    return euc_pts


def gen_sphere(n_points, noise_lvl):
    radius = np.array([1] * n_points)
    theta = np.random.rand(n_points) * np.pi
    phi = np.random.rand(n_points) * 2 * np.pi

    sph_pts = np.array([radius, theta, phi]).T
    if noise_lvl != NoiseLevel.NoNoise:
        pass
    euc_pts = spherical_to_euclidean(sph_pts)
    return euc_pts


for i in range(N_OBJ):
    pts = gen_sphere(N_POINTS, NoiseLevel.NoNoise)
    normals = pts
    sdf = np.array([0] * N_POINTS)

    fname = f"{i+1}."
    np.savetxt(fname + "xyz", pts)
    np.savetxt(fname + "normals", normals)
    np.savetxt(fname + "sdf", sdf)


TRAIN_FRAC = 0.7
VAL_FRAC = 0.1

all_idx = np.array(range(N_OBJ))

train_idx = np.random.choice(all_idx,
                             size=int(np.floor(TRAIN_FRAC * (N_OBJ))),
                             replace=False) + 1
remaining_idx = np.array(list(set(all_idx) - set(train_idx)))

val_idx = np.random.choice(remaining_idx,
                           size=int(np.floor(VAL_FRAC * (N_OBJ))),
                           replace=False) + 1

test_idx = np.array(list(set(remaining_idx) - set(val_idx))) + 1

np.savetxt("train_nonoise.txt", train_idx, fmt="%d")
np.savetxt("val_nonoise.txt", val_idx, fmt="%d")
np.savetxt("test_nonoise.txt", test_idx, fmt="%d")

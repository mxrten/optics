# MIT License
#
# Copyright 2024 Mårten Selin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


def project_at_angle(image, center, length, width, angle=0, degrees=False, supersampling=4):
    """Make a 1D line-projection of a (2D) image.

    Project image-pixels onto bins at a tilted axis. At each bin the
    average is computed.

    Parameters
    ----------
    image : ndarray
        The image to process.
    center : (float, float)
        The origin to use in the projection.
    length : float
        Length of projection.
    width : float
        Width of projection.
    angle : scalar
        Projection angle (0 for projection onto horizontal axis).
    degrees : boolean, optional
        Whether angle is given in degrees rather than radians.
    supersampling : int

    Returns
    -------
    bins : ndarray
        The bins' (center) position along the tilted axis.
    projection : ndarray
        The line-projected average for every bin.
    """

    # Sanitize input
    image = np.asarray(image)
    angle = np.deg2rad(angle) if degrees else angle
    cx = center[0] if center is not None else (image.shape[1] - 1) / 2
    cy = center[1] if center is not None else (image.shape[0] - 1) / 2

    # Coordinates, with origin in image center
    x = np.arange(image.shape[1]) - cx
    y = np.arange(image.shape[0]) - cy
    X, Y = np.meshgrid(x, y)

    # Longitudinal and transverse position
    l = -np.cos(angle) * X - np.sin(angle) * Y
    t = -np.sin(angle) * X + np.cos(angle) * Y

    # Indices, rounded and fractional
    num_bins = int(width * supersampling) + 1
    bins = (np.arange(num_bins) - (num_bins - 1) / 2) / supersampling
    projection = np.zeros_like(bins)

    ti = np.round((t + width/2) * supersampling).astype(int)
    if length is not None:
        ti[(l < -length / 2) | (length / 2 < l)] = -1

    for i in range(num_bins):
        vals = image[ti == i]
        projection[i] = np.mean(vals) if len(vals) > 0 else np.nan

    return bins, projection


def repair_profile(profile, ensure_positive_flank=False):
    """Repair an edge-profile in case of NaN-values"""
    x = np.arange(len(profile))
    mid = len(profile) // 2
    profile = np.interp(x, x[np.isfinite(profile)], profile[np.isfinite(profile)])
    if ensure_positive_flank and np.mean(profile[:mid]) > np.mean(profile[mid:]):
        profile = -profile
    return profile


def refine_edge(image, edge_points):
    """Improve edge-guess by doing local edge-detect"""
    # Transversal direction
    t_direction = [[0, 1], [-1, 0]] @ np.diff(edge_points, axis=0)[0]
    t_direction = t_direction / np.linalg.norm(t_direction)
    
    # Refine each edge point by taking a transverse profile and find flank-middle
    refined_edge = []
    angle = np.arctan2(np.diff(edge_points[:, 1]), np.diff(edge_points[:, 0]))
    width = min(20, np.linalg.norm(np.diff(edge_points, axis=0)[0]))
    for p in edge_points:
        bins, profile = project_at_angle(image, p, 5, width, angle, supersampling=4)
        repaired_profile = repair_profile(profile, ensure_positive_flank=True)
        profile_average = np.mean(repaired_profile)
        idx = np.argmin(repaired_profile < profile_average)
        t_offset = np.interp(profile_average, repaired_profile[[idx, idx - 1]], bins[[idx, idx - 1]])
        refined_edge.append(p - t_offset * t_direction)
    return np.array(refined_edge)


if __name__ == '__main__':
    # --------------
    # 0. Load image
    # --------------
    image_file = os.path.join(os.path.dirname(__file__), 'mtf_test.png')
    image = np.array(Image.open(image_file))
    
    # ----------------------------
    # 1. Provide rough edge guess
    # ----------------------------
    edge_guess = np.array([
        [144, 240], # [x1, y1]
        [157, 134]  # [x2, y2]
    ])

    # ---------------
    # 2. Refine edge
    # ---------------
    refined_edge = refine_edge(image, edge_guess)
    edge_center = np.mean(refined_edge, axis=0)
    edge_angle = np.arctan2(np.diff(refined_edge[:, 1])[0], np.diff(refined_edge[:, 0])[0])
    edge_length = np.linalg.norm(refined_edge[1] - refined_edge[0])
    edge_width = 20

    # ------------------------------------
    # 3. Take (supersampled) edge profile
    # ------------------------------------
    bins, profile = project_at_angle(
        image,
        edge_center,
        edge_length,
        edge_width,
        edge_angle,
        supersampling=4,
    )

    # ---------------------------------------------------
    # 4. Repair edge profile (in case of missing values)
    # ---------------------------------------------------
    repaired_profile = repair_profile(profile, ensure_positive_flank=True)
    repaired_profile_for_visualization = repair_profile(profile, ensure_positive_flank=False)

    # --------------------------------------
    # 5. Compensate for background gradient
    # --------------------------------------
    # TODO

    # -----------------
    # 6. Calculate LSF
    # -----------------
    lsf = np.column_stack([
        np.convolve(bins, [0.5, 0.5], 'valid'),
        np.diff(repaired_profile),
    ])

    # -----------------
    # 7. Calculate MTF
    # -----------------
    f = np.arange(len(lsf)) / (lsf[-1, 0] - lsf[0, 0])
    mtf = np.real(np.fft.fft(np.fft.ifftshift(lsf[:, 1])))
    mtf = np.column_stack([f, mtf / mtf[0]])

    # -------------
    # 8. Visualize
    # -------------
    plt.figure(figsize=(12, 7))
    plt.subplot(position=[0.05, 0.08, 0.50, 0.86])
    plt.title('Image, edge & ROI')
    plt.imshow(image, cmap='gray')
    plt.gca().add_patch(patches.Rectangle(
        (edge_center[0] - edge_length/2, edge_center[1] - edge_width/2),
        edge_length,
        edge_width,
        angle=np.rad2deg(edge_angle),
        rotation_point='center',
        facecolor='r',
        alpha=0.25,
    ))
    plt.plot(edge_guess[:, 0], edge_guess[:, 1], 'b+')
    plt.plot(refined_edge[:, 0], refined_edge[:, 1], 'rx-')
    plt.subplot(position=[0.60, 0.58, 0.35, 0.36])
    plt.title('Edge profile')
    plt.plot(bins, profile, 'k.')
    plt.plot(bins, repaired_profile_for_visualization, 'r')
    plt.xlabel('x [px]')
    plt.subplot(position=[0.60, 0.08, 0.35, 0.36])
    plt.title('MTF')
    plt.plot([0, 1], [0, 0], 'k--', mtf[:, 0], mtf[:, 1], 'b')
    plt.xlim(0, 1.0)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('f [cy/px]')
    plt.show()

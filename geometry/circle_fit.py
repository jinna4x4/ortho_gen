"""
Fit a circle through 3 points in 3D space.
"""
import numpy as np


def fit_circle_3d(p1, p2, p3):
    """
    Fits a circle to 3 points in 3D space.

    Args:
        p1, p2, p3: Points as array-like (x, y, z)

    Returns:
        tuple: (center, radius, normal, basis_u, basis_w) or None if collinear
            - center: 3D center point of the circle
            - radius: radius of the circle (major radius for torus)
            - normal: unit normal vector to the plane
            - basis_u: unit vector from center to p1
            - basis_w: unit vector perpendicular to normal and basis_u
    """
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    p3 = np.array(p3, dtype=np.float64)

    v1 = p2 - p1
    v2 = p3 - p1

    # Normal to the plane defined by p1, p2, p3
    normal = np.cross(v1, v2)
    norm_mag = np.linalg.norm(normal)

    if norm_mag < 1e-10:
        return None  # Collinear points

    normal = normal / norm_mag

    # Create a local coordinate system on the plane
    # u = v1 normalized
    u = v1 / np.linalg.norm(v1)
    # w = normal x u (perpendicular to both)
    w = np.cross(normal, u)

    # Project points to 2D local plane (origin at p1)
    # p1_2d = (0, 0)
    p2_2d = np.array([np.dot(p2 - p1, u), np.dot(p2 - p1, w)])
    p3_2d = np.array([np.dot(p3 - p1, u), np.dot(p3 - p1, w)])

    # Solve for circumcenter in 2D
    # Using the formula: solve the linear system for center (cx, cy)
    # 2*x2*cx + 2*y2*cy = x2^2 + y2^2
    # 2*x3*cx + 2*y3*cy = x3^2 + y3^2

    A = np.array([
        [2 * p2_2d[0], 2 * p2_2d[1]],
        [2 * p3_2d[0], 2 * p3_2d[1]]
    ])
    b = np.array([
        p2_2d[0]**2 + p2_2d[1]**2,
        p3_2d[0]**2 + p3_2d[1]**2
    ])

    try:
        center_2d = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    # Radius
    radius = np.linalg.norm(center_2d)

    # Transform center back to 3D
    center = p1 + center_2d[0] * u + center_2d[1] * w

    # Compute basis vectors from center
    basis_u = (p1 - center)
    basis_u = basis_u / np.linalg.norm(basis_u)
    basis_w = np.cross(normal, basis_u)

    return center, radius, normal, basis_u, basis_w

"""
Generate a torus segment mesh that sweeps through 3 points.
"""
import numpy as np
import trimesh


def get_angle(p, center, basis_u, basis_w):
    """
    Returns angle of point p relative to center in basis (u, w) in [0, 2pi).
    """
    vec = p - center
    x = np.dot(vec, basis_u)
    y = np.dot(vec, basis_w)
    return np.arctan2(y, x) % (2 * np.pi)


def generate_torus_segment(center, normal, major_radius, minor_radius,
                           p1, p2, p3, resolution=64):
    """
    Generates a torus segment mesh passing through p1 -> p2 -> p3.

    Args:
        center: 3D center point of the torus
        normal: unit normal vector to the torus plane
        major_radius: radius from center to tube center
        minor_radius: radius of the tube
        p1, p2, p3: the 3 points defining the arc
        resolution: mesh resolution

    Returns:
        trimesh.Trimesh: the torus segment mesh (watertight with end caps)
    """
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    p3 = np.array(p3, dtype=np.float64)
    center = np.array(center, dtype=np.float64)
    normal = np.array(normal, dtype=np.float64)

    # Establish basis vectors
    u = (p1 - center)
    u = u / np.linalg.norm(u)
    w = np.cross(normal, u)

    # Calculate angles for p2 and p3 relative to p1 (theta1 = 0)
    theta2 = get_angle(p2, center, u, w)
    theta3 = get_angle(p3, center, u, w)

    # Determine sweep: we want to go from 0 to theta3, passing through theta2
    # If theta2 > theta3, we need to wrap around
    if theta2 > theta3:
        # theta3 is "before" theta2 going CCW, so add 2pi to theta3
        theta3 += 2 * np.pi

    total_sweep = theta3

    # Generate mesh
    n_major = max(int(resolution * (total_sweep / (2 * np.pi))), 8) + 1
    n_minor = max(resolution // 2, 16)

    theta = np.linspace(0, total_sweep, n_major)
    phi = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)

    # Grid of angles
    Theta, Phi = np.meshgrid(theta, phi)

    # Parametric torus in local coordinates
    # Radial distance from center in the major plane
    rc = major_radius + minor_radius * np.cos(Phi)

    # Components along basis vectors
    comp_u = rc * np.cos(Theta)
    comp_w = rc * np.sin(Theta)
    comp_n = minor_radius * np.sin(Phi)

    # Construct vertices in 3D
    vx = center[0] + comp_u * u[0] + comp_w * w[0] + comp_n * normal[0]
    vy = center[1] + comp_u * u[1] + comp_w * w[1] + comp_n * normal[1]
    vz = center[2] + comp_u * u[2] + comp_w * w[2] + comp_n * normal[2]

    vertices = np.stack([vx.ravel(), vy.ravel(), vz.ravel()], axis=1)

    # Generate faces (grid topology)
    faces = []
    for i in range(n_minor):
        for j in range(n_major - 1):
            # Indices (row-major: i * n_major + j)
            r0 = i
            r1 = (i + 1) % n_minor
            c0 = j
            c1 = j + 1

            idx00 = r0 * n_major + c0
            idx10 = r1 * n_major + c0
            idx01 = r0 * n_major + c1
            idx11 = r1 * n_major + c1

            # Two triangles per quad
            faces.append([idx00, idx01, idx11])
            faces.append([idx00, idx11, idx10])

    faces = np.array(faces)

    # Add end caps to make watertight
    n_verts = len(vertices)

    # Start cap (at theta=0)
    start_center = center + major_radius * u
    vertices = np.vstack([vertices, start_center])
    start_center_idx = n_verts

    start_cap_faces = []
    for i in range(n_minor):
        i_next = (i + 1) % n_minor
        idx0 = i * n_major  # vertex at theta=0, phi=i
        idx1 = i_next * n_major
        # Triangle: center -> idx1 -> idx0 (CCW when viewed from outside)
        start_cap_faces.append([start_center_idx, idx1, idx0])

    # End cap (at theta=total_sweep)
    end_u = np.cos(total_sweep) * u + np.sin(total_sweep) * w
    end_center = center + major_radius * end_u
    vertices = np.vstack([vertices, end_center])
    end_center_idx = n_verts + 1

    end_cap_faces = []
    for i in range(n_minor):
        i_next = (i + 1) % n_minor
        idx0 = i * n_major + (n_major - 1)  # vertex at theta=end, phi=i
        idx1 = i_next * n_major + (n_major - 1)
        # Triangle: center -> idx0 -> idx1 (CCW when viewed from outside)
        end_cap_faces.append([end_center_idx, idx0, idx1])

    all_faces = np.vstack([faces, start_cap_faces, end_cap_faces])

    mesh = trimesh.Trimesh(vertices=vertices, faces=all_faces)
    mesh.fix_normals()

    return mesh

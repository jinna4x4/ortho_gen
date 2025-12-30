"""
Boolean mesh operations using trimesh.
"""
import trimesh


def subtract_meshes(target_mesh, tool_mesh):
    """
    Subtracts tool_mesh from target_mesh.

    Args:
        target_mesh: trimesh.Trimesh object (the original model)
        tool_mesh: trimesh.Trimesh object (the shape to subtract)

    Returns:
        trimesh.Trimesh: the result, or None if operation failed
    """
    # Try to ensure meshes are watertight
    if not tool_mesh.is_watertight:
        tool_mesh.fill_holes()

    if not target_mesh.is_watertight:
        target_mesh.fill_holes()

    # Try different boolean engines in order of preference
    engines = ['manifold', 'blender', 'scad']

    for engine in engines:
        try:
            result = trimesh.boolean.difference(
                [target_mesh, tool_mesh],
                engine=engine
            )
            if result is not None and not result.is_empty:
                return result
        except Exception as e:
            print(f"Boolean engine '{engine}' failed: {e}")
            continue

    # If all engines fail, return None
    print("All boolean engines failed")
    return None


def check_mesh_validity(mesh):
    """
    Check if a mesh is suitable for boolean operations.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        tuple: (is_valid, message)
    """
    issues = []

    if not mesh.is_watertight:
        issues.append("Mesh is not watertight (has holes)")

    if not mesh.is_winding_consistent:
        issues.append("Mesh has inconsistent face winding")

    if mesh.is_empty:
        issues.append("Mesh is empty")

    if len(issues) == 0:
        return True, "Mesh is valid for boolean operations"
    else:
        return False, "; ".join(issues)

"""
Boolean mesh operations using trimesh.
"""
import trimesh


def subtract_meshes(target_mesh, tool_mesh):
    """
    Subtracts tool_mesh from target_mesh. Handles both solid volumes and shells.

    Args:
        target_mesh: trimesh.Trimesh object (the original model)
        tool_mesh: trimesh.Trimesh object (the shape to subtract)

    Returns:
        trimesh.Trimesh: the result, or None if operation failed
    """
    import sys
    import pyvista as pv
    import numpy as np
    
    print(f"DEBUG: Starting subtraction. Target: {target_mesh.vertices.shape}, Tool: {tool_mesh.vertices.shape}", file=sys.stderr)
    
    # Check if target is a shell (not watertight)
    if not target_mesh.is_watertight:
        print("DEBUG: Target is a shell. Using PyVista clip_surface...", file=sys.stderr)
        try:
            # Convert to PyVista
            target_pv = pv.wrap(target_mesh)
            tool_pv = pv.wrap(tool_mesh)
            
            # clip_surface with invert=True removes the part of target_pv that is INSIDE tool_pv
            # This is exactly what subtraction from a shell means.
            result_pv = target_pv.clip_surface(tool_pv, invert=True)
            
            # Convert back to trimesh
            # result_pv is usually a PolyData
            result = trimesh.Trimesh(vertices=result_pv.points, faces=result_pv.faces.reshape(-1, 4)[:, 1:])
            
            if not result.is_empty:
                print(f"DEBUG: PyVista clipping success! Result: {result.vertices.shape}", file=sys.stderr)
                return result
            else:
                print("DEBUG: PyVista clipping returned an empty mesh.", file=sys.stderr)
        except Exception as e:
            print(f"DEBUG: PyVista clipping failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    # Try direct manifold3d path for solid-solid subtraction
    try:
        import manifold3d
        from manifold3d import Manifold, Mesh
        print("DEBUG: Trying direct manifold3d interface...", file=sys.stderr)
        
        # Convert target
        target_m = Manifold(Mesh(
            vert_properties=np.array(target_mesh.vertices, dtype=np.float32),
            tri_verts=np.array(target_mesh.faces, dtype=np.uint32)
        ))
        
        # Convert tool
        tool_m = Manifold(Mesh(
            vert_properties=np.array(tool_mesh.vertices, dtype=np.float32),
            tri_verts=np.array(tool_mesh.faces, dtype=np.uint32)
        ))
        
        # Operation
        result_m = target_m - tool_m
        
        # Convert back
        res_mesh_data = result_m.to_mesh()
        result = trimesh.Trimesh(
            vertices=res_mesh_data.vert_properties, 
            faces=res_mesh_data.tri_verts
        )
        
        if not result.is_empty:
             print(f"DEBUG: Direct manifold3d success! Result: {result.vertices.shape}", file=sys.stderr)
             return result
        else:
             print("DEBUG: Direct manifold3d returned an empty mesh.", file=sys.stderr)
             
    except Exception as e:
        print(f"DEBUG: Direct manifold3d failed: {e}", file=sys.stderr)

    # Fallback to trimesh engines for watertight meshes
    if target_mesh.is_watertight and tool_mesh.is_watertight:
        engines = ['manifold', 'blender', 'scad']
        for engine in engines:
            try:
                print(f"DEBUG: Trying engine '{engine}'...", file=sys.stderr)
                result = trimesh.boolean.difference(
                    [target_mesh, tool_mesh],
                    engine=engine
                )
                if result is not None and not result.is_empty:
                    print(f"DEBUG: Engine '{engine}' success! Result: {result.vertices.shape}", file=sys.stderr)
                    return result
            except Exception as e:
                print(f"Boolean engine '{engine}' failed: {e}", file=sys.stderr)
                continue

    # If all engines fail, return None
    print("All boolean operations failed", file=sys.stderr)
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

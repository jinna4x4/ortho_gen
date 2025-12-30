"""
Web-based GUI viewer using Trame and PyVista.
"""
import os
import numpy as np
import pyvista as pv
import trimesh

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify, vtk as vtk_widgets, html

from geometry.circle_fit import fit_circle_3d
from geometry.torus_generator import generate_torus_segment
from operations.boolean_ops import subtract_meshes


class TorusToolApp:
    """Web-based torus subtraction tool."""

    def __init__(self, server=None):
        self.server = server or get_server(client_type="vue3")
        self.state = self.server.state
        self.ctrl = self.server.controller

        # Initialize state
        self.state.trame__title = "STL Torus Subtraction Tool"
        self.state.minor_radius = 2.0
        self.state.status_message = "Enter STL file path and click Load"
        self.state.file_path = ""
        self.state.points_count = 0
        self.state.can_preview = False
        self.state.can_subtract = False
        self.state.can_export = False
        self.state.point1_text = "Point 1: -"
        self.state.point2_text = "Point 2: -"
        self.state.point3_text = "Point 3: -"
        self.state.export_path = "output.stl"

        # Data
        self.source_mesh_pv = None
        self.source_mesh_tri = None
        self.picked_points = []
        self.marker_scale = 1.0

        # PyVista plotter (offscreen for web)
        pv.OFF_SCREEN = True
        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.set_background("slategray")
        self.plotter.add_axes()

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the web UI."""
        with SinglePageLayout(self.server) as layout:
            layout.title.set_text("STL Torus Subtraction Tool")

            with layout.toolbar:
                vuetify.VTextField(
                    v_model=("file_path", ""),
                    label="STL File Path",
                    density="compact",
                    hide_details=True,
                    style="max-width: 400px;",
                    variant="outlined",
                )
                vuetify.VBtn(
                    "Load",
                    click=self._load_file,
                    variant="flat",
                    color="primary",
                    classes="ml-2",
                )
                vuetify.VSpacer()
                vuetify.VTextField(
                    v_model=("export_path", "output.stl"),
                    label="Export Path",
                    density="compact",
                    hide_details=True,
                    style="max-width: 200px;",
                    variant="outlined",
                )
                vuetify.VBtn(
                    "Export",
                    click=self._export_stl,
                    variant="outlined",
                    classes="ml-2",
                    disabled=("!can_export",),
                )

            with layout.content:
                with vuetify.VContainer(fluid=True, classes="fill-height pa-0"):
                    with vuetify.VRow(classes="fill-height", no_gutters=True):
                        # 3D Viewer
                        with vuetify.VCol(cols=9, classes="fill-height"):
                            view = vtk_widgets.VtkRemoteView(
                                self.plotter.render_window,
                                interactive_ratio=1,
                            )
                            self.ctrl.view_update = view.update
                            self.ctrl.view_reset_camera = view.reset_camera

                        # Sidebar
                        with vuetify.VCol(cols=3, classes="pa-4"):
                            vuetify.VAlert(
                                text=("status_message",),
                                type="info",
                                variant="tonal",
                                classes="mb-4"
                            )

                            vuetify.VCard(
                                classes="mb-4",
                                children=[
                                    vuetify.VCardTitle("Selected Points"),
                                    vuetify.VCardText([
                                        html.Div("{{ point1_text }}", classes="text-body-2", style="color: red;"),
                                        html.Div("{{ point2_text }}", classes="text-body-2", style="color: green;"),
                                        html.Div("{{ point3_text }}", classes="text-body-2", style="color: blue;"),
                                        vuetify.VBtn(
                                            "Clear Points",
                                            click=self._clear_points,
                                            block=True,
                                            variant="outlined",
                                            size="small",
                                            classes="mt-3"
                                        ),
                                    ]),
                                ]
                            )

                            vuetify.VCard(
                                classes="mb-4",
                                children=[
                                    vuetify.VCardTitle("Tube Radius"),
                                    vuetify.VCardText([
                                        vuetify.VSlider(
                                            v_model=("minor_radius", 2.0),
                                            min=0.1,
                                            max=20,
                                            step=0.1,
                                            thumb_label="always",
                                            hide_details=True,
                                        ),
                                        vuetify.VBtn(
                                            "Update Preview",
                                            click=self._on_radius_change,
                                            block=True,
                                            variant="text",
                                            size="small",
                                            classes="mt-2"
                                        ),
                                    ]),
                                ]
                            )

                            vuetify.VBtn(
                                "Preview Torus",
                                click=self._show_preview,
                                block=True,
                                color="primary",
                                disabled=("!can_preview",),
                                classes="mb-2"
                            )

                            vuetify.VBtn(
                                "Subtract",
                                click=self._perform_subtraction,
                                block=True,
                                color="error",
                                disabled=("!can_subtract",),
                            )

    def _load_file(self):
        """Load STL file from path."""
        path = self.state.file_path
        if not path:
            self.state.status_message = "Please enter a file path"
            return

        # Expand user home directory
        path = os.path.expanduser(path)

        if not os.path.exists(path):
            self.state.status_message = f"File not found: {path}"
            return

        try:
            self._load_stl_file(path)
        except Exception as e:
            self.state.status_message = f"Error loading: {str(e)}"
            import traceback
            traceback.print_exc()

    def _load_stl_file(self, path):
        """Load STL file from path."""
        self.source_mesh_pv = pv.read(path)
        self.source_mesh_tri = trimesh.load(path)

        # Calculate marker scale
        bounds = self.source_mesh_pv.bounds
        diag = np.sqrt(
            (bounds[1] - bounds[0])**2 +
            (bounds[3] - bounds[2])**2 +
            (bounds[5] - bounds[4])**2
        )
        self.marker_scale = diag / 50
        self.state.minor_radius = max(diag / 30, 0.5)

        # Display
        self.plotter.clear()
        self.plotter.add_mesh(
            self.source_mesh_pv,
            color='white',
            name='source_mesh',
            pickable=True
        )
        self.plotter.add_axes()

        # Enable picking
        self.plotter.enable_point_picking(
            callback=self._on_point_picked,
            use_mesh=True,
            show_message=False,
            show_point=False,
            picker='cell'
        )

        self.plotter.reset_camera()
        self._clear_points_internal()
        self.state.can_export = True
        self.state.status_message = "Model loaded! Click on mesh to place Point 1"
        self.ctrl.view_update()

    def _on_point_picked(self, point):
        """Handle point picking."""
        if point is None:
            return
        if len(self.picked_points) >= 3:
            return

        self.picked_points.append(np.array(point))

        # Add marker
        colors = ['red', 'green', 'blue']
        color = colors[len(self.picked_points) - 1]
        sphere = pv.Sphere(radius=self.marker_scale, center=point)
        self.plotter.add_mesh(
            sphere,
            color=color,
            name=f'point_marker_{len(self.picked_points)}'
        )

        # Update state
        self._update_point_labels()
        self.state.points_count = len(self.picked_points)

        if len(self.picked_points) == 3:
            self.state.status_message = "3 points selected! Click Preview or Subtract"
            self.state.can_preview = True
            self._show_preview()
        else:
            self.state.status_message = f"Click to place Point {len(self.picked_points) + 1}"

        self.ctrl.view_update()

    def _update_point_labels(self):
        """Update point labels in UI."""
        labels = ['point1_text', 'point2_text', 'point3_text']
        for i, key in enumerate(labels):
            if i < len(self.picked_points):
                p = self.picked_points[i]
                setattr(self.state, key, f"Point {i+1}: ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")
            else:
                setattr(self.state, key, f"Point {i+1}: -")

    def _clear_points_internal(self):
        """Clear points without updating view."""
        self.picked_points = []
        for i in range(3):
            self.plotter.remove_actor(f'point_marker_{i+1}')
        self.plotter.remove_actor('torus_preview')
        self.state.can_preview = False
        self.state.can_subtract = False
        self.state.points_count = 0
        self._update_point_labels()

    def _clear_points(self):
        """Clear all picked points."""
        self._clear_points_internal()
        if self.source_mesh_pv is not None:
            self.state.status_message = "Click on mesh to place Point 1"
        self.ctrl.view_update()

    def _on_radius_change(self):
        """Handle radius slider change."""
        if len(self.picked_points) == 3:
            self._show_preview()

    def _calculate_torus(self):
        """Calculate torus mesh."""
        if len(self.picked_points) != 3:
            return None

        result = fit_circle_3d(*self.picked_points)
        if result is None:
            self.state.status_message = "Error: Points are collinear!"
            return None

        center, major_radius, normal, _, _ = result

        return generate_torus_segment(
            center, normal, major_radius, self.state.minor_radius,
            *self.picked_points
        )

    def _show_preview(self):
        """Show torus preview."""
        torus_mesh = self._calculate_torus()
        if torus_mesh is None:
            return

        pv_mesh = pv.wrap(torus_mesh)
        self.plotter.remove_actor('torus_preview')
        self.plotter.add_mesh(
            pv_mesh,
            color='orange',
            opacity=0.6,
            name='torus_preview'
        )

        self.state.can_subtract = True
        self.ctrl.view_update()

    def _perform_subtraction(self):
        """Perform boolean subtraction."""
        torus_mesh = self._calculate_torus()
        if torus_mesh is None:
            return

        self.state.status_message = "Processing boolean subtraction..."
        self.ctrl.view_update()

        result = subtract_meshes(self.source_mesh_tri, torus_mesh)

        if result is None or result.is_empty:
            self.state.status_message = "Boolean operation failed! Try different points."
            return

        # Update meshes
        self.source_mesh_tri = result
        self.source_mesh_pv = pv.wrap(result)

        # Redisplay
        self.plotter.clear()
        self.plotter.add_mesh(
            self.source_mesh_pv,
            color='white',
            name='source_mesh',
            pickable=True
        )
        self.plotter.add_axes()
        self.plotter.enable_point_picking(
            callback=self._on_point_picked,
            use_mesh=True,
            show_message=False,
            show_point=False,
            picker='cell'
        )

        self._clear_points_internal()
        self.state.status_message = "Subtraction complete! Select more points or export."
        self.ctrl.view_update()

    def _export_stl(self):
        """Export mesh to STL file."""
        if self.source_mesh_tri is None:
            self.state.status_message = "No mesh to export!"
            return

        path = os.path.expanduser(self.state.export_path)

        try:
            self.source_mesh_tri.export(path)
            self.state.status_message = f"Exported to: {path}"
        except Exception as e:
            self.state.status_message = f"Export failed: {str(e)}"


def create_app():
    """Create and return the Trame app."""
    app = TorusToolApp()
    return app.server

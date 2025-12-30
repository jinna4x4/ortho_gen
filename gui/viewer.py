"""
Main GUI viewer widget using PySide6 and PyVista.
"""
import numpy as np
import pyvista as pv
import trimesh
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QFileDialog, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt
from pyvistaqt import QtInteractor

from geometry.circle_fit import fit_circle_3d
from geometry.torus_generator import generate_torus_segment
from operations.boolean_ops import subtract_meshes, check_mesh_validity


class TorusToolViewer(QWidget):
    """Main viewer widget with 3D viewport and controls."""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data
        self.source_mesh_pv = None      # PyVista PolyData for display
        self.source_mesh_tri = None     # Trimesh for operations
        self.preview_actor = None       # Actor for torus preview
        self.picked_points = []         # List of picked 3D points
        self.point_actors = []          # Actors for point markers

        self.minor_radius = 2.0
        self.marker_scale = 1.0

        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface."""
        main_layout = QHBoxLayout(self)

        # Sidebar
        sidebar = QWidget()
        sidebar.setMaximumWidth(220)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setAlignment(Qt.AlignTop)

        # File operations group
        grp_file = QGroupBox("File")
        l_file = QVBoxLayout()
        btn_load = QPushButton("Load STL")
        btn_load.clicked.connect(self._load_stl)
        btn_export = QPushButton("Export Result")
        btn_export.clicked.connect(self._export_stl)
        l_file.addWidget(btn_load)
        l_file.addWidget(btn_export)
        grp_file.setLayout(l_file)
        side_layout.addWidget(grp_file)

        # Parameters group
        grp_params = QGroupBox("Parameters")
        l_params = QVBoxLayout()
        l_params.addWidget(QLabel("Tube Radius:"))
        self.slider_radius = QSlider(Qt.Horizontal)
        self.slider_radius.setRange(1, 200)  # 0.1 to 20.0
        self.slider_radius.setValue(20)
        self.slider_radius.valueChanged.connect(self._update_radius)
        l_params.addWidget(self.slider_radius)
        self.lbl_radius_val = QLabel("2.0")
        l_params.addWidget(self.lbl_radius_val)
        grp_params.setLayout(l_params)
        side_layout.addWidget(grp_params)

        # Points group
        grp_points = QGroupBox("Points")
        l_points = QVBoxLayout()
        self.lbl_status = QLabel("Load a model to begin")
        self.lbl_status.setWordWrap(True)
        l_points.addWidget(self.lbl_status)

        self.lbl_point1 = QLabel("Point 1: -")
        self.lbl_point2 = QLabel("Point 2: -")
        self.lbl_point3 = QLabel("Point 3: -")
        l_points.addWidget(self.lbl_point1)
        l_points.addWidget(self.lbl_point2)
        l_points.addWidget(self.lbl_point3)

        btn_clear = QPushButton("Clear Points")
        btn_clear.clicked.connect(self._clear_points)
        l_points.addWidget(btn_clear)
        grp_points.setLayout(l_points)
        side_layout.addWidget(grp_points)

        # Actions group
        grp_act = QGroupBox("Actions")
        l_act = QVBoxLayout()
        self.btn_preview = QPushButton("Preview Torus")
        self.btn_preview.clicked.connect(self._show_preview)
        self.btn_preview.setEnabled(False)

        self.btn_subtract = QPushButton("Subtract")
        self.btn_subtract.clicked.connect(self._perform_subtraction)
        self.btn_subtract.setEnabled(False)

        l_act.addWidget(self.btn_preview)
        l_act.addWidget(self.btn_subtract)
        grp_act.setLayout(l_act)
        side_layout.addWidget(grp_act)

        side_layout.addStretch()

        # 3D Viewer
        self.plotter = QtInteractor(self)
        self.plotter.set_background("gray")
        self.plotter.add_axes()

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.plotter, stretch=1)

    def _load_stl(self):
        """Load an STL file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open STL", "", "STL Files (*.stl);;All Files (*)"
        )
        if not path:
            return

        try:
            # Load with PyVista for display
            self.source_mesh_pv = pv.read(path)
            # Load with trimesh for operations
            self.source_mesh_tri = trimesh.load(path)

            # Calculate marker scale based on model size
            bounds = self.source_mesh_pv.bounds
            diag = np.sqrt(
                (bounds[1] - bounds[0])**2 +
                (bounds[3] - bounds[2])**2 +
                (bounds[5] - bounds[4])**2
            )
            self.marker_scale = diag / 100

            # Update slider range based on model size
            self.slider_radius.setRange(1, int(diag / 2))
            self.minor_radius = diag / 50
            self.slider_radius.setValue(int(self.minor_radius * 10))

            # Display
            self.plotter.clear()
            self.plotter.add_mesh(
                self.source_mesh_pv,
                color='white',
                opacity=1.0,
                name='source_mesh'
            )
            self.plotter.reset_camera()

            # Enable point picking
            self.plotter.enable_point_picking(
                callback=self._on_point_picked,
                use_mesh=True,
                show_message=False,
                show_point=False
            )

            self._clear_points()
            self.lbl_status.setText("Click on mesh to place Point 1")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load STL: {e}")

    def _on_point_picked(self, point):
        """Handle point picking on mesh surface."""
        if len(self.picked_points) >= 3:
            return

        self.picked_points.append(np.array(point))

        # Add marker sphere
        colors = ['red', 'green', 'blue']
        color = colors[len(self.picked_points) - 1]
        sphere = pv.Sphere(radius=self.marker_scale * 2, center=point)
        actor = self.plotter.add_mesh(
            sphere,
            color=color,
            name=f'point_marker_{len(self.picked_points)}'
        )
        self.point_actors.append(actor)

        # Update labels
        self._update_point_labels()

        if len(self.picked_points) == 3:
            self.lbl_status.setText("3 points selected. Preview or adjust radius.")
            self.btn_preview.setEnabled(True)
            self._show_preview()
        else:
            self.lbl_status.setText(
                f"Click on mesh to place Point {len(self.picked_points) + 1}"
            )

    def _update_point_labels(self):
        """Update the point coordinate labels."""
        labels = [self.lbl_point1, self.lbl_point2, self.lbl_point3]
        for i, lbl in enumerate(labels):
            if i < len(self.picked_points):
                p = self.picked_points[i]
                lbl.setText(f"Point {i+1}: ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")
            else:
                lbl.setText(f"Point {i+1}: -")

    def _clear_points(self):
        """Clear all picked points."""
        self.picked_points = []

        # Remove point markers
        for i in range(3):
            self.plotter.remove_actor(f'point_marker_{i+1}')
        self.point_actors = []

        # Remove preview
        self.plotter.remove_actor('torus_preview')
        self.preview_actor = None

        self.btn_preview.setEnabled(False)
        self.btn_subtract.setEnabled(False)
        self._update_point_labels()

        if self.source_mesh_pv is not None:
            self.lbl_status.setText("Click on mesh to place Point 1")

    def _update_radius(self):
        """Update minor radius from slider."""
        val = self.slider_radius.value() / 10.0
        self.minor_radius = val
        self.lbl_radius_val.setText(f"{val:.1f}")

        # Update preview if 3 points are selected
        if len(self.picked_points) == 3:
            self._show_preview()

    def _calculate_torus(self):
        """Calculate torus mesh from picked points."""
        if len(self.picked_points) != 3:
            return None

        result = fit_circle_3d(*self.picked_points)
        if result is None:
            self.lbl_status.setText("Error: Points are collinear!")
            return None

        center, major_radius, normal, _, _ = result

        torus_mesh = generate_torus_segment(
            center, normal, major_radius, self.minor_radius,
            *self.picked_points
        )

        return torus_mesh

    def _show_preview(self):
        """Show torus preview overlay."""
        torus_mesh = self._calculate_torus()
        if torus_mesh is None:
            return

        # Convert trimesh to PyVista
        pv_mesh = pv.wrap(torus_mesh)

        # Remove old preview
        self.plotter.remove_actor('torus_preview')

        # Add new preview
        self.preview_actor = self.plotter.add_mesh(
            pv_mesh,
            color='orange',
            opacity=0.5,
            name='torus_preview'
        )

        self.btn_subtract.setEnabled(True)

    def _perform_subtraction(self):
        """Perform boolean subtraction."""
        torus_mesh = self._calculate_torus()
        if torus_mesh is None:
            return

        self.lbl_status.setText("Processing boolean subtraction...")
        self.repaint()

        # Check mesh validity
        valid, msg = check_mesh_validity(self.source_mesh_tri)
        if not valid:
            QMessageBox.warning(
                self, "Mesh Warning",
                f"Source mesh has issues: {msg}\nAttempting operation anyway."
            )

        result = subtract_meshes(self.source_mesh_tri, torus_mesh)

        if result is None or result.is_empty:
            QMessageBox.critical(
                self, "Error",
                "Boolean subtraction failed. Try adjusting the torus position."
            )
            self.lbl_status.setText("Boolean operation failed")
            return

        # Update meshes
        self.source_mesh_tri = result
        self.source_mesh_pv = pv.wrap(result)

        # Redisplay
        self.plotter.clear()
        self.plotter.add_mesh(
            self.source_mesh_pv,
            color='white',
            opacity=1.0,
            name='source_mesh'
        )
        self.plotter.add_axes()

        # Re-enable picking
        self.plotter.enable_point_picking(
            callback=self._on_point_picked,
            use_mesh=True,
            show_message=False,
            show_point=False
        )

        self._clear_points()
        self.lbl_status.setText("Subtraction complete! Click to add more points.")

    def _export_stl(self):
        """Export the current mesh to STL."""
        if self.source_mesh_tri is None:
            QMessageBox.warning(self, "Warning", "No mesh to export!")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save STL", "", "STL Files (*.stl)"
        )
        if not path:
            return

        try:
            self.source_mesh_tri.export(path)
            QMessageBox.information(self, "Success", f"Exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {e}")

import numpy as np


class GridPlaneRenderer:
    """Manages the napari Vectors layer that displays a grid plane."""

    def __init__(self, viewer, render_config, on_clamped=None):
        self.viewer = viewer
        self.render_config = render_config
        self.on_clamped = on_clamped  # callable(max_distance_um: float) | None
        self.grid_plane_layer = None
        self.default_line_thickness_nm = None
        self.grid_plane_layer_opacity = 0.75
        self.current_grid_plane_color = "white"
        self.current_grid_plane_z_pos = None
        self.line_thickness_value = 50

    def _apply_no_depth_blending(self):
        """Set translucent_no_depth blending on the grid plane layer.

        This disables depth testing so the grid plane (a spatial overlay)
        always renders on top of any reference images or other layers that
        happen to sit at the same Z coordinate.  Without this the two
        surfaces fight for the same depth-buffer slot which causes parts of
        the grid to flicker or vanish as the camera is rotated.
        """
        if self.grid_plane_layer is not None:
            self.grid_plane_layer.blending = "translucent_no_depth"

    @staticmethod
    def _window(axis_range, percent_range):
        """Return the selected world-coordinate interval for one axis."""
        low, high = np.asarray(axis_range, dtype=float)
        selected = low + np.asarray(percent_range, dtype=float) / 100 * (high - low)
        return float(selected[0]), float(selected[1])

    def _grid_metrics(self, render_range_x, render_range_y, line_distance_nm):
        """Return selected bounds, spans, and unique line counts."""
        if not np.isfinite(line_distance_nm) or line_distance_nm <= 0:
            raise ValueError("Grid line distance must be a finite value above zero")

        x0, x1 = self._window(
            render_range_x, self.render_config.range_x_percent
        )
        y0, y1 = self._window(
            render_range_y, self.render_config.range_y_percent
        )
        x_span = max(0.0, x1 - x0)
        y_span = max(0.0, y1 - y0)

        # X-oriented lines are distributed along Y, and vice versa.
        x_line_intervals = int(np.floor(y_span / line_distance_nm))
        y_line_intervals = int(np.floor(x_span / line_distance_nm))
        return (
            x0,
            y0,
            x_span,
            y_span,
            x_line_intervals,
            y_line_intervals,
        )

    def _make_vectors(self, render_range_x, render_range_y, line_distance_nm):
        """Build grid vectors in napari's (z, y, x) coordinate order."""
        (
            x0,
            y0,
            x_span,
            y_span,
            x_line_intervals,
            y_line_intervals,
        ) = self._grid_metrics(render_range_x, render_range_y, line_distance_nm)

        # Column 1 is y and column 2 is x: lines that run along x step in y,
        # and vice versa.
        vectors_x = np.zeros((x_line_intervals + 1, 2, 3), dtype=np.float32)
        vectors_x[:, 0, 0] = self.current_grid_plane_z_pos
        vectors_x[:, 0, 2] = x0
        vectors_x[:, 0, 1] = (
            y0 + np.arange(x_line_intervals + 1) * line_distance_nm
        )
        vectors_x[:, 1, 2] = x_span

        vectors_y = np.zeros((y_line_intervals + 1, 2, 3), dtype=np.float32)
        vectors_y[:, 0, 0] = self.current_grid_plane_z_pos
        vectors_y[:, 0, 2] = (
            x0 + np.arange(y_line_intervals + 1) * line_distance_nm
        )
        vectors_y[:, 0, 1] = y0
        vectors_y[:, 1, 1] = y_span

        # VisPy's vector stroke can taper slightly along its direction.  Draw
        # the exact reverse of every vector as well so the two directional
        # profiles compensate while preserving the same endpoints and spacing.
        vectors_x_reverse = vectors_x.copy()
        vectors_x_reverse[:, 0, :] += vectors_x[:, 1, :]
        vectors_x_reverse[:, 1, :] *= -1
        vectors_y_reverse = vectors_y.copy()
        vectors_y_reverse[:, 0, :] += vectors_y[:, 1, :]
        vectors_y_reverse[:, 1, :] *= -1

        vectors = np.concatenate(
            (vectors_x, vectors_x_reverse, vectors_y, vectors_y_reverse)
        )
        return vectors, x_line_intervals, y_line_intervals, x_span, y_span

    @staticmethod
    def _edge_width(slider_value, x_line_intervals, y_line_intervals, x_span, y_span):
        """Map the 1..100 thickness slider around a geometry-derived default."""
        base_width = (
            0.05
            / max(np.mean((x_line_intervals, y_line_intervals)), 1)
            * np.mean((x_span, y_span))
        )
        return base_width * np.exp((slider_value - 50) / 10)

    def _add_layer(self, vectors, edge_width):
        self.grid_plane_layer = self.viewer.add_vectors(
            vectors,
            edge_width=edge_width,
            name="Grid_Plane",
            edge_color=self.current_grid_plane_color,
            ndim=3,
            opacity=self.grid_plane_layer_opacity,
        )
        self._apply_no_depth_blending()

    def create_remove(self, enable, render_range_x, render_range_y, render_range_z):
        """Create or remove the grid plane Vectors layer."""
        if enable:
            if (
                self.grid_plane_layer is not None
                and self.grid_plane_layer in self.viewer.layers
            ):
                return
            if self.render_config.zdim:
                if (
                    self.current_grid_plane_z_pos is None
                    or not (
                        render_range_z[0]
                        <= self.current_grid_plane_z_pos
                        <= render_range_z[1]
                    )
                ):
                    self.current_grid_plane_z_pos = np.mean(render_range_z)
            else:
                self.current_grid_plane_z_pos = 1
            default_line_dist_nm = self.render_config.grid_plane_line_distance_um * 1000
            vectors, num_of_lines_x, num_of_lines_y, x_span, y_span = (
                self._make_vectors(
                    render_range_x, render_range_y, default_line_dist_nm
                )
            )
            self.default_line_thickness_nm = self._edge_width(
                self.line_thickness_value,
                num_of_lines_x,
                num_of_lines_y,
                x_span,
                y_span,
            )
            self._add_layer(vectors, self.default_line_thickness_nm)
        else:
            if self.grid_plane_layer is None:
                return
            self.default_line_thickness_nm = self.grid_plane_layer.edge_width
            self.grid_plane_layer_opacity = self.grid_plane_layer.opacity
            if self.grid_plane_layer in self.viewer.layers:
                self.viewer.layers.remove(self.grid_plane_layer)
            self.grid_plane_layer = None

    def update(
        self,
        render_range_x,
        render_range_y,
        render_range_z,
        z_pos=None,
        line_thickness=None,
        line_distance_nm=None,
        color=None,
        opacity=None,
    ):
        """Update grid plane properties."""
        if line_distance_nm is not None:
            if self.grid_plane_layer is None:
                return
            self.grid_plane_layer_opacity = self.grid_plane_layer.opacity
            vectors, num_of_lines_x, num_of_lines_y, x_span, y_span = (
                self._make_vectors(render_range_x, render_range_y, line_distance_nm)
            )
            if num_of_lines_x < 1 or num_of_lines_y < 1:
                tmp_max_line_dist = np.round(min(x_span, y_span) * 0.001, 3)
                if self.on_clamped:
                    self.on_clamped(tmp_max_line_dist)
                return
            default_line_thickness_nm = self._edge_width(
                self.line_thickness_value,
                num_of_lines_x,
                num_of_lines_y,
                x_span,
                y_span,
            )
            self.viewer.layers.remove(self.grid_plane_layer)
            self._add_layer(vectors, default_line_thickness_nm)

        if z_pos is not None:
            vectors = self.grid_plane_layer.data
            if self.render_config.zdim:
                self.current_grid_plane_z_pos = (
                    render_range_z[0]
                    + z_pos / 100 * (render_range_z[1] - render_range_z[0])
                )
                vectors[:, 0, 0] = self.current_grid_plane_z_pos
            else:
                vectors[:, 0, 0] = 1
            self.grid_plane_layer.data = vectors
        if line_thickness is not None:
            self.line_thickness_value = line_thickness
            line_distance_nm = self.render_config.grid_plane_line_distance_um * 1000
            (
                _x0,
                _y0,
                x_span,
                y_span,
                num_of_lines_x,
                num_of_lines_y,
            ) = self._grid_metrics(
                render_range_x, render_range_y, line_distance_nm
            )
            self.grid_plane_layer.edge_width = self._edge_width(
                line_thickness,
                num_of_lines_x,
                num_of_lines_y,
                x_span,
                y_span,
            )
        if color is not None:
            self.current_grid_plane_color = color
            self.grid_plane_layer.edge_color = color
        if opacity is not None:
            self.grid_plane_layer.opacity = opacity / 100

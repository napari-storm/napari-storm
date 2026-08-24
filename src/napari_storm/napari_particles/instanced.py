"""Instanced Gaussian billboards — the Level 3 prototype.

`backend-comparison.md` ruled out the napari-native path and left instancing as
the only remaining candidate, with a concrete target: billboard quality at
points-like memory. This is the prototype that says whether that is reachable.

The idea is the one the plan describes: "store one position, sigma/uncertainty,
intensity/value, and flags record per localization plus one static quad". The
current renderer expands each localization into six vertices and repeats its
centre, sigma and value across all of them. Here the quad exists once, and each
localization contributes one row to each per-instance buffer, bound with
``divisor=1`` so the GPU reuses the quad for every instance.

**Status: prototype.** It draws, and it is measured, but it is not wired into
napari — that needs a `Layer` subclass and an entry in napari's private
`layer_to_visual` map, which belongs behind `_napari_compat`. What it settles is
whether the memory target is real and whether the Gaussian survives the change,
which are the two questions the decision rests on.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "InstancedBillboards",
    "instanced_bytes_per_localization",
    "VERTEX_SHADER",
    "FRAGMENT_SHADER",
]

#: The quad, once.  Corners in [-0.5, 0.5]; every instance reuses these.
QUAD = np.array(
    [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]], dtype=np.float32
)
QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

#: Quad half-extent as a multiple of sigma.  Matches the existing renderer's
#: `5 * max(sigma)` billboard, so a Gaussian is cut at the same 4% of its peak.
SIGMA_EXTENT = 5.0

VERTEX_SHADER = """
#version 120

// Per-vertex: the shared quad, four corners, uploaded once.
attribute vec2 a_quad;

// Per-instance: one row per localization, bound with divisor 1.
attribute vec3 a_center;
attribute vec3 a_sigma;
attribute float a_value;
attribute float a_size;

uniform mat4 u_transform;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;

varying vec2 v_quad;
varying float v_value;
varying vec2 v_inv_sigma;

void main() {
    // Expand the shared quad around this instance's centre, facing the camera.
    vec3 offset = (u_camera_right * a_quad.x + u_camera_up * a_quad.y) * a_size;
    gl_Position = u_transform * vec4(a_center + offset, 1.0);

    v_quad = a_quad;
    v_value = a_value;
    // Quad coordinate -> multiples of this instance's own sigma.  Sigmas
    // arrive normalized against the largest in the dataset, which is what
    // a_size was derived from, so the conversion is a constant over sigma.
    v_inv_sigma = vec2(
        SIGMA_EXTENT / max(a_sigma.y, 1e-6),
        SIGMA_EXTENT / max(a_sigma.z, 1e-6)
    );
}
""".replace("SIGMA_EXTENT", repr(SIGMA_EXTENT))

FRAGMENT_SHADER = """
#version 120

varying vec2 v_quad;
varying float v_value;
varying vec2 v_inv_sigma;

void main() {
    vec2 d = v_quad * v_inv_sigma;
    float gaussian = exp(-0.5 * dot(d, d));
    // The falloff goes in alpha as well as colour, so additive blending
    // (src_alpha, one) writes the Gaussian squared.  That is what the existing
    // renderer's shader does too -- it returns `val*vec4(1,1,1,1)` under the
    // same blend -- so the prototype matches the backend it would replace
    // rather than quietly changing how a reconstruction looks.
    gl_FragColor = vec4(vec3(v_value * gaussian), gaussian);
}
"""


def instanced_bytes_per_localization(include_value=True, include_size=True):
    """Host bytes one localization costs in this layout.

    The quad and its indices are shared and amortize to nothing, so the whole
    cost is the per-instance rows: centre and sigma at ``(N, 3)`` float32, plus
    a value and a size at ``(N,)``.
    """
    total = 3 * 4 + 3 * 4  # centre + sigma
    if include_value:
        total += 4
    if include_size:
        total += 4
    return total


class InstancedBillboards:
    """Per-instance buffers plus one shared quad.

    Holds the arrays and, when a GL context is available, the gloo program that
    draws them. The arrays exist and can be measured without a context, which
    is what makes the memory claim testable in CI.
    """

    def __init__(self, coords, sigmas, values, size):
        self._program = None
        self._indices = None
        self.set_data(coords, sigmas, values, size)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def set_data(self, coords, sigmas, values, size):
        """One row per localization.  No expansion, no repetition."""
        coords = np.ascontiguousarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must be (N, 3)")
        n = len(coords)

        sigmas = np.ascontiguousarray(
            np.broadcast_to(np.asarray(sigmas, dtype=np.float32), (n, 3))
        )
        values = np.ascontiguousarray(
            np.broadcast_to(np.asarray(values, dtype=np.float32), (n,))
        )
        sizes = np.ascontiguousarray(
            np.broadcast_to(np.asarray(size, dtype=np.float32), (n,))
        )

        self.centers = coords
        self.sigmas = sigmas
        self.values = values
        self.sizes = sizes
        if self._program is not None:
            self._upload()

    @property
    def n_instances(self):
        return len(self.centers)

    def host_bytes(self):
        """Everything this holds, including the amortized shared quad."""
        return int(
            self.centers.nbytes
            + self.sigmas.nbytes
            + self.values.nbytes
            + self.sizes.nbytes
            + QUAD.nbytes
            + QUAD_INDICES.nbytes
        )

    def bytes_per_localization(self):
        return self.host_bytes() / max(self.n_instances, 1)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def build_program(self):
        """Create the gloo program.  Requires a current GL context."""
        from vispy.gloo import IndexBuffer, Program, VertexBuffer

        program = Program(VERTEX_SHADER, FRAGMENT_SHADER)
        program["a_quad"] = VertexBuffer(QUAD)
        self._program = program
        self._indices = IndexBuffer(QUAD_INDICES)
        self._upload()
        return program

    def _upload(self):
        """Bind the per-instance buffers with divisor 1.

        ``divisor=1`` is the whole mechanism: the GPU advances these once per
        instance rather than once per vertex, so four quad vertices serve every
        localization in the dataset.
        """
        from vispy.gloo import VertexBuffer

        self._program["a_center"] = VertexBuffer(self.centers, divisor=1)
        self._program["a_sigma"] = VertexBuffer(self.sigmas, divisor=1)
        self._program["a_value"] = VertexBuffer(self.values, divisor=1)
        self._program["a_size"] = VertexBuffer(self.sizes, divisor=1)

    def draw(self, transform, camera_right, camera_up):
        """One instanced draw call for the whole dataset."""
        if self._program is None:
            self.build_program()
        self._program["u_transform"] = np.asarray(transform, dtype=np.float32)
        self._program["u_camera_right"] = np.asarray(camera_right, dtype=np.float32)
        self._program["u_camera_up"] = np.asarray(camera_up, dtype=np.float32)
        self._program.draw("triangles", self._indices)

"""Where the user is looking: the rotation centre, and the focus region.

A 3-D view rotates about the camera centre. Panning slides that point across
the screen but never changes its depth, so it ends up in front of or behind
whatever is being looked at, and the structure swings away on every turn.
`pivot_on_view_axis` finds the depth of the localizations in the middle of the
view, so the centre can be put on them -- along the line of sight, which moves
nothing on screen. `point_under_ray` finds the structure under a click.

Focus mode fades out what lies further than a radius from that centre.
`focus_weight` is the falloff the shader draws, written out in numpy so it can
be tested and documented without a GPU.

Points and directions are ``(z, y, x)``, napari's world order, in nanometres.
Host-free, like the rest of `core`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "DEFAULT_FOCUS_RADIUS_NM",
    "DEFAULT_OUTSIDE_OPACITY",
    "FOCUS_EDGE_FRACTION",
    "FocusState",
    "focus_weight",
    "pivot_on_view_axis",
    "point_under_ray",
]

#: How far the focus region reaches by default.
DEFAULT_FOCUS_RADIUS_NM = 1000.0

#: What is left of a localization far outside the focus region.
DEFAULT_OUTSIDE_OPACITY = 0.15

#: The fade from full to the outside opacity takes this fraction of the radius.
FOCUS_EDGE_FRACTION = 0.25


@dataclass(frozen=True)
class FocusState:
    """A focus region: a sphere, or a cylinder along z when *planar*.

    *planar* measures distance in the y-x plane alone, which is what a 2-D
    view of 3-D data needs: everything along the line of sight is equally in
    focus.
    """

    center_world: tuple
    radius_nm: float = DEFAULT_FOCUS_RADIUS_NM
    outside_opacity: float = DEFAULT_OUTSIDE_OPACITY
    planar: bool = False

    def __post_init__(self):
        if not np.isfinite(self.radius_nm) or self.radius_nm <= 0:
            raise ValueError("the focus radius must be a positive number of nm")
        if not 0.0 <= self.outside_opacity <= 1.0:
            raise ValueError("the outside opacity must lie between 0 and 1")

    @property
    def edge_nm(self):
        """Width of the fade, beyond the radius."""
        return max(self.radius_nm * FOCUS_EDGE_FRACTION, 1e-6)


def focus_weight(points_zyx, focus):
    """What each point keeps of its opacity: 1 inside, falling to the floor.

    The same smoothstep the shader applies, from the radius out to the radius
    plus the edge.
    """
    points = np.asarray(points_zyx, dtype=np.float64)
    delta = points - np.asarray(focus.center_world, dtype=np.float64)
    if focus.planar:
        delta = delta[:, 1:]
    distance = np.linalg.norm(delta, axis=1)
    t = np.clip((distance - focus.radius_nm) / focus.edge_nm, 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return 1.0 + (focus.outside_opacity - 1.0) * smooth


def _along_ray(points, origin, direction):
    """Signed distance along the ray, and squared distance from it."""
    direction = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("the view direction must be a non-zero vector")
    direction = direction / norm
    relative = np.asarray(points, dtype=np.float64) - np.asarray(
        origin, dtype=np.float64
    )
    depth = relative @ direction
    lateral = np.einsum("ij,ij->i", relative, relative) - depth * depth
    return depth, lateral, direction


def pivot_on_view_axis(points_zyx, center, view_direction, radius_nm):
    """The camera centre moved along the line of sight onto the data, or None.

    Of the points within *radius_nm* of the line through *center*, the median
    depth: robust to a stray localization in front or a sparse one behind.
    Only the depth changes, so the picture does not move.
    """
    points = np.asarray(points_zyx, dtype=np.float64)
    if not len(points):
        return None
    depth, lateral, direction = _along_ray(points, center, view_direction)
    near = lateral <= float(radius_nm) ** 2
    if not near.any():
        return None
    return tuple(
        np.asarray(center, dtype=np.float64) + direction * np.median(depth[near])
    )


def point_under_ray(points_zyx, origin, view_direction, radius_nm):
    """The middle of the localizations under a click, or None.

    The median of each coordinate over the points within *radius_nm* of the
    line through *origin*: the structure the user pointed at, at its depth.
    """
    points = np.asarray(points_zyx, dtype=np.float64)
    if not len(points):
        return None
    _depth, lateral, _direction = _along_ray(points, origin, view_direction)
    near = lateral <= float(radius_nm) ** 2
    if not near.any():
        return None
    return tuple(np.median(points[near], axis=0))

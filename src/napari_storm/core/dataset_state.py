"""Per-dataset application state, separated from the localizations themselves.

:class:`~napari_storm.core.LocalizationTable` owns the measurements. This owns
everything the *application* decided about them: what they are called, where
they sit in world space, and how they are displayed. Both were previously
scattered — the name on the dataset object, the placement nowhere at all, and
the appearance inside Qt widget state, where it could not survive the widget or
be tested without one.

State changes announce themselves. A control that moves a slider changes this
object; whoever draws hears :class:`AppearanceChanged` and acts. That is the
inversion §4.1 asks for: a change carries what changed, rather than every
consumer rediscovering global state.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .bounds import EMPTY, Bounds
from .renderer import LayerAppearance
from .world_transform import IDENTITY, WorldTransform

__all__ = ["DatasetState", "MaskChanged", "AppearanceChanged", "TransformChanged"]


@dataclass(frozen=True)
class MaskChanged:
    """Which localizations are drawn changed: a filter, or the render budget."""

    dataset_id: int


@dataclass(frozen=True)
class AppearanceChanged:
    """How a dataset is drawn changed.  Nothing about the data moved."""

    dataset_id: int
    appearance: LayerAppearance = None


@dataclass(frozen=True)
class TransformChanged:
    """Where a dataset sits in world space changed."""

    dataset_id: int
    transform: WorldTransform = None


@dataclass
class DatasetState:
    """What the application knows about one loaded dataset.

    Deliberately not the localizations: this can be copied, compared and
    persisted without dragging a million rows along, which is what makes it a
    plausible basis for the scene format Level 4 has to define.
    """

    dataset_id: int
    name: str = ""
    appearance: LayerAppearance = field(default_factory=LayerAppearance)
    transform: WorldTransform = IDENTITY
    #: Extent per axis in world nanometres, derived from the table and cached
    #: here so consumers do not each recompute it from a million coordinates.
    bounds: dict = field(default_factory=lambda: {"x": EMPTY, "y": EMPTY, "z": EMPTY})

    def bounds_for(self, axis):
        return self.bounds.get(axis, EMPTY)

    def merged_bounds(self, axis, other):
        """This dataset's extent on *axis*, widened to include *other*."""
        return self.bounds_for(axis).union(other)

    def with_appearance(self, **fields):
        """The appearance updated in place; unspecified fields keep their value.

        Returns the new appearance, so a caller can hand it straight to an
        :class:`AppearanceChanged`.
        """
        changes = {name: value for name, value in fields.items() if value is not None}
        self.appearance = replace(self.appearance, **changes)
        return self.appearance

    def update_bounds_from(self, table, zdim_present):
        """Recompute the derived extents from a table.  Returns the bounds."""
        axes = ("x", "y", "z") if zdim_present else ("x", "y")
        bounds = {"x": EMPTY, "y": EMPTY, "z": EMPTY}
        for axis in axes:
            if not table.has_axis(axis):
                continue
            values = table.coordinate_nm(axis)
            bounds[axis] = Bounds.of(self.transform.apply_axis(axis, values))
        self.bounds = bounds
        return bounds

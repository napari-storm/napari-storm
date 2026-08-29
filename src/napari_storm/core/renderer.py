"""The contract between render planning and whatever actually draws.

Level 3 of ``docs/modernization-review.md`` requires prototyping two renderer
backends "on the same fixtures before selecting the production backend". That is
only possible if there is a seam to swap: without one, a prototype means forking
the code that computes *what* to draw along with the code that draws it.

This module is that seam. Everything above it — filtering, coordinate
conversion, sigma and value computation, the memory budget — decides what a
dataset should look like and hands over a :class:`RenderRequest`. Everything
below it owns GPU resources and host-specific layer objects.

Host-free, like the rest of ``napari_storm.core``: a request is numpy arrays and
a name. The one deliberately loose field is ``colormap``, which is whatever
palette handle the backend understands; the planner passes it through without
inspecting it, so a napari ``Colormap`` never has to be named here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Flag, auto
from typing import Any

import numpy as np

__all__ = [
    "Changed",
    "LayerAppearance",
    "RenderRequest",
    "LocalizationRenderer",
    "NullRenderer",
]


class Changed(Flag):
    """Which parts of a request differ from what the backend already has.

    Level 3 has to compare backends on targeted updates, and a backend that
    cannot tell a sigma change from a filter change has to re-upload
    everything. Passing the dirty set makes the comparison fair -- a backend
    that *can* update only its sigma buffer is allowed to show it.

    A backend may ignore this and rebuild everything; it is a hint about what
    is needed, never a licence to leave a declared change unapplied.
    """

    NOTHING = 0
    #: The set of localizations changed: filtering, or the render budget.
    SELECTION = auto()
    #: Positions moved.  Extent and any spatial index are invalid.
    POSITIONS = auto()
    #: Gaussian widths, and with them the billboard size.
    SIGMAS = auto()
    #: Per-localization intensity or colour index.
    VALUES = auto()
    EVERYTHING = SELECTION | POSITIONS | SIGMAS | VALUES


#: Palette used when a caller does not choose one.  Visible on a dark canvas
#: and neutral enough not to imply a channel identity.
DEFAULT_COLORMAP = "gray"


@dataclass(frozen=True)
class LayerAppearance:
    """How a dataset is displayed, as distinct from what is displayed.

    None means "leave this as it is", so a control that owns one slider can
    send only what it changed rather than having to know the rest.
    """

    colormap: Any = None
    opacity: float = None
    contrast_limits: tuple = None
    visible: bool = None


@dataclass(frozen=True)
class RenderRequest:
    """Everything a backend needs to draw one dataset.

    Attributes:
        coords: ``(N, 3)`` float32 world coordinates in ``(z, y, x)`` order --
            napari's own, and the same order as ``sigmas`` beside it, so a
            backend may pair the two column-wise. This is the one place the
            renderer's axis order is stated; the canonical table keys positions
            by axis *name* precisely so that the ordering lives at this
            boundary rather than throughout.
        sigmas: ``(N, 3)`` float32, normalized to the largest of them.
        size: billboard edge length in world units, already clamped to the
            screen-space budget.
        values: ``(N,)`` float32 per-localization intensity, or colour index
            when Z encoding is on.
        name: what the layer is called in the host.
        colormap: backend-defined palette handle, passed through untouched.
            Defaults to :data:`DEFAULT_COLORMAP` rather than to None, because
            "no colormap" is not a neutral choice: the host picks an arbitrary
            one, and the instanced backend -- which samples the colormap in its
            own shader -- resolved napari's unnamed default to black. An
            embedder who omitted it got a blank canvas with every piece of
            state reporting itself healthy, which is the hardest kind of
            nothing to debug.
        antialias: shader antialias distance cutoff; 0 disables it.
        active_ids: stable localization ids, in the same order as ``coords``.
            A backend that keeps persistent per-localization GPU storage needs
            these to know which rows the arrays correspond to; without them the
            only possible update is a full re-upload, which is precisely what
            Level 3 is trying to measure its way out of.
        changed: which parts differ from the last request for this dataset.
    """

    coords: np.ndarray
    sigmas: np.ndarray
    size: float
    values: np.ndarray
    name: str
    colormap: Any = DEFAULT_COLORMAP
    antialias: float = 0.0
    active_ids: np.ndarray = None
    changed: Changed = Changed.EVERYTHING

    def with_changes(self, changed):
        """The same request, declaring a narrower dirty set."""
        return replace(self, changed=changed)


class LocalizationRenderer:
    """What a backend must provide.  Subclass or duck-type it.

    Deliberately not a ``typing.Protocol``: the null implementation below wants
    to inherit the docstrings, and a plain base class costs nothing here.

    Backends key their resources by ``dataset_id``, never by layer name or by
    position in a list (§4.2). Ids are never reused, so a handle left behind by
    a closed dataset can never be mistaken for a live one.
    """

    def open(self, dataset_id, request):
        """Start drawing *dataset_id*.  Replaces anything already open for it."""
        raise NotImplementedError

    def update(self, dataset_id, request):
        """Redraw *dataset_id* from a new request, keeping its resources.

        The distinction from :meth:`open` is the whole point of the interface:
        an update must not destroy and recreate host resources, because doing
        so is what P0-01 was about.
        """
        raise NotImplementedError

    def set_visible(self, dataset_id, visible):
        """Show or hide *dataset_id* without discarding its resources."""
        raise NotImplementedError

    def set_appearance(self, dataset_id, appearance):
        """Apply a :class:`LayerAppearance`; None fields are left alone.

        Appearance is deliberately separate from :meth:`update`: changing a
        colormap must not require rebuilding a single buffer, and a control
        that owns one slider should not have to reach past the backend to a
        host layer object to move it.
        """
        raise NotImplementedError

    def appearance(self, dataset_id):
        """The current :class:`LayerAppearance`, or None if not open."""
        raise NotImplementedError

    def value_range(self, dataset_id):
        """``(low, high)`` the values span, for a contrast control to scale to.

        On the protocol rather than backend-specific because the contrast
        control needs it: a backend that cannot answer this cannot drive the
        application, which would make it untestable as a substitute.
        """
        raise NotImplementedError

    def close(self, dataset_id):
        """Release everything owned for *dataset_id*.  Safe if nothing is open."""
        raise NotImplementedError

    def close_all(self):
        """Release everything."""
        raise NotImplementedError

    def is_open(self, dataset_id):
        """True while *dataset_id* has live resources."""
        raise NotImplementedError

    def host_bytes(self, dataset_id):
        """Host-side bytes held for *dataset_id*, or 0.

        Part of the contract because the Level 3 decision turns on it: two
        backends can only be compared on memory if both report it the same
        way, rather than the harness reaching into whichever arrays it happens
        to know about.
        """
        return 0


@dataclass
class NullRenderer(LocalizationRenderer):
    """Records what it was asked to draw, and draws nothing.

    Lets the planning side — filtering, ranges, budget, sigma and value
    computation — be exercised with no viewer, no GPU and no Qt.
    """

    requests: dict = field(default_factory=dict)
    visibility: dict = field(default_factory=dict)
    appearances: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def open(self, dataset_id, request):
        self.calls.append(("open", dataset_id))
        self.requests[dataset_id] = request
        self.visibility[dataset_id] = True
        self.appearances[dataset_id] = LayerAppearance(
            colormap=request.colormap, opacity=1.0, visible=True
        )

    def update(self, dataset_id, request):
        self.calls.append(("update", dataset_id))
        if dataset_id not in self.requests:
            raise KeyError(f"dataset {dataset_id} is not open")
        self.requests[dataset_id] = request

    def set_visible(self, dataset_id, visible):
        self.calls.append(("set_visible", dataset_id))
        self.visibility[dataset_id] = bool(visible)

    def set_appearance(self, dataset_id, appearance):
        self.calls.append(("set_appearance", dataset_id))
        current = self.appearances.get(dataset_id)
        if current is None:
            raise KeyError(f"dataset {dataset_id} is not open")
        changes = {
            name: value for name, value in vars(appearance).items() if value is not None
        }
        self.appearances[dataset_id] = replace(current, **changes)
        if appearance.visible is not None:
            self.visibility[dataset_id] = bool(appearance.visible)

    def appearance(self, dataset_id):
        return self.appearances.get(dataset_id)

    def value_range(self, dataset_id):
        request = self.requests.get(dataset_id)
        if request is None or request.values is None or len(request.values) == 0:
            return None
        return float(np.min(request.values)), float(np.max(request.values))

    def close(self, dataset_id):
        self.calls.append(("close", dataset_id))
        self.requests.pop(dataset_id, None)
        self.visibility.pop(dataset_id, None)
        self.appearances.pop(dataset_id, None)

    def close_all(self):
        self.calls.append(("close_all", None))
        self.requests.clear()
        self.visibility.clear()
        self.appearances.clear()

    def is_open(self, dataset_id):
        return dataset_id in self.requests

    def host_bytes(self, dataset_id):
        request = self.requests.get(dataset_id)
        if request is None:
            return 0
        return sum(
            array.nbytes
            for array in (request.coords, request.sigmas, request.values)
            if isinstance(array, np.ndarray)
        )

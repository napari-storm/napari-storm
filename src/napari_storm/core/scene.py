"""The scene/project persistence format.

Level 4 lists this as a deliverable and notes that it "does not exist today".
It is defined here rather than derived from napari's own layer state, for one
decisive reason: napari serialises a layer's *mesh*, and the instanced backend's
mesh is a four-vertex quad standing in for every localization. Saving through
that path would write a quad and call it a reconstruction.

What a scene records is therefore **the decisions**, never the pixels and never
the localizations:

* where each dataset's file is, and where the dataset sits in world space,
* how it is displayed -- colormap, opacity, contrast, visibility, rendering
  style, and whether its traces are connected and how,
* how localizations are turned into Gaussians, and what colours them when
  colour coding,
* where reference images are and how they are placed,
* where the camera was.

Reloading is re-deriving: read the files again, apply the recorded decisions.
That keeps the format small, diffable and readable, and it means a scene never
goes stale against the data it describes -- if a file changed, you see the
change rather than a cached copy of what it used to be.

Nothing here imports a host. A scene can be written from a script, inspected in
a text editor, and read back without a viewer.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace

from .render_planner import GaussianSettings
from .renderer import LayerAppearance
from .world_transform import WorldTransform

__all__ = [
    "SCENE_FORMAT",
    "SCENE_VERSION",
    "CameraState",
    "ColorCodingEntry",
    "DatasetEntry",
    "ReferenceImageEntry",
    "Scene",
    "SceneFormatError",
    "ViewState",
    "drop_edits",
    "edits_of",
    "load_scene",
    "record_edit",
    "resolve_source_path",
    "save_scene",
]

#: Written into every file so a reader can tell what it is holding.
SCENE_FORMAT = "napari-storm-scene"

#: Bumped when the meaning of a field changes.  A reader refuses a *newer*
#: major version rather than guessing at fields it does not know.
SCENE_VERSION = 1

#: Lengths are nanometres throughout, stated in the file rather than assumed --
#: the same rule the OME export follows, and for the same reason.
LENGTH_UNIT = "nm"


class SceneFormatError(ValueError):
    """A file is not a scene, or is a scene this version cannot read."""


@dataclass
class CameraState:
    """Where the viewer was looking.  Cosmetic, and optional."""

    center_nm: tuple = (0.0, 0.0, 0.0)
    zoom: float = 1.0
    angles: tuple = (0.0, 0.0, 90.0)
    ndisplay: int = 3
    #: napari's field of view in degrees; 0 is orthographic.  A scene written
    #: before this was saved reads as 0, which is also napari's default.
    perspective: float = 0.0


@dataclass
class ColorCodingEntry:
    """Colour coding by a property, as the user set it.

    *limits* are the ones the user typed, or None for "Auto" -- not the values
    Auto happened to resolve to when the scene was saved, which would freeze
    a choice the user left to the data.  `Scene.gaussian` carries those, as
    what was drawn.
    """

    color_by: str = None
    colormap: str = "hsv"
    limits: tuple = None


@dataclass
class DatasetEntry:
    """One localization dataset: where it came from, and every decision made.

    The source fields are what reopening needs: *source_path* says which file,
    *source_index* which of the datasets read from it (a molecule set holds one
    per channel), *import_route* how it was opened, and *import_answers* what
    was typed in while it was -- a pixel size the file did not carry, say -- so
    that reopening it does not ask again.
    """

    name: str
    source_path: str = None
    transform: WorldTransform = field(default_factory=WorldTransform)
    appearance: LayerAppearance = field(default_factory=LayerAppearance)
    source_index: int = 0
    import_route: str = "known"
    import_answers: dict = None
    #: What was done to the data after it was read, in order: filter bands and
    #: Data adjustment arithmetic, as recorded by `record_edit`.  Replayed on
    #: the reopened file, so the scene stays a list of decisions rather than a
    #: copy of which rows survived.
    edits: tuple = ()
    #: The channel card as the user left it: ``colormap`` (the channel's own,
    #: which `appearance.colormap` loses while colour coding), ``contrast``
    #: (the slider's two positions), ``color_coding_opacity`` and ``hidden``.
    #: Restored through the card so that the card and the picture agree --
    #: restoring only the picture was undone by the card's next refresh.
    channel: dict = None
    #: *source_path* relative to the scene file's folder, as read from a file;
    #: the fallback when the absolute path is gone -- a folder of data and
    #: scene moved, or opened on another machine.  Not set by hand: it is
    #: derived when saving.
    source_path_relative: str = None


@dataclass
class ReferenceImageEntry:
    """One overlay image, with the placement §7.4 asks to round-trip."""

    name: str
    source_path: str = None
    orientation: str = "XY"
    pixel_size_xy_nm: float = 1.0
    pixel_size_z_nm: float = 1.0
    #: Placement in the layer's ``(z, y, x)`` axis order, matching what a host
    #: reads off ``layer.translate`` rather than a second convention to convert
    #: between.
    offset_nm: tuple = (0.0, 0.0, 0.0)
    appearance: LayerAppearance = field(default_factory=LayerAppearance)
    #: See `DatasetEntry.source_path_relative`.
    source_path_relative: str = None
    #: The layer's own ``scale`` and ``rotate``, in its ``(z, y, x)`` order.
    #: The pixel sizes above say what was typed at import; these say where the
    #: layer ended up, after the scale fields and the 90-degree buttons.
    scale_nm: tuple = None
    rotate: tuple = None


@dataclass
class ViewState:
    """What the canvas shows beyond the datasets themselves.

    *render_range_nm* holds the crop window per axis in world nanometres, not
    the sliders' percentages: a percentage is relative to an extent that
    depends on load history, and the same file reopened can have another.
    ``None`` for an axis means "the full extent".
    """

    render_range_nm: dict = None
    scalebar: bool = False
    scalebar_size_nm: float = None
    grid_plane: bool = False
    grid_line_distance_um: float = None
    grid_margin_percent: float = None
    #: The Decorators tab's sliders, 0..100, as the user left them.
    grid_line_thickness: int = None
    grid_z_position: int = None
    grid_opacity: int = None
    grid_color_index: int = None


@dataclass
class Scene:
    """A whole session's decisions, as a value."""

    datasets: tuple = ()
    reference_images: tuple = ()
    gaussian: GaussianSettings = field(default_factory=GaussianSettings)
    camera: CameraState = field(default_factory=CameraState)
    #: None when the scene says nothing about colour coding -- one written
    #: before it was saved -- so loading it leaves colour coding as it is.
    color_coding: ColorCodingEntry = None
    #: None when the scene predates it, so loading leaves the view alone.
    view: ViewState = None
    #: The folder the scene was read from, which relative source paths are
    #: relative to.  Set by `load_scene`; never written.
    base_dir: str = None

    # ------------------------------------------------------------------

    def to_dict(self, relative_to=None):
        """A plain dict, ready for json.dump.

        With *relative_to* -- the folder the file is being written to -- each
        source path is also written relative to it, which is what lets a
        folder holding a scene and its data be moved or shared and still open.
        """
        return {
            "format": SCENE_FORMAT,
            "version": SCENE_VERSION,
            "length_unit": LENGTH_UNIT,
            "gaussian": asdict(self.gaussian),
            "camera": asdict(self.camera),
            # Written only when set, like the newer appearance fields, so a
            # reader that predates it ignores it and needs no new version.
            **(
                {}
                if self.color_coding is None
                else {"color_coding": _color_coding_to_dict(self.color_coding)}
            ),
            **({} if self.view is None else {"view": _jsonable(asdict(self.view))}),
            "datasets": [
                {
                    "name": entry.name,
                    "source_path": entry.source_path,
                    **_relative_path_field(entry.source_path, relative_to),
                    **_source_fields(entry),
                    "transform": {
                        "scale": list(entry.transform.scale),
                        "translation_nm": list(entry.transform.translation_nm),
                    },
                    "appearance": _appearance_to_dict(entry.appearance),
                }
                for entry in self.datasets
            ],
            "reference_images": [
                {
                    "name": entry.name,
                    "source_path": entry.source_path,
                    **_relative_path_field(entry.source_path, relative_to),
                    "orientation": entry.orientation,
                    "pixel_size_xy_nm": entry.pixel_size_xy_nm,
                    "pixel_size_z_nm": entry.pixel_size_z_nm,
                    "offset_nm": list(entry.offset_nm),
                    **(
                        {}
                        if entry.scale_nm is None
                        else {"scale_nm": _jsonable(entry.scale_nm)}
                    ),
                    **(
                        {}
                        if entry.rotate is None
                        else {"rotate": _jsonable(entry.rotate)}
                    ),
                    "appearance": _appearance_to_dict(entry.appearance),
                }
                for entry in self.reference_images
            ],
        }

    @classmethod
    def from_dict(cls, raw):
        """Rebuild a scene, refusing anything this version cannot honour."""
        if not isinstance(raw, dict):
            raise SceneFormatError("a scene must be a JSON object")
        if raw.get("format") != SCENE_FORMAT:
            raise SceneFormatError(
                f"not a napari-storm scene (format={raw.get('format')!r})"
            )
        version = raw.get("version")
        if not isinstance(version, int):
            raise SceneFormatError(f"missing or invalid version: {version!r}")
        if version > SCENE_VERSION:
            raise SceneFormatError(
                f"this scene is version {version}; this napari-storm reads up to "
                f"{SCENE_VERSION}. Refusing rather than guessing at fields it "
                "does not know."
            )

        return cls(
            datasets=tuple(
                DatasetEntry(
                    name=item.get("name", "dataset"),
                    source_path=item.get("source_path"),
                    source_path_relative=item.get("source_path_relative"),
                    source_index=int(item.get("source_index", 0)),
                    import_route=item.get("import_route", "known"),
                    import_answers=_answers_from_dict(item.get("import_answers")),
                    edits=_edits_from_list(item.get("edits")),
                    channel=_object_or_none(item.get("channel"), "channel"),
                    transform=_transform_from_dict(item.get("transform")),
                    appearance=_appearance_from_dict(item.get("appearance")),
                )
                for item in raw.get("datasets", [])
            ),
            reference_images=tuple(
                ReferenceImageEntry(
                    name=item.get("name", "image"),
                    source_path=item.get("source_path"),
                    source_path_relative=item.get("source_path_relative"),
                    orientation=item.get("orientation", "XY"),
                    pixel_size_xy_nm=float(item.get("pixel_size_xy_nm", 1.0)),
                    pixel_size_z_nm=float(item.get("pixel_size_z_nm", 1.0)),
                    offset_nm=tuple(item.get("offset_nm", (0.0, 0.0, 0.0))),
                    scale_nm=_tuple_or_none(item.get("scale_nm")),
                    rotate=_tuple_or_none(item.get("rotate")),
                    appearance=_appearance_from_dict(item.get("appearance")),
                )
                for item in raw.get("reference_images", [])
            ),
            gaussian=_dataclass_from_dict(GaussianSettings, raw.get("gaussian")),
            camera=_dataclass_from_dict(CameraState, raw.get("camera")),
            color_coding=_color_coding_from_dict(raw),
            view=(
                _dataclass_from_dict(ViewState, raw["view"])
                if isinstance(raw.get("view"), dict)
                else None
            ),
        )


def _colormap_name(colormap):
    """The name a colormap can be rebuilt from, not its repr.

    Hosts hand back rich colormap objects whose `str()` is the whole
    definition -- colours, controls, interpolation mode. Writing that into a
    scene produces a file that cannot be reloaded, because the name is what a
    reader looks up.
    """
    if colormap is None:
        return None
    name = getattr(colormap, "name", None)
    return str(name) if name else str(colormap)


def _appearance_to_dict(appearance):
    raw = {
        "colormap": _colormap_name(appearance.colormap),
        "opacity": appearance.opacity,
        "contrast_limits": (
            None
            if appearance.contrast_limits is None
            else list(appearance.contrast_limits)
        ),
        "visible": appearance.visible,
    }
    # Written only when set, so a scene that never used them reads exactly as
    # it did before they existed -- and a reader that predates them ignores
    # them, which is why they need no new format version.
    if appearance.footprint is not None:
        raw["footprint"] = appearance.footprint
    if appearance.min_size_px is not None:
        raw["min_size_px"] = appearance.min_size_px
    if appearance.traces is not None:
        raw["traces"] = bool(appearance.traces)
    if appearance.trace_color_by is not None:
        raw["trace_color_by"] = appearance.trace_color_by
    if appearance.trace_width_px is not None:
        raw["trace_width_px"] = float(appearance.trace_width_px)
    return raw


def _appearance_from_dict(raw):
    if not isinstance(raw, dict):
        return LayerAppearance()
    limits = raw.get("contrast_limits")
    try:
        return LayerAppearance(
            colormap=raw.get("colormap"),
            opacity=raw.get("opacity"),
            contrast_limits=None if limits is None else tuple(limits),
            visible=raw.get("visible"),
            footprint=raw.get("footprint"),
            min_size_px=raw.get("min_size_px"),
            traces=raw.get("traces"),
            trace_color_by=raw.get("trace_color_by"),
            trace_width_px=raw.get("trace_width_px"),
        )
    except ValueError as exc:
        raise SceneFormatError(f"invalid appearance: {exc}") from None


def _color_coding_to_dict(entry):
    return {
        "color_by": entry.color_by,
        "colormap": entry.colormap,
        "limits": None if entry.limits is None else [float(v) for v in entry.limits],
    }


def _color_coding_from_dict(raw):
    """The scene's colour coding, or None if it does not say.

    Scenes from before colour coding took any property record only
    ``gaussian.z_color_encoding``; true there means colour by z, in hsv,
    scaled to the data.
    """
    block = raw.get("color_coding")
    if isinstance(block, dict):
        limits = block.get("limits")
        if limits is not None:
            if len(limits) != 2:
                raise SceneFormatError(f"colour limits need two values, not {limits!r}")
            limits = (float(limits[0]), float(limits[1]))
        return ColorCodingEntry(
            color_by=block.get("color_by"),
            colormap=block.get("colormap") or "hsv",
            limits=limits,
        )
    gaussian = raw.get("gaussian")
    if isinstance(gaussian, dict) and "z_color_encoding" in gaussian:
        return ColorCodingEntry(color_by="z" if gaussian["z_color_encoding"] else None)
    return None


def _relative_path_field(path, relative_to):
    """``{"source_path_relative": ...}``, or nothing when there is none.

    Windows has no relative path between drives, and a scene written without
    a destination has nothing to be relative to; both just omit the field.
    """
    if not path or relative_to is None:
        return {}
    try:
        relative = os.path.relpath(os.path.abspath(path), os.path.abspath(relative_to))
    except ValueError:
        return {}
    # Forward slashes, so a scene written on Windows reads on Linux and back.
    return {"source_path_relative": relative.replace(os.sep, "/")}


def _source_fields(entry):
    """How a dataset was opened, written only when it says something.

    A single-dataset file opened by type with nothing asked is the common case
    and the default; leaving it out keeps those scenes as they always were.
    """
    raw = {}
    if entry.source_index:
        raw["source_index"] = int(entry.source_index)
    if entry.import_route and entry.import_route != "known":
        raw["import_route"] = entry.import_route
    if entry.import_answers:
        raw["import_answers"] = dict(entry.import_answers)
    if entry.edits:
        raw["edits"] = [_jsonable(dict(edit)) for edit in entry.edits]
    if entry.channel:
        raw["channel"] = _jsonable(dict(entry.channel))
    return raw


def _jsonable(value):
    """*value* with numpy scalars and arrays turned into plain Python.

    Values read off Qt widgets and napari layers arrive as numpy types, which
    `json.dump` refuses.
    """
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):  # numpy array or scalar
        return _jsonable(value.tolist())
    return value


def _tuple_or_none(raw):
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return tuple(
            _tuple_or_none(item) if isinstance(item, list) else item for item in raw
        )
    raise SceneFormatError(f"expected a list, not {raw!r}")


def _object_or_none(raw, what):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SceneFormatError(f"{what} must be an object, not {raw!r}")
    return dict(raw)


#: The edits a scene knows how to replay.  Anything else is refused rather
#: than skipped: a dropped filter would show data the user had removed.
EDIT_KINDS = ("filter", "adjust")


def _edits_from_list(raw):
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise SceneFormatError(f"edits must be a list, not {raw!r}")
    edits = []
    for edit in raw:
        if not isinstance(edit, dict) or edit.get("kind") not in EDIT_KINDS:
            raise SceneFormatError(f"unknown edit {edit!r}")
        edits.append(dict(edit))
    return tuple(edits)


# ---------------------------------------------------------------------------
# The edit log a scene replays
# ---------------------------------------------------------------------------


def record_edit(dataset, edit):
    """Append *edit* to the log a scene saves for *dataset*.

    Kept on the dataset object, so it lives exactly as long as the data it
    describes and goes away when that is unloaded.
    """
    if edit.get("kind") not in EDIT_KINDS:
        raise ValueError(f"unknown edit kind {edit.get('kind')!r}")
    log = getattr(dataset, "scene_edits", None)
    if log is None:
        log = []
        dataset.scene_edits = log
    log.append(dict(edit))


def edits_of(dataset):
    """The edits recorded for *dataset*, oldest first."""
    return tuple(getattr(dataset, "scene_edits", None) or ())


def drop_edits(dataset, kind):
    """Forget every recorded edit of *kind*: what "Reset filters" undoes."""
    log = getattr(dataset, "scene_edits", None)
    if log:
        log[:] = [edit for edit in log if edit.get("kind") != kind]


def _answers_from_dict(raw):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SceneFormatError(f"import answers must be an object, not {raw!r}")
    return {str(key): value for key, value in raw.items()}


def resolve_source_path(entry, base_dir=None):
    """The file *entry* refers to, if it can be found; otherwise None.

    The absolute path first -- it is what was opened -- then the path relative
    to the scene's folder, for a scene that moved together with its data.
    """
    candidates = []
    if entry.source_path:
        candidates.append(entry.source_path)
    relative = getattr(entry, "source_path_relative", None)
    if relative and base_dir:
        candidates.append(os.path.normpath(os.path.join(base_dir, relative)))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _transform_from_dict(raw):
    if not isinstance(raw, dict):
        return WorldTransform()
    return WorldTransform(
        scale=tuple(raw.get("scale", (1.0, 1.0, 1.0))),
        translation_nm=tuple(raw.get("translation_nm", (0.0, 0.0, 0.0))),
    )


def _dataclass_from_dict(cls, raw):
    """Build *cls* from *raw*, ignoring keys it does not have.

    Forward compatibility within a major version: a scene written by a later
    build may carry settings this one has never heard of, and dropping them is
    better than refusing the file.  Fields that *changed meaning* are what the
    version number is for.
    """
    instance = cls()
    if not isinstance(raw, dict):
        return instance
    known = {f for f in instance.__dataclass_fields__}
    return replace(instance, **{k: v for k, v in raw.items() if k in known})


def save_scene(path, scene, indent=2):
    """Write *scene* to *path* as JSON.

    Indented on purpose: a scene is small, and being able to read and diff one
    in a text editor is worth more than the bytes.
    """
    folder = os.path.dirname(os.path.abspath(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            scene.to_dict(relative_to=folder), handle, indent=indent, sort_keys=False
        )
        handle.write("\n")
    return path


def load_scene(path):
    """Read a scene from *path*, or raise :class:`SceneFormatError`."""
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as error:
        raise SceneFormatError(f"{path} is not valid JSON: {error}") from error
    scene = Scene.from_dict(raw)
    scene.base_dir = os.path.dirname(os.path.abspath(path))
    return scene

"""The scene/project persistence format.

Level 4 lists this as a deliverable and notes that it "does not exist today".
It is defined here rather than derived from napari's own layer state, for one
decisive reason: napari serialises a layer's *mesh*, and the instanced backend's
mesh is a four-vertex quad standing in for every localization. Saving through
that path would write a quad and call it a reconstruction.

What a scene records is therefore **the decisions**, never the pixels and never
the localizations:

* where each dataset's file is, and where the dataset sits in world space,
* how it is displayed -- colormap, opacity, contrast, visibility,
* how localizations are turned into Gaussians,
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
from dataclasses import asdict, dataclass, field, replace

from .render_planner import GaussianSettings
from .renderer import LayerAppearance
from .world_transform import WorldTransform

__all__ = [
    "SCENE_FORMAT",
    "SCENE_VERSION",
    "CameraState",
    "DatasetEntry",
    "ReferenceImageEntry",
    "Scene",
    "SceneFormatError",
    "load_scene",
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


@dataclass
class DatasetEntry:
    """One localization dataset: where it came from, and every decision made."""

    name: str
    source_path: str = None
    transform: WorldTransform = field(default_factory=WorldTransform)
    appearance: LayerAppearance = field(default_factory=LayerAppearance)


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


@dataclass
class Scene:
    """A whole session's decisions, as a value."""

    datasets: tuple = ()
    reference_images: tuple = ()
    gaussian: GaussianSettings = field(default_factory=GaussianSettings)
    camera: CameraState = field(default_factory=CameraState)

    # ------------------------------------------------------------------

    def to_dict(self):
        """A plain dict, ready for json.dump."""
        return {
            "format": SCENE_FORMAT,
            "version": SCENE_VERSION,
            "length_unit": LENGTH_UNIT,
            "gaussian": asdict(self.gaussian),
            "camera": asdict(self.camera),
            "datasets": [
                {
                    "name": entry.name,
                    "source_path": entry.source_path,
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
                    "orientation": entry.orientation,
                    "pixel_size_xy_nm": entry.pixel_size_xy_nm,
                    "pixel_size_z_nm": entry.pixel_size_z_nm,
                    "offset_nm": list(entry.offset_nm),
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
                    transform=_transform_from_dict(item.get("transform")),
                    appearance=_appearance_from_dict(item.get("appearance")),
                )
                for item in raw.get("datasets", [])
            ),
            reference_images=tuple(
                ReferenceImageEntry(
                    name=item.get("name", "image"),
                    source_path=item.get("source_path"),
                    orientation=item.get("orientation", "XY"),
                    pixel_size_xy_nm=float(item.get("pixel_size_xy_nm", 1.0)),
                    pixel_size_z_nm=float(item.get("pixel_size_z_nm", 1.0)),
                    offset_nm=tuple(item.get("offset_nm", (0.0, 0.0, 0.0))),
                    appearance=_appearance_from_dict(item.get("appearance")),
                )
                for item in raw.get("reference_images", [])
            ),
            gaussian=_dataclass_from_dict(GaussianSettings, raw.get("gaussian")),
            camera=_dataclass_from_dict(CameraState, raw.get("camera")),
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
    return {
        "colormap": _colormap_name(appearance.colormap),
        "opacity": appearance.opacity,
        "contrast_limits": (
            None
            if appearance.contrast_limits is None
            else list(appearance.contrast_limits)
        ),
        "visible": appearance.visible,
    }


def _appearance_from_dict(raw):
    if not isinstance(raw, dict):
        return LayerAppearance()
    limits = raw.get("contrast_limits")
    return LayerAppearance(
        colormap=raw.get("colormap"),
        opacity=raw.get("opacity"),
        contrast_limits=None if limits is None else tuple(limits),
        visible=raw.get("visible"),
    )


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
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scene.to_dict(), handle, indent=indent, sort_keys=False)
        handle.write("\n")
    return path


def load_scene(path):
    """Read a scene from *path*, or raise :class:`SceneFormatError`."""
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as error:
        raise SceneFormatError(f"{path} is not valid JSON: {error}") from error
    return Scene.from_dict(raw)

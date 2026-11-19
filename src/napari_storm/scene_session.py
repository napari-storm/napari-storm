"""A dock widget's session, read into a scene and put back -- files included.

`core.scene` defines the file and stays host-free. This is the host side: it
reads every decision off the live widget, and puts a scene back by reopening
the files it names and then restoring, in an order that keeps one restored
setting from undoing another.

Why the order matters, step by step (see `apply`):

1. **Edits** -- filters and Data adjustment arithmetic -- replay first: they
   change which values exist, which everything after depends on.
2. **Alignment** next, since it moves data and widens the extents.
3. **Gaussian mode, then widths.** Switching the mode writes default widths
   into the fields, and colour coding is only allowed in fixed mode.
4. **Colour coding**, then **rendering style and traces**, then each
   dataset's **appearance**.
5. **Render range**, which re-plans every layer -- and each re-plan re-applies
   the channel cards' values to the picture.
6. Therefore **channel cards** after that, with a hidden channel hidden last.
7. **Grid plane** (enabled before its sliders: they act on its layer), and
   **reference images**.
8. **Camera**, after everything that moves it, and the **scalebar** last,
   because it is drawn where the camera is looking.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from .core import (
    CameraState,
    ColorCodingEntry,
    DatasetEntry,
    LayerAppearance,
    ReferenceImageEntry,
    Scene,
    ViewState,
    WorldTransform,
    edits_of,
    resolve_source_path,
)
from .FileToLocalizationDataInterface import (
    IMPORT_ROUTE_CUSTOM,
    IMPORT_ROUTE_RECOGNITION,
)
from .ns_constants import FWHM_TO_SIGMA, standard_colors

__all__ = [
    "ReopenPlan",
    "SceneFile",
    "apply",
    "capture",
    "match_datasets",
    "plan_reopen",
    "reopen_and_apply",
    "unsaved_state",
]

AXES = ("x", "y", "z")

_log = logging.getLogger(__name__)


def _same_file_key(path):
    """A key two spellings of one path share, on any platform."""
    return os.path.normcase(os.path.abspath(str(path)))


def _channel_for(widget, dataset):
    for channel in widget.channel:
        if getattr(channel, "dataset", None) is dataset:
            return channel
    return None


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture(widget):
    """The widget's session as a :class:`~napari_storm.core.Scene`."""
    store = widget.dataset_store
    datasets = []
    for dataset in widget.localization_datasets:
        state = store.state(getattr(dataset, "dataset_id", None))
        channel = _channel_for(widget, dataset)
        datasets.append(
            DatasetEntry(
                name=getattr(dataset, "name", "dataset"),
                source_path=getattr(dataset, "file_path", None),
                source_index=int(getattr(dataset, "source_index", 0) or 0),
                import_route=getattr(dataset, "import_route", None) or "known",
                import_answers=getattr(dataset, "import_answers", None) or None,
                edits=edits_of(dataset),
                channel=None if channel is None else _channel_state(channel),
                transform=state.transform if state else WorldTransform(),
                appearance=state.appearance if state else LayerAppearance(),
            )
        )

    camera = widget.viewer.camera
    return Scene(
        datasets=tuple(datasets),
        reference_images=tuple(_capture_reference_images(widget)),
        gaussian=widget.data_to_layer_itf.gaussian_settings(),
        color_coding=ColorCodingEntry(
            color_by=widget.render_config.color_by,
            colormap=widget.render_config.color_colormap,
            limits=widget.render_config.color_limits,
        ),
        view=_capture_view(widget),
        camera=CameraState(
            center_nm=tuple(float(v) for v in camera.center),
            zoom=float(camera.zoom),
            angles=tuple(float(v) for v in camera.angles),
            ndisplay=int(widget.viewer.dims.ndisplay),
            perspective=float(getattr(camera, "perspective", 0.0) or 0.0),
        ),
    )


def _channel_state(channel):
    return {
        "colormap": channel.Colormap_selector.currentText(),
        "contrast": [float(v) for v in channel.Slider_colormap_range.value()],
        "color_coding_opacity": channel.Slider_opacity.value() / 100.0,
        "hidden": not channel.show_channel_state,
    }


def _capture_view(widget):
    itf = widget.data_to_layer_itf
    config = widget.render_config
    ranges = {}
    for axis in AXES:
        percent = getattr(config, f"range_{axis}_percent", None)
        extent = getattr(itf, f"render_range_{axis}", None)
        if percent is None or extent is None:
            continue
        low, high = float(percent[0]), float(percent[1])
        if low <= 0.0 and high >= 100.0:
            continue  # the whole axis: nothing to restore
        if not np.all(np.isfinite(extent)):
            continue
        ranges[axis] = [float(v) for v in itf.percent_to_absolute(extent, (low, high))]
    return ViewState(
        render_range_nm=ranges or None,
        scalebar=bool(widget.Cscalebar.isChecked()),
        scalebar_size_nm=float(config.scalebar_size_nm),
        grid_plane=bool(widget.Cgrid_plane.isChecked()),
        grid_line_distance_um=float(widget.grid_plane_line_distance_um),
        grid_margin_percent=float(widget.grid_plane_margin_percent),
        grid_line_thickness=int(widget.Sgrid_line_thickness.value()),
        grid_z_position=int(widget.Sgrid_z_pos.value()),
        grid_opacity=int(widget.Sgrid_plane_opacity.value()),
        grid_color_index=int(widget.Bgrid_plane_color.currentIndex()),
    )


def _capture_reference_images(widget):
    images = []
    for controls in widget.image_layer_controls:
        layer = getattr(controls, "_layer", None)
        if layer is None or layer not in widget.viewer.layers:
            continue
        result = getattr(controls, "_result", None)
        source = getattr(result, "file_path", None)
        rgb = bool(getattr(layer, "rgb", False))
        rotate = np.asarray(layer.rotate, dtype=float)
        images.append(
            ReferenceImageEntry(
                name=layer.name,
                source_path=None if not source else os.path.abspath(source),
                orientation=getattr(controls, "_orientation", "XY"),
                # What was typed at import, not read back off the layer: the
                # layer's scale is ordered by orientation, and reading index 1
                # as "xy" was wrong for XZ.
                pixel_size_xy_nm=float(getattr(result, "px_xy_nm", layer.scale[1])),
                pixel_size_z_nm=float(getattr(result, "px_z_nm", layer.scale[0])),
                offset_nm=tuple(float(v) for v in layer.translate),
                scale_nm=tuple(float(v) for v in layer.scale),
                rotate=(
                    None
                    if np.allclose(rotate, np.eye(rotate.shape[0]))
                    else tuple(tuple(float(v) for v in row) for row in rotate)
                ),
                appearance=LayerAppearance(
                    colormap=None if rgb else layer.colormap.name,
                    contrast_limits=(
                        None if rgb else tuple(float(v) for v in layer.contrast_limits)
                    ),
                    opacity=float(layer.opacity),
                    visible=bool(layer.visible),
                ),
            )
        )
    return images


def unsaved_state(widget):
    """What is on screen that a scene cannot record, as sentences.

    Said at save time rather than discovered at load time.
    """
    notes = []
    store = widget.dataset_store
    drifted = [
        dataset.name
        for dataset in widget.localization_datasets
        if getattr(store.state(getattr(dataset, "dataset_id", None)), "drift", None)
        is not None
    ]
    if drifted:
        notes.append(
            "drift correction is not part of a scene ("
            + ", ".join(drifted)
            + "); save it with Post-processing > Save drift… and load it "
            "again after the scene"
        )
    unsourced = [
        dataset.name
        for dataset in widget.localization_datasets
        if not getattr(dataset, "file_path", None)
    ]
    if unsourced:
        notes.append(
            "the scene cannot reopen datasets that did not come from a file: "
            + ", ".join(unsourced)
        )
    return notes


# ---------------------------------------------------------------------------
# Matching scene entries to loaded datasets
# ---------------------------------------------------------------------------


def match_datasets(widget, scene):
    """Pair each scene entry with a loaded dataset.

    By source first -- the same file, and the same dataset within it -- and by
    name only for an entry or a dataset that has no source. Names alone are
    fragile: they come from file names and get a suffix when taken, so two
    channels or two files can trade them.

    Returns ``(pairs, unmatched_names)``.
    """
    loaded = list(widget.localization_datasets)
    by_source = {}
    for dataset in loaded:
        path = getattr(dataset, "file_path", None)
        if path:
            key = (_same_file_key(path), int(getattr(dataset, "source_index", 0) or 0))
            by_source.setdefault(key, dataset)
    by_name = {}
    for dataset in loaded:
        by_name.setdefault(getattr(dataset, "name", None), dataset)

    pairs, unmatched, taken = [], [], set()
    for entry in scene.datasets:
        dataset = None
        path = resolve_source_path(entry, scene.base_dir)
        if path is not None:
            dataset = by_source.get((_same_file_key(path), int(entry.source_index)))
        if dataset is None:
            candidate = by_name.get(entry.name)
            # A dataset that has a source of its own is someone else's match.
            if candidate is not None and not (
                path is not None and getattr(candidate, "file_path", None)
            ):
                dataset = candidate
        if dataset is None or id(dataset) in taken:
            unmatched.append(entry.name)
            continue
        taken.add(id(dataset))
        pairs.append((entry, dataset))
    return pairs, unmatched


# ---------------------------------------------------------------------------
# Reopening
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneFile:
    """One file to open, the way it was opened before."""

    path: str
    route: str = "known"
    answers: dict = None


@dataclass
class ReopenPlan:
    """What loading a scene will open, and what it cannot find."""

    files: list = field(default_factory=list)
    #: Names of entries whose file is at neither its saved path nor beside
    #: the scene.
    missing: list = field(default_factory=list)

    @property
    def needed(self):
        return bool(self.files)


def plan_reopen(widget, scene):
    """Which files reloading *scene* has to open.

    Nothing, when every dataset it names is loaded already: the scene is then
    applied in place, as it always was. Otherwise every file it names, in the
    order they were opened, into a fresh session -- reopening only the missing
    ones would leave the session in another order, with other names, beside
    whatever unrelated data happened to be open.
    """
    _pairs, unmatched = match_datasets(widget, scene)
    missing = [
        entry.name
        for entry in scene.datasets
        if resolve_source_path(entry, scene.base_dir) is None
    ]
    reopenable = [name for name in unmatched if name not in missing]
    if not reopenable:
        return ReopenPlan(files=[], missing=missing)

    files, seen = [], set()
    for entry in scene.datasets:
        path = resolve_source_path(entry, scene.base_dir)
        if path is None:
            continue
        key = _same_file_key(path)
        if key in seen:
            continue
        seen.add(key)
        files.append(SceneFile(path, entry.import_route, entry.import_answers))
    return ReopenPlan(files=files, missing=missing)


def reopen_and_apply(widget, scene, plan, background=False, on_done=None):
    """Open *plan*'s files one after another, then apply *scene*.

    The first file replaces the session; the rest are merged into it. One
    file failing does not stop the others -- it is reported, and the scene is
    applied to whatever did open. A cancel stops the sequence.

    With *background* the files are read behind the usual cancellable
    progress dialog and this returns at once; *on_done* is called with the
    unmatched names when the scene has been applied.
    """
    files = list(plan.files)
    opened = {"any": False}

    def finish():
        if plan.missing:
            widget._warn_user(
                "the scene's files for "
                + ", ".join(plan.missing)
                + " were not found at their saved paths or beside the scene"
            )
        fresh = {id(dataset) for dataset in widget.localization_datasets}
        unmatched = apply(widget, scene, fresh=fresh)
        if on_done is not None:
            on_done(unmatched)

    def step(index):
        if index >= len(files):
            finish()
            return
        scene_file = files[index]
        options = {
            "merge": opened["any"],
            "file_path": scene_file.path,
            "file_recognition": scene_file.route == IMPORT_ROUTE_RECOGNITION,
            "custom_import": scene_file.route == IMPORT_ROUTE_CUSTOM,
            "metadata_answers": scene_file.answers,
        }

        def loaded(_datasets=None):
            opened["any"] = True
            step(index + 1)

        if background:
            widget.open_localization_data_file_and_get_dataset(
                background=True,
                on_loaded=loaded,
                on_failed=lambda _error=None: step(index + 1),
                on_cancelled=lambda: widget._warn_user(
                    "loading the scene was cancelled"
                ),
                **options,
            )
            return
        try:
            ok = widget.open_localization_data_file_and_get_dataset(**options)
        except Exception as error:  # noqa: BLE001 - one file must not stop the rest
            widget.file_to_data_itf.sync_dataset_entries(widget.localization_datasets)
            widget._warn_user(f"could not open {scene_file.path}: {error}")
            ok = False
        if ok:
            loaded()
        else:
            step(index + 1)

    step(0)


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def apply(widget, scene, fresh=()):
    """Put *scene* back onto the datasets loaded now.

    *fresh* holds the ``id()`` of datasets just opened by the scene itself:
    their recorded edits are replayed. On datasets that were already open
    they are not -- replaying an adjustment twice would apply it twice -- and
    a difference is reported instead.

    Returns the names of scene entries that matched no loaded dataset.
    """
    pairs, unmatched = match_datasets(widget, scene)

    _replay_edits(widget, pairs, fresh)

    for entry, dataset in pairs:
        dataset_id = getattr(dataset, "dataset_id", None)
        if dataset_id is None:
            continue
        transform = entry.transform or WorldTransform()
        widget.dataset_store.set_transform(dataset_id, transform)
        _sync_shift_spins(_channel_for(widget, dataset), transform)

    _restore_gaussian(widget, scene.gaussian)
    # Before the per-dataset appearance: switching colour coding resets each
    # channel's contrast, which the scene's own contrast then replaces.
    widget._restore_color_coding(scene.color_coding)
    _restore_appearance(widget, pairs)

    if scene.view is not None:
        _restore_render_range(widget, scene.view)

    for entry, dataset in pairs:
        _restore_channel(_channel_for(widget, dataset), entry.channel)

    if scene.view is not None:
        _restore_grid(widget, scene.view)
    _restore_reference_images(widget, scene)
    _restore_camera(widget, scene.camera)
    if scene.view is not None:
        _restore_scalebar(widget, scene.view)

    if unmatched:
        widget._warn_user(
            "the scene refers to datasets that are not loaded: "
            + ", ".join(unmatched)
            + ". Open them and load the scene again to place them."
        )
    return unmatched


def _replay_edits(widget, pairs, fresh):
    changed = False
    differs = []
    for entry, dataset in pairs:
        if not entry.edits:
            continue
        if id(dataset) not in fresh:
            if tuple(entry.edits) != edits_of(dataset):
                differs.append(entry.name)
            continue
        for edit in entry.edits:
            try:
                if edit["kind"] == "adjust":
                    _replay_adjustment(dataset, edit)
                else:
                    widget.data_filter_itf.replay_band(dataset, edit)
            except (KeyError, ValueError, AttributeError) as error:
                widget._warn_user(
                    f"could not replay {edit.get('kind')} on {dataset.name}: {error}"
                )
                continue
            changed = True
    if changed:
        widget.data_to_layer_itf.set_render_range_and_offset()
        widget.data_to_layer_itf.update_layers()
    if differs:
        widget._warn_user(
            "the scene's filters and adjustments differ from those on the "
            "loaded data for " + ", ".join(differs) + "; they were not "
            "re-applied. Close the data and load the scene to reopen it as "
            "saved."
        )


def _replay_adjustment(dataset, edit):
    from .core import record_edit

    column = edit["column"]
    if "offset" in edit:
        dataset.adjust_column(column, offset=float(edit["offset"]))
    elif "scale" in edit:
        dataset.adjust_column(column, scale=float(edit["scale"]))
    else:
        raise ValueError("an adjustment needs an offset or a scale")
    record_edit(dataset, edit)


def _sync_shift_spins(channel, transform):
    """The Shift boxes, showing the restored alignment.

    Otherwise the next edit of one axis writes the boxes' stale values for
    the other two back over it.
    """
    spins = getattr(channel, "_shift_spins", None)
    if not spins:
        return
    for axis, value in zip(("x", "y", "z"), transform.translation_nm):
        spin = spins.get(axis)
        if spin is None:
            continue
        spin.blockSignals(True)
        spin.setValue(float(value) / 1000.0)
        spin.blockSignals(False)


def _fwhm_text(sigma_nm):
    return f"{float(sigma_nm) * FWHM_TO_SIGMA:.6g}"


def _restore_gaussian(widget, settings):
    """Mode first, then the widths into both the config and the fields."""
    if settings is None:
        return
    config = widget.render_config
    mode = int(settings.mode)
    if widget.Brenderoptions.currentIndex() != mode:
        # Runs _render_options_changed, which refuses variable mode for data
        # without uncertainties and writes default widths into the fields.
        widget.Brenderoptions.setCurrentIndex(mode)

    for name in (
        "fixed_sigma_xy_nm",
        "fixed_sigma_z_nm",
        "var_psf_sigma_xy_nm",
        "var_psf_sigma_z_nm",
        "var_sigma_min_xy_nm",
        "var_sigma_min_z_nm",
    ):
        value = getattr(settings, name, None)
        if value is not None:
            setattr(config, name, float(value))

    if config.gaussian_mode == 0:
        fields = (
            (widget.Esigma_xy, config.fixed_sigma_xy_nm),
            (widget.Esigma_z, config.fixed_sigma_z_nm),
        )
    else:
        fields = (
            (widget.Esigma_xy, config.var_psf_sigma_xy_nm),
            (widget.Esigma_z, config.var_psf_sigma_z_nm),
            (widget.Esigma_min_xy, config.var_sigma_min_xy_nm),
            (widget.Esigma_min_z, config.var_sigma_min_z_nm),
        )
    for line_edit, sigma in fields:
        line_edit.blockSignals(True)
        line_edit.setText(_fwhm_text(sigma))
        line_edit.blockSignals(False)
    widget.typing_timer_sigma.stop()
    widget.update_sigma()


def _restore_appearance(widget, pairs):
    restored_style = {}
    restored_traces = {}
    for entry, dataset in pairs:
        appearance = entry.appearance
        if appearance is None:
            continue
        widget.data_to_layer_itf.set_appearance(
            dataset,
            colormap=appearance.colormap,
            opacity=appearance.opacity,
            contrast_limits=appearance.contrast_limits,
            visible=appearance.visible,
        )
        if appearance.footprint is not None:
            restored_style["footprint"] = appearance.footprint
        if appearance.min_size_px is not None:
            restored_style["min_size_px"] = appearance.min_size_px
        # One trace style for the session, like the footprint: any dataset
        # that drew traces turns them on.
        if appearance.traces is not None:
            restored_traces["traces"] = bool(
                restored_traces.get("traces") or appearance.traces
            )
        if appearance.trace_color_by is not None:
            restored_traces["trace_color_by"] = appearance.trace_color_by
        if appearance.trace_width_px is not None:
            restored_traces["trace_width_px"] = appearance.trace_width_px
    if restored_style:
        widget._restore_rendering_style(**restored_style)
    if restored_traces:
        widget._restore_trace_style(**restored_traces)


def _restore_render_range(widget, view):
    """The crop window, converted back from nanometres to slider percent."""
    itf = widget.data_to_layer_itf
    windows = view.render_range_nm or {}
    sliders = {
        "x": widget.Srender_rangex,
        "y": widget.Srender_rangey,
        "z": widget.Srender_rangez,
    }
    for axis in AXES:
        if axis == "z" and not widget.zdim:
            continue
        extent = getattr(itf, f"render_range_{axis}", None)
        window = windows.get(axis)
        percent = (0.0, 100.0)
        if window is not None and extent is not None and np.all(np.isfinite(extent)):
            low, high = float(extent[0]), float(extent[1])
            span = high - low
            if span > 0:
                percent = tuple(
                    float(np.clip((float(v) - low) / span * 100.0, 0.0, 100.0))
                    for v in window
                )
        setattr(widget.render_config, f"range_{axis}_percent", np.asarray(percent))
        slider = sliders[axis]
        maximum = slider.maximum()
        shown = (
            # The X slider is drawn inverted; see RenderRangeSlider._apply_range.
            (maximum - percent[1], maximum - percent[0])
            if axis == "x"
            else percent
        )
        slider.blockSignals(True)
        slider.setValue(shown)
        slider.blockSignals(False)
    itf.update_layers()
    if widget.render_config.color_by in AXES:
        widget._refresh_color_bar()


def _restore_channel(channel, state):
    if channel is None or not state:
        return
    name = state.get("colormap")
    if name:
        index = channel.Colormap_selector.findText(str(name))
        if index >= 0 and index != channel.Colormap_selector.currentIndex():
            channel.Colormap_selector.setCurrentIndex(index)  # applies it

    contrast = state.get("contrast")
    if contrast is not None and len(contrast) == 2:
        values = (float(contrast[0]), float(contrast[1]))
        channel.Slider_colormap_range.blockSignals(True)
        channel.Slider_colormap_range.setValue(values)
        channel.Slider_colormap_range.blockSignals(False)
        channel._on_contrast_slider_changed(channel.Slider_colormap_range.value())

    opacity = state.get("color_coding_opacity")
    if opacity is not None:
        channel.Slider_opacity.blockSignals(True)
        channel.Slider_opacity.setValue(int(round(float(opacity) * 100)))
        channel.Slider_opacity.blockSignals(False)
        if channel.color_coding_active:
            channel.adjust_color_coding_opacity()

    hidden = bool(state.get("hidden", False))
    if hidden == channel.show_channel_state:  # i.e. it differs: toggle
        channel.show_channel()


def _restore_grid(widget, view):
    itf = widget.data_to_layer_itf
    if view.grid_line_distance_um is not None:
        widget.grid_plane_line_distance_um = float(view.grid_line_distance_um)
        widget.Egrid_line_distance.blockSignals(True)
        widget.Egrid_line_distance.setText(f"{view.grid_line_distance_um:g}")
        widget.Egrid_line_distance.blockSignals(False)
    if view.grid_margin_percent is not None:
        widget.grid_plane_margin_percent = float(view.grid_margin_percent)
        widget.Egrid_margin.blockSignals(True)
        widget.Egrid_margin.setText(f"{view.grid_margin_percent:g}")
        widget.Egrid_margin.blockSignals(False)
    color_index = view.grid_color_index
    if color_index is not None and 0 <= color_index < len(standard_colors):
        # Not through the combo's signal: with the grid off, that reaches for
        # a layer which does not exist.
        widget.Bgrid_plane_color.blockSignals(True)
        widget.Bgrid_plane_color.setCurrentIndex(color_index)
        widget.Bgrid_plane_color.blockSignals(False)
        widget.grid_plane_standard_color_index = color_index
        itf.current_grid_plane_color = standard_colors[color_index]

    if widget.Cgrid_plane.isChecked() != bool(view.grid_plane):
        widget.Cgrid_plane.setChecked(bool(view.grid_plane))
    elif view.grid_plane:
        itf.update_grid_plane(
            line_distance_nm=widget.grid_plane_line_distance_um * 1000
        )
    if not view.grid_plane:
        return
    if color_index is not None and 0 <= color_index < len(standard_colors):
        itf.update_grid_plane(color=standard_colors[color_index])
    for slider, value in (
        (widget.Sgrid_line_thickness, view.grid_line_thickness),
        (widget.Sgrid_z_pos, view.grid_z_position),
        (widget.Sgrid_plane_opacity, view.grid_opacity),
    ):
        if value is None:
            continue
        if slider.value() != int(value):
            slider.setValue(int(value))  # applies through its own signal
        else:
            slider._apply_value()


def _restore_reference_images(widget, scene):
    missing = []
    for entry in scene.reference_images:
        controls = next(
            (
                c
                for c in widget.image_layer_controls
                if getattr(c, "_layer", None) is not None
                and c._layer.name == entry.name
                and c._layer in widget.viewer.layers
            ),
            None,
        )
        if controls is None:
            controls = _reopen_reference_image(widget, scene, entry, missing)
            if controls is None:
                continue
        _place_reference_image(controls, entry)
    if missing:
        widget._warn_user(
            "the scene's reference images were not found: " + ", ".join(missing)
        )


def _reopen_reference_image(widget, scene, entry, missing):
    from .pyqt.image_import_dialog import ImageImportResult, _load_image

    path = resolve_source_path(entry, scene.base_dir)
    if path is None:
        missing.append(entry.name)
        return None
    try:
        image = _load_image(path)
    except Exception as error:  # noqa: BLE001 - reported, not raised
        widget._warn_user(f"could not open reference image {path}: {error}")
        return None
    return widget._add_image_layer_from_result(
        ImageImportResult(
            file_path=path,
            img=image,
            layer_name=entry.name,
            orientation=entry.orientation,
            px_xy_nm=float(entry.pixel_size_xy_nm),
            px_z_nm=float(entry.pixel_size_z_nm),
            x_off_nm=0.0,
            y_off_nm=0.0,
            z_off_nm=0.0,
        )
    )


def _place_reference_image(controls, entry):
    """Set the layer directly; its spin boxes use another axis order."""
    layer = controls._layer
    if entry.scale_nm is not None:
        layer.scale = tuple(float(v) for v in entry.scale_nm)
    if entry.rotate is not None:
        layer.rotate = np.asarray(entry.rotate, dtype=float)
    layer.translate = tuple(float(v) for v in entry.offset_nm)
    appearance = entry.appearance or LayerAppearance()
    if not getattr(layer, "rgb", False):
        if appearance.colormap:
            try:
                layer.colormap = appearance.colormap
            except (KeyError, ValueError) as error:
                _log.info("napari-storm: colormap %s: %s", appearance.colormap, error)
        if appearance.contrast_limits is not None:
            layer.contrast_limits = tuple(appearance.contrast_limits)
    if appearance.opacity is not None:
        layer.opacity = float(appearance.opacity)
    if appearance.visible is not None:
        layer.visible = bool(appearance.visible)
    controls._sync_from_layer()


def _restore_camera(widget, camera):
    if camera is None:
        return
    viewer = widget.viewer
    viewer.dims.ndisplay = int(camera.ndisplay)
    viewer.camera.center = tuple(camera.center_nm)
    viewer.camera.zoom = float(camera.zoom)
    viewer.camera.angles = tuple(camera.angles)
    viewer.camera.perspective = float(camera.perspective or 0.0)
    # The view buttons and the `r` key return to these, not to the camera.
    widget.data_to_layer_itf.camera = [
        viewer.camera.zoom,
        viewer.camera.center,
        viewer.camera.angles,
    ]
    widget.zoom = viewer.camera.zoom


def _restore_scalebar(widget, view):
    if view.scalebar_size_nm:
        widget.Esbsize.blockSignals(True)
        widget.Esbsize.setText(str(int(view.scalebar_size_nm)))
        widget.Esbsize.blockSignals(False)
    if widget.Cscalebar.isChecked() != bool(view.scalebar):
        widget.Cscalebar.setChecked(bool(view.scalebar))
    elif view.scalebar:
        # Already on: redraw it where the restored camera is looking.
        widget._sync_scalebar_config()
    widget.typing_timer_sbscale.stop()

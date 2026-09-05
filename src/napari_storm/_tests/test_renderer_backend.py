"""The seam between deciding what to draw and drawing it.

Level 3 requires prototyping a second renderer "on the same fixtures before
selecting the production backend", which needs a contract both implementations
satisfy. These tests pin that contract, and check the napari backend against it.
"""

import numpy as np
import pytest

from napari_storm._dock_widget import napari_storm
from napari_storm.core.renderer import (
    Changed,
    LayerAppearance,
    LocalizationRenderer,
    NullRenderer,
    RenderRequest,
)
from napari_storm.napari_particles.instanced_renderer import InstancedRenderer
from napari_storm.napari_particles.points_renderer import NapariPointsRenderer
from napari_storm.napari_particles.renderer import NapariParticlesRenderer

#: Every Level 3 candidate, including the one that won.  Every contract test
#: below runs against all of them, which is the only thing that makes "swap the
#: backend" more than an aspiration -- and the gate the instanced backend had to
#: pass before it could become the default.
BACKENDS = [NapariParticlesRenderer, NapariPointsRenderer, InstancedRenderer]


def _request(n=8, name="layer", size=10.0):
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 1] = np.arange(n)
    coords[:, 2] = np.arange(n) * 2
    return RenderRequest(
        coords=coords,
        sigmas=np.ones((n, 3), dtype=np.float32),
        size=size,
        values=np.ones(n, dtype=np.float32),
        name=name,
    )


# --------------------------------------------------------------- the contract


def test_the_base_class_refuses_to_pretend_it_renders():
    """An incomplete backend must fail loudly, not silently draw nothing."""
    renderer = LocalizationRenderer()
    for call in (
        lambda: renderer.open(1, _request()),
        lambda: renderer.update(1, _request()),
        lambda: renderer.set_visible(1, True),
        lambda: renderer.close(1),
        lambda: renderer.close_all(),
        lambda: renderer.is_open(1),
    ):
        with pytest.raises(NotImplementedError):
            call()


def test_null_renderer_records_the_lifecycle():
    renderer = NullRenderer()
    assert not renderer.is_open(1)

    renderer.open(1, _request(name="a"))
    assert renderer.is_open(1)
    assert renderer.requests[1].name == "a"
    assert renderer.visibility[1] is True

    renderer.update(1, _request(name="a", size=20.0))
    assert renderer.requests[1].size == 20.0

    renderer.set_visible(1, False)
    assert renderer.visibility[1] is False

    renderer.close(1)
    assert not renderer.is_open(1)


def test_updating_something_that_was_never_opened_is_an_error():
    """Silently opening instead would hide a lifecycle bug in a backend."""
    renderer = NullRenderer()
    with pytest.raises(KeyError):
        renderer.update(1, _request())


def test_close_all_releases_every_dataset():
    renderer = NullRenderer()
    renderer.open(1, _request())
    renderer.open(2, _request())
    renderer.close_all()
    assert renderer.requests == {}


def test_closing_something_absent_is_a_no_op():
    NullRenderer().close(404)


# ------------------------------------------------------------ napari backend


def test_open_adds_a_layer_and_hands_back_a_handle(make_napari_viewer):
    viewer = make_napari_viewer()
    renderer = NapariParticlesRenderer(viewer)

    layer = renderer.open(7, _request(name="channel"))

    assert renderer.layer(7) is layer
    assert renderer.is_open(7)
    assert layer in viewer.layers
    assert layer.name == "channel"
    # The Gaussian shader goes on our own attribute, never napari's `shading`.
    assert layer.shader == "gaussian"
    assert layer.shading == "none"


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_update_keeps_the_layer(make_napari_viewer, backend_class):
    """The contract: an update must not replace the host resource."""
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    layer = renderer.open(7, _request(n=8))
    n_layers = len(viewer.layers)

    renderer.update(7, _request(n=3))

    assert renderer.layer(7) is layer
    assert len(viewer.layers) == n_layers
    assert len(layer.data) in (3, 18)  # points, or six vertices per billboard


def test_a_billboard_update_keeps_the_vispy_visual(make_napari_viewer):
    """Backend-specific: the expensive thing the billboard backend must retain."""
    viewer = make_napari_viewer()
    renderer = NapariParticlesRenderer(viewer)
    layer = renderer.open(7, _request(n=8))
    visual = layer._visual

    renderer.update(7, _request(n=3))

    assert layer._visual is visual
    assert len(layer._coords) == 3


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_update_before_open_is_an_error(make_napari_viewer, backend_class):
    renderer = backend_class(make_napari_viewer())
    with pytest.raises(KeyError):
        renderer.update(7, _request())


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_reopening_replaces_rather_than_accumulates(make_napari_viewer, backend_class):
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    first = renderer.open(7, _request())
    second = renderer.open(7, _request())

    assert second is not first
    assert first not in viewer.layers
    assert len(renderer.layers) == 1


def test_close_removes_the_layer_and_releases_what_it_installed(make_napari_viewer):
    """Closing must leave the viewer as it found it.

    This used to check that a layer-list callback count went *down*, because the
    layer kept additive blending alive by subscribing to inserted/removed/
    reordered. It no longer subscribes -- the blend function is forced through
    the visual's own `set_gl_state`, which catches triggers those three events
    missed -- so what has to be given back on close is the setter.
    """
    from napari_storm.napari_particles._napari_compat import _FORCED_BLENDING

    viewer = make_napari_viewer()
    renderer = NapariParticlesRenderer(viewer)
    layer = renderer.open(7, _request())
    visual = layer._visual
    assert visual in _FORCED_BLENDING

    renderer.close(7)

    assert not renderer.is_open(7)
    assert layer not in viewer.layers
    assert visual not in _FORCED_BLENDING


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_set_visible_does_not_discard_resources(make_napari_viewer, backend_class):
    renderer = backend_class(make_napari_viewer())
    layer = renderer.open(7, _request())

    renderer.set_visible(7, False)
    assert layer.visible is False
    assert renderer.layer(7) is layer

    renderer.set_visible(7, True)
    assert layer.visible is True


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_visibility_on_an_absent_dataset_is_a_no_op(make_napari_viewer, backend_class):
    backend_class(make_napari_viewer()).set_visible(404, False)


# --------------------------------------------------------------- injection


def test_the_widget_uses_the_instanced_backend_by_default(make_napari_viewer):
    """The measured decision, actually taken.

    `backend-comparison.md` settled Level 3's backend question -- 28 B per
    localization against 352, and 0.16 s against 2.47 s for one update at 5M --
    but the plugin went on constructing the renderer it had measured against.
    A decision the default does not follow is a document, not a decision.
    """
    widget = napari_storm(napari_viewer=make_napari_viewer())
    assert isinstance(widget.data_to_layer_itf.renderer, InstancedRenderer)


def test_the_billboard_backend_takes_over_when_instancing_is_unavailable(
    make_napari_viewer, monkeypatch
):
    """A session without `gl+` must still draw, and must say why it is slower.

    napari selects `gl+` itself for its own instanced Points, so in practice
    this holds -- but it falls back to `gl2` with a warning when PyOpenGL is
    missing, and `gl2` has no `glDrawElementsInstanced` at all. Discovering that
    at the first draw call would mean an exception instead of an image.
    """
    from napari_storm.napari_particles import selection

    monkeypatch.setattr(selection, "instancing_available", lambda: False)
    renderer_class, reason = selection.renderer_class_for_session()

    assert renderer_class is NapariParticlesRenderer
    assert "gl+" in reason

    widget = napari_storm(napari_viewer=make_napari_viewer())
    assert isinstance(widget.data_to_layer_itf.renderer, NapariParticlesRenderer)


def test_the_fallback_warns_through_napari(make_napari_viewer, monkeypatch):
    """Silently running 12x heavier would be the worst version of this."""
    from napari_storm.napari_particles import selection

    monkeypatch.setattr(selection, "instancing_available", lambda: False)
    warnings = []
    monkeypatch.setattr("napari.utils.notifications.show_warning", warnings.append)

    selection.select_renderer(make_napari_viewer())

    assert len(warnings) == 1
    assert "Instanced rendering is unavailable" in warnings[0]


def test_a_selected_renderer_still_draws(make_napari_viewer):
    """Whatever the selection returns has to satisfy the contract."""
    from napari_storm.napari_particles.selection import select_renderer

    viewer = make_napari_viewer()
    renderer = select_renderer(viewer)
    layer = renderer.open(7, _request())

    assert renderer.is_open(7)
    assert layer in viewer.layers


def test_a_backend_can_be_injected(make_napari_viewer):
    """The Level 3 prototype gate: swap the backend, keep the planner."""
    from napari_storm.DataToLayerInterface import DataToLayerInterface

    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    injected = NullRenderer()
    interface = DataToLayerInterface(
        parent=widget,
        viewer=viewer,
        render_config=widget.render_config,
        renderer=injected,
    )
    assert interface.renderer is injected


# --------------------------------------------------------- host lifecycle


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_a_layer_the_user_deletes_stops_being_open(make_napari_viewer, backend_class):
    """napari's layer list is the user's; they can delete our layer any time."""
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    removed = []
    renderer.on_layer_removed_by_host = removed.append
    layer = renderer.open(7, _request())

    viewer.layers.remove(layer)

    assert not renderer.is_open(7)
    assert removed == [7]


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_our_own_close_does_not_look_like_a_user_deletion(
    make_napari_viewer, backend_class
):
    """close() removes the layer too; that must not re-enter as a host event."""
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    removed = []
    renderer.on_layer_removed_by_host = removed.append
    renderer.open(7, _request())

    renderer.close(7)

    assert removed == []
    assert not renderer.is_open(7)


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_detach_stops_the_renderer_listening(make_napari_viewer, backend_class):
    viewer = make_napari_viewer()
    renderer = backend_class(viewer)
    removed = []
    renderer.on_layer_removed_by_host = removed.append
    layer = renderer.open(7, _request())

    renderer.detach()
    viewer.layers.remove(layer)

    assert removed == []


def test_deleting_a_layer_in_napari_unloads_the_dataset(make_napari_viewer):
    """The session must not keep half a dataset whose layer is gone."""
    from napari_storm._tests.test_data_filter import _dataset as _plain_dataset

    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode(
        [_plain_dataset(range(10), "a"), _plain_dataset(range(10), "b")]
    )
    doomed, kept = widget.localization_datasets
    doomed_id = doomed.dataset_id

    viewer.layers.remove(widget.data_to_layer_itf.layer_for(doomed))

    assert widget.localization_datasets == [kept]
    assert widget.n_datasets == 1
    assert doomed_id not in widget.data_to_layer_itf.render_state
    assert len(widget.channel) == 1


def test_closing_the_dock_releases_the_viewer(make_napari_viewer):
    from napari_storm._tests.test_data_filter import _dataset as _plain_dataset

    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer)
    widget.get_dataset_from_test_mode([_plain_dataset(range(10))])
    store = widget.dataset_store

    widget.close_session()

    assert len(store) == 0
    assert widget.data_to_layer_itf.render_state == {}
    assert store._listeners == []
    # And a viewer event arriving afterwards must not resurrect anything.
    widget.close_session()
    assert len(store) == 0


# ------------------------------------------- the substitute must actually work


def test_the_null_backend_can_run_the_application(make_napari_viewer):
    """A seam you cannot substitute across is not a seam.

    The whole point of the protocol is that Level 3 can drive the real
    application with a different backend. This loads data, filters it and
    drives the channel controls with a backend that owns no napari layer at
    all -- if anything in the application still reached through to one, this
    would raise.
    """
    from napari_storm._tests.test_data_filter import _dataset as _plain_dataset

    viewer = make_napari_viewer()
    backend = NullRenderer()
    widget = napari_storm(napari_viewer=viewer, renderer=backend)

    dataset = _plain_dataset(range(20), "headless")
    widget.get_dataset_from_test_mode([dataset])

    # Drawn through the protocol, and no napari layer was created for it.
    assert backend.is_open(dataset.dataset_id)
    assert len(backend.requests[dataset.dataset_id].coords) == 20
    assert not any(layer.name == "headless" for layer in viewer.layers)

    # Appearance: the channel control drives the backend, not a layer.
    controls = widget.channel[0]
    controls.Slider_opacity.setValue(40)
    assert backend.appearance(dataset.dataset_id).opacity == pytest.approx(0.4)

    controls.show_channel()
    assert backend.appearance(dataset.dataset_id).opacity == 0.0

    # Filtering: the selection changes and the backend is updated, not reopened.
    opens_before = backend.calls.count(("open", dataset.dataset_id))
    widget.render_config.range_x_percent = np.array([0, 50])
    widget.data_to_layer_itf.update_layers()

    assert backend.calls.count(("open", dataset.dataset_id)) == opens_before
    assert ("update", dataset.dataset_id) in backend.calls
    assert len(backend.requests[dataset.dataset_id].coords) < 20


def test_a_request_carries_the_ids_of_what_it_draws(make_napari_viewer):
    """A backend with persistent per-localization storage needs these."""
    from napari_storm._tests.test_data_filter import _dataset as _plain_dataset

    backend = NullRenderer()
    widget = napari_storm(napari_viewer=make_napari_viewer(), renderer=backend)
    dataset = _plain_dataset(range(20))
    widget.get_dataset_from_test_mode([dataset])

    widget.render_config.range_x_percent = np.array([0, 50])
    widget.data_to_layer_itf.update_layers()

    request = backend.requests[dataset.dataset_id]
    assert request.active_ids is not None
    assert len(request.active_ids) == len(request.coords)
    # Stable canonical row indices, so the backend can diff against last time.
    assert request.active_ids.tolist() == list(range(len(request.coords)))


def test_updates_declare_what_actually_changed(make_napari_viewer):
    """Level 3 cannot compare targeted updates if every update says EVERYTHING."""
    from napari_storm._tests.test_data_filter import _dataset as _plain_dataset

    backend = NullRenderer()
    widget = napari_storm(napari_viewer=make_napari_viewer(), renderer=backend)
    dataset = _plain_dataset(range(20))
    widget.get_dataset_from_test_mode([dataset])

    widget.data_to_layer_itf.update_layer_appearance()
    appearance_only = backend.requests[dataset.dataset_id].changed
    assert appearance_only == (Changed.SIGMAS | Changed.VALUES)
    assert not (appearance_only & Changed.POSITIONS)

    widget.data_to_layer_itf.update_layers()
    assert backend.requests[dataset.dataset_id].changed == Changed.EVERYTHING


def test_appearance_leaves_unspecified_fields_alone():
    backend = NullRenderer()
    backend.open(1, _request())
    backend.set_appearance(1, LayerAppearance(opacity=0.25))
    backend.set_appearance(1, LayerAppearance(contrast_limits=(0.0, 2.0)))

    appearance = backend.appearance(1)
    assert appearance.opacity == 0.25
    assert appearance.contrast_limits == (0.0, 2.0)


def test_appearance_on_a_closed_dataset_is_an_error():
    with pytest.raises(KeyError):
        NullRenderer().set_appearance(404, LayerAppearance(opacity=1.0))


# ------------------------------------------------- the Level 3 comparison


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_every_backend_reports_its_own_memory(make_napari_viewer, backend_class):
    """The decision turns on this number, so both must report it the same way."""
    renderer = backend_class(make_napari_viewer())
    assert renderer.host_bytes(7) == 0

    renderer.open(7, _request(n=100))
    per_localization = renderer.host_bytes(7) / 100

    assert per_localization > 0
    renderer.close(7)
    assert renderer.host_bytes(7) == 0


def test_the_points_backend_is_much_smaller_than_the_billboards(
    make_napari_viewer,
):
    """The trade the Level 3 decision is actually about.

    Billboards expand every localization into six vertices with its centre,
    sigma and value repeated across them. Points store a position, a size and
    a colour. What that buys and what it costs is in `docs/backend-comparison`.
    """
    viewer = make_napari_viewer()
    billboards = NapariParticlesRenderer(viewer)
    points = NapariPointsRenderer(viewer)
    request = _request(n=500)

    billboards.open(1, request)
    points.open(2, request)

    assert points.host_bytes(2) < billboards.host_bytes(1) / 3


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_either_backend_can_run_the_application(make_napari_viewer, backend_class):
    """Not just the contract in isolation -- the real widget, both ways."""
    from napari_storm._tests.test_data_filter import _dataset as _plain_dataset

    viewer = make_napari_viewer()
    widget = napari_storm(napari_viewer=viewer, renderer=backend_class(viewer))
    widget.get_dataset_from_test_mode([_plain_dataset(range(20))])
    dataset = widget.localization_datasets[0]

    assert widget.data_to_layer_itf.renderer.is_open(dataset.dataset_id)

    widget.channel[0].Slider_opacity.setValue(50)
    widget.render_config.range_x_percent = np.array([0, 50])
    widget.data_to_layer_itf.update_layers()

    assert widget.data_to_layer_itf.renderer.is_open(dataset.dataset_id)
    assert dataset.number_of_active_entries() < 20

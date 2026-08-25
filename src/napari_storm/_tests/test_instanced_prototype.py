"""The instanced-billboard prototype: does it fit, and does it still look right.

Two questions decide whether instancing is worth building out, and this answers
both. The memory question is arithmetic and runs anywhere. The quality question
needs a GL context and VisPy's ``gl+`` backend, which has to be selected before
any context exists — so those run in a subprocess, for the same reason the
host-free gate does.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from napari_storm.napari_particles.instanced import (
    QUAD, QUAD_INDICES, InstancedBillboards, instanced_bytes_per_localization)


def _run_gl(body):
    """Run *body* in a subprocess with VisPy's instancing backend selected."""
    script = textwrap.dedent(
        """
        import sys, numpy as np, vispy
        vispy.use(gl="gl+")
        from vispy import app, gloo
        from napari_storm.napari_particles.instanced import InstancedBillboards
        # Whichever binding qtpy resolved, not a fixed one: the package pins no
        # Qt binding, and naming PyQt5 here only passed because napari[all]
        # happens to pull PyQt5 in on Python 3.9.  On 3.10+ it resolves napari
        # 0.7, which does not, and both drawing tests failed on the import.
        from qtpy import API_NAME
        app.use_app(API_NAME.lower())
        canvas = app.Canvas(show=False, size=(128, 128))
        canvas.native
        """
    ) + textwrap.dedent(body)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )


def _run_subprocess(body):
    """Run *body* in a clean interpreter, so GL globals stay clean here."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
    )


def _billboards(n=1000):
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 1] = np.linspace(-0.4, 0.4, n)
    return InstancedBillboards(
        coords, np.ones((n, 3), "f4"), np.ones(n, "f4"), 0.1
    )


# --------------------------------------------------------------------- memory


def test_one_row_per_localization_and_one_shared_quad():
    """The claim: no six-fold expansion, no repeated attributes."""
    billboards = _billboards(n=1000)

    assert billboards.n_instances == 1000
    assert billboards.centers.shape == (1000, 3)
    assert billboards.sigmas.shape == (1000, 3)
    assert billboards.values.shape == (1000,)
    # The quad exists once, for any number of localizations.
    assert QUAD.shape == (4, 2)
    assert QUAD_INDICES.shape == (6,)


def test_the_memory_target_is_met():
    """Level 3 asked for 64-96 B/localization; the native path reached 52."""
    assert instanced_bytes_per_localization() == 32

    # At scale the shared quad amortizes away entirely.
    assert _billboards(n=1_000_000).bytes_per_localization() == pytest.approx(
        32.0, abs=0.01
    )


def test_it_is_smaller_than_both_measured_backends():
    per_localization = _billboards(n=1_000_000).bytes_per_localization()
    assert per_localization < 52  # the napari Points backend
    assert per_localization < 352 / 5  # the current billboards


def test_updating_replaces_the_rows_without_growing_anything():
    billboards = _billboards(n=100)
    billboards.set_data(
        np.zeros((10, 3), "f4"), np.ones((10, 3), "f4"), np.ones(10, "f4"), 0.1
    )
    assert billboards.n_instances == 10
    assert billboards.centers.shape == (10, 3)


def test_malformed_coordinates_are_refused():
    with pytest.raises(ValueError):
        InstancedBillboards(np.zeros((4, 2), "f4"), 1.0, 1.0, 1.0)


def test_scalar_sigma_value_and_size_broadcast():
    billboards = InstancedBillboards(np.zeros((5, 3), "f4"), 1.0, 1.0, 2.0)
    assert billboards.sigmas.shape == (5, 3)
    assert billboards.values.shape == (5,)
    assert billboards.sizes.tolist() == [2.0] * 5


# --------------------------------------------------------------------- drawing


@pytest.mark.slow
def test_the_shader_compiles_and_draws_gaussians():
    """One instanced draw call, two splats, correct positions and falloff."""
    result = _run_gl(
        """
        with canvas:
            gloo.set_state(blend=True, blend_func=("src_alpha", "one"),
                           depth_test=False)
            fbo = gloo.FrameBuffer(gloo.Texture2D(shape=(128, 128, 4)),
                                   gloo.RenderBuffer((128, 128)))
            coords = np.array([[-0.4, 0., 0.], [0.4, 0., 0.]], dtype=np.float32)
            billboards = InstancedBillboards(
                coords, np.ones((2, 3), "f4"), np.ones(2, "f4"), 0.3
            )
            with fbo:
                gloo.clear(color=(0, 0, 0, 1))
                gloo.set_viewport(0, 0, 128, 128)
                billboards.draw(np.eye(4), (1, 0, 0), (0, 1, 0))
                image = fbo.read()

        column = image[:, :, 0].astype(float)
        row = column[64]
        # Clip x of -0.4 and +0.4 land at pixels 38 and 89.
        assert row[38] > 200, row[38]
        assert row[89] > 200, row[89]
        # Dark between the two splats and in the corners.
        assert row[64] < 5, row[64]
        assert column[2, 2] < 5
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "OK" in result.stdout


@pytest.mark.slow
def test_the_profile_is_actually_gaussian():
    """The quality claim.  A disc would pass the test above; this it would not."""
    result = _run_gl(
        """
        with canvas:
            gloo.set_state(blend=True, blend_func=("src_alpha", "one"),
                           depth_test=False)
            fbo = gloo.FrameBuffer(gloo.Texture2D(shape=(256, 256, 4)),
                                   gloo.RenderBuffer((256, 256)))
            billboards = InstancedBillboards(
                np.zeros((1, 3), "f4"), np.ones((1, 3), "f4"),
                np.ones(1, "f4"), 1.0
            )
            with fbo:
                gloo.clear(color=(0, 0, 0, 1))
                gloo.set_viewport(0, 0, 256, 256)
                billboards.draw(np.eye(4), (1, 0, 0), (0, 1, 0))
                image = fbo.read()

        row = image[128, :, 0].astype(float) / 255.0
        centre = 128

        # The quad has half-extent 0.5 in clip space and the viewport spans
        # [-1, 1] over 256 px, so it is 64 px across; the shader cuts it at
        # 2.5 sigma, putting one sigma at 25.6 px.
        sigma_px = 64 / 2.5

        # The written value is the Gaussian *squared*: the shader puts the
        # falloff in alpha as well as colour, and additive blending multiplies
        # by alpha.  The existing renderer's shader does exactly the same
        # (`return val*vec4(1,1,1,1)` with the same blend), so this matches the
        # backend it has to replace rather than being a quirk of the prototype.
        for offset in (0, 6, 12, 18, 24):
            gaussian = np.exp(-0.5 * (offset / sigma_px) ** 2)
            expected = gaussian ** 2
            measured = row[centre + offset]
            assert abs(measured - expected) < 0.03, (offset, measured, expected)

        # And it is a Gaussian, not a disc: strictly decreasing away from centre.
        profile = row[centre : centre + 40]
        assert np.all(np.diff(profile) <= 1e-6), profile
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "OK" in result.stdout


@pytest.mark.slow
def test_instancing_needs_the_gl_plus_backend():
    """Recorded because it constrains the decision, not because it is desired.

    VisPy's default ``gl2`` backend has no ``glDrawElementsInstanced``. The
    backend is selected process-wide, before any GL context exists, and
    switching it puts every napari layer through PyOpenGL, not only ours.

    But the switch is one napari already makes for itself: its own Points layer
    renders instanced, and ``napari/_vispy/visuals/markers.py`` calls
    ``use(gl='gl+')`` at import. So requiring it costs us nothing that a napari
    session is not already paying.

    Checked in a clean subprocess, because by the time any other test here has
    built a viewer the process is already on ``gl+`` and the check would prove
    nothing.
    """
    result = _run_subprocess(
        """
        from vispy.gloo import gl
        from vispy.gloo.gl import gl2, glplus

        # The default backend cannot do it; gl+ can.
        assert not hasattr(gl2, "glDrawElementsInstanced")
        assert not hasattr(gl2, "glVertexAttribDivisor")
        assert hasattr(glplus, "glDrawElementsInstanced")
        assert hasattr(glplus, "glVertexAttribDivisor")

        # Importing a backend module does not select it.
        assert gl.current_backend is gl2, gl.current_backend

        # napari selects it itself, for its own instanced Points rendering.
        import napari
        viewer = napari.Viewer(show=False)
        try:
            assert gl.current_backend is glplus, gl.current_backend
        finally:
            viewer.close()
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "OK" in result.stdout


# ------------------------------------------------- inside napari, end to end


def _run_backend(body):
    """Drive the real plugin with the instanced backend, in a subprocess.

    A subprocess because VisPy's GL backend is process-global and chosen before
    any context exists: by the time pytest has built a napari viewer for another
    test, it is too late to switch, and the instanced renderer refuses to
    construct rather than draw nothing.
    """
    script = textwrap.dedent(
        """
        import numpy as np
        from napari_storm.napari_particles._napari_compat import (
            enable_instanced_backend)
        assert enable_instanced_backend(), "gl+ unavailable"

        import napari
        from napari_storm._dock_widget import napari_storm
        from napari_storm.napari_particles.instanced_renderer import (
            InstancedRenderer)
        from napari_storm.localization_dataset_types import (
            LocalizationDataBaseClass)

        def dataset(n=5000, name="probe"):
            locs = np.zeros(n, dtype=[("x_pos_nm", "f4"), ("y_pos_nm", "f4")])
            locs["x_pos_nm"] = np.linspace(10_000, 12_000, n)
            locs["y_pos_nm"] = np.linspace(40_000, 50_000, n)
            return LocalizationDataBaseClass(
                np.rec.array(locs), name=name, zdim_present=False
            )

        viewer = napari.Viewer(show=False)
        widget = napari_storm(
            napari_viewer=viewer, renderer=InstancedRenderer(viewer)
        )
        """
    ) + textwrap.dedent(body) + textwrap.dedent(
        """
        viewer.close()
        print("OK")
        """
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env
    )


@pytest.mark.slow
def test_the_plugin_runs_on_the_instanced_backend():
    """The checkpoint: load, draw, filter, all through the real widget."""
    result = _run_backend(
        """
        widget.get_dataset_from_test_mode([dataset(5000)])
        data = widget.localization_datasets[0]
        renderer = widget.data_to_layer_itf.renderer
        layer = renderer.layer(data.dataset_id)

        assert layer.n_instances == 5000
        # The mesh is one quad however many localizations there are.
        assert len(layer.data[0]) == 4
        assert renderer.host_bytes(data.dataset_id) / 5000 == 28

        # It reaches the GPU without error.
        viewer.screenshot(canvas_only=True, flash=False)

        # Filtering updates in place, keeping the layer.
        widget.render_config.range_x_percent = np.array([0, 50])
        widget.data_to_layer_itf.update_layers()
        assert renderer.layer(data.dataset_id) is layer
        assert layer.n_instances < 5000
        assert len(layer.data[0]) == 4
        """
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "OK" in result.stdout


@pytest.mark.slow
def test_the_camera_frames_the_data_not_the_quad():
    """Without an _extent_data override napari frames a quad at the origin."""
    result = _run_backend(
        """
        widget.get_dataset_from_test_mode([dataset(1000)])
        data = widget.localization_datasets[0]
        layer = widget.data_to_layer_itf.renderer.layer(data.dataset_id)

        # (z, y, x): x is the last axis, y the middle one.
        extent = layer._extent_data
        assert extent[0][2] < 10_100, extent
        assert extent[1][2] > 11_900, extent
        assert extent[0][1] < 40_100, extent
        assert extent[1][1] > 49_900, extent
        """
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "OK" in result.stdout


@pytest.mark.slow
def test_appearance_and_unloading_work_on_the_instanced_backend():
    result = _run_backend(
        """
        widget.get_dataset_from_test_mode([dataset(1000, "a"), dataset(1000, "b")])
        first, second = widget.localization_datasets
        renderer = widget.data_to_layer_itf.renderer

        widget.channel[0].Slider_opacity.setValue(40)
        assert abs(renderer.appearance(first.dataset_id).opacity - 0.4) < 1e-6

        widget.unload_dataset(0)
        assert not renderer.is_open(first.dataset_id)
        assert renderer.is_open(second.dataset_id)
        """
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "OK" in result.stdout


def test_the_backend_refuses_to_construct_without_instancing(
    make_napari_viewer, monkeypatch
):
    """Better a clear error than a viewer that silently draws nothing.

    The unavailability is forced rather than assumed: whether *this* process
    has ``gl+`` selected depends on what ran before, and a test that skips
    itself on that is not checking the guard.
    """
    from napari_storm.napari_particles import instanced_renderer

    monkeypatch.setattr(
        instanced_renderer, "instancing_available", lambda: False
    )
    with pytest.raises(RuntimeError, match="gl\\+"):
        instanced_renderer.InstancedRenderer(make_napari_viewer())

"""napari-storm's Gaussian model, held against a voxel-based reference.

napari-storm draws one Gaussian per localization.  The classical way of
rendering localizations is to accumulate every Gaussian into a pixel or voxel
grid, then display the grid.  The reference implementation of
the classical way is CrossCorrelate's GPU Gaussian renderer -- ThunderSTORM's
model, in three dimensions, on a GPU.

A comparison of speed and memory is only worth reading once the two draw the
same thing, and *that* is what this module establishes for the export raster.
It is not a benchmark but the correctness floor beneath one; the canvas is held
against the same model in ``test_screen_width.py``, and the memory law of the
two representations is pinned in ``test_memory_accounting.py``.
:func:`voxel_render` is meant to be imported by benchmarks, so the conversion
it states is the one that gets measured.

The two models are not written the same way.  The conversions are stated here
once, where a test pins them:

* napari-storm's Gaussians have unit *peak* amplitude (``values`` scales the
  peak); CrossCorrelate's have unit *mass* (``weights`` is the integral).  The
  factor between them is the Gaussian's volume in pixel units,
  ``2π σy σx / (py px)`` in two dimensions and ``(2π)^1.5 σz σy σx / (pz py px)``
  in three.
* napari-storm truncates at :data:`SPLAT_SIGMAS` on a square in world units;
  CrossCorrelate truncates at ``truncate_sd`` on a square of integer pixel
  offsets.  Asked for the same cutoff they disagree only on a one-pixel rim
  where the Gaussian is below 1e-5 of its peak.
* Both sample the Gaussian at pixel *centres*.  A napari-storm grid's origin is
  the centre of pixel (0, 0); CrossCorrelate's ``origin_nm`` is the coordinate
  of sample (0, 0).  The same number is passed to both.
* CrossCorrelate renormalises a Gaussian that is clipped by the image border
  so that it keeps its mass; napari-storm does not.  Every scene here keeps
  its localizations more than five sigma from the edge so the question never
  arises.

Everything is skipped when CrossCorrelate is not importable: it is a sibling
project, not a dependency.  Install it from its repository to run these.
"""

import numpy as np
import pytest

gaussian_rendering = pytest.importorskip(
    "CrossCorrelation.rendering.gpu_gaussian_rendering",
    reason="CrossCorrelate, the voxel baseline, is not installed in this env",
)
torch = pytest.importorskip("torch")

from napari_storm.core.raster import SPLAT_SIGMAS, GaussianGrid, rasterize  # noqa: E402

# ------------------------------------------------------------ the baseline

#: Upper bound on the elements of CrossCorrelate's per-chunk patch tensor
#: ``(chunk, Kz, Ky, Kx)``.  Above roughly 2e8 the MPS backend returns a volume
#: of zeros without raising (measured 2026-09-05 on an Apple M-series GPU), and
#: well below that it thrashes.  Two hundred megabytes of float32 is safe on
#: every backend tried and small enough that the chunk loop still amortises.
PATCH_ELEMENT_BUDGET = 20_000_000


def devices():
    """Every torch device this machine can run the baseline on."""
    found = ["cpu"]
    if torch.backends.mps.is_available():
        found.append("mps")
    if torch.cuda.is_available():
        found.append("cuda")
    return found


def chunk_size_for(sigma_nm, voxel_nm, truncate_sd, budget=PATCH_ELEMENT_BUDGET):
    """Localizations per chunk that keep the patch tensor inside *budget*.

    *sigma_nm* is the largest sigma per axis and *voxel_nm* the sampling per
    axis, both in the same axis order and of the same length.
    """
    per_localization = 1
    for sigma, voxel in zip(sigma_nm, voxel_nm):
        radius = int(np.ceil(truncate_sd * float(sigma) / float(voxel)))
        per_localization *= 2 * radius + 1
    return max(1, int(budget // per_localization))


def voxel_render(coords_nm, sigmas_nm, values, grid, device, truncate_sd=SPLAT_SIGMAS):
    """The voxel baseline, called with napari-storm's model and grid.

    Takes exactly what :func:`napari_storm.core.raster.rasterize` takes --
    ``(N, 3)`` coordinates and sigmas in ``(z, y, x)`` nanometres, ``(N,)``
    peak amplitudes and a :class:`GaussianGrid` -- and returns an array of the
    grid's shape, so the two can be compared element for element.  The
    peak-to-mass conversion described in the module docstring happens here.
    """
    coords_nm = np.asarray(coords_nm, dtype=np.float64)
    sigmas_nm = np.asarray(sigmas_nm, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    pixel = np.asarray(grid.pixel_size_nm, dtype=np.float64)

    if grid.is_2d:
        mass = (
            values
            * 2.0
            * np.pi
            * sigmas_nm[:, 1]
            * sigmas_nm[:, 2]
            / (pixel[1] * pixel[2])
        )
        image = gaussian_rendering.render_gaussians_2d_torch(
            coords_nm[:, 1:],
            grid.shape[1:],
            tuple(pixel[1:]),
            sigmas_nm[:, 1:],
            grid.origin_nm[1:],
            truncate_sd=truncate_sd,
            weights=mass,
            device=device,
            chunk_size=chunk_size_for(
                sigmas_nm.max(axis=0)[1:], pixel[1:], truncate_sd
            ),
        )
        return image[None]

    mass = values * (2.0 * np.pi) ** 1.5 * np.prod(sigmas_nm, axis=1) / np.prod(pixel)
    return gaussian_rendering.render_gaussians_3d_torch(
        coords_nm,
        grid.shape,
        tuple(pixel),
        sigmas_nm,
        grid.origin_nm,
        truncate_sd=truncate_sd,
        weights=mass,
        device=device,
        chunk_size=chunk_size_for(sigmas_nm.max(axis=0), pixel, truncate_sd),
    )


# --------------------------------------------------------------- fixtures


def _scene(n, sigma_nm, *, zdim, seed=0, span_nm=2_000.0):
    """*n* localizations with *sigma_nm* (``(z, y, x)``) inside a padded grid.

    The span differs per axis so that a transposition changes the answer, and
    the grid is padded by more than the truncation radius on every side so
    that no Gaussian touches a border (see the module docstring on why).
    """
    rng = np.random.default_rng(seed)
    sigma = np.asarray(sigma_nm, dtype=np.float64)
    spans = (
        np.array([0.3, 1.0, 0.6]) * span_nm
        if zdim
        else np.array([0.0, 1.0, 0.6]) * span_nm
    )
    coords = rng.uniform(0.0, 1.0, size=(n, 3)) * spans + 10_000.0
    if not zdim:
        coords[:, 0] = 0.0
    sigmas = np.broadcast_to(sigma, (n, 3)).astype(np.float64)
    values = rng.uniform(0.5, 2.0, size=n)
    pad = (SPLAT_SIGMAS + 1.0) * sigma
    lo, hi = coords.min(axis=0) - pad, coords.max(axis=0) + pad
    return coords, sigmas, values, (lo, hi)


# ----------------------------------------------------- baseline sanity


@pytest.mark.parametrize("device", devices())
def test_the_chosen_chunk_conserves_mass(device):
    """The baseline's own invariant, checked on every device before use.

    CrossCorrelate normalises every Gaussian to its weight, so the sum of the
    output is the sum of the weights whatever the sigma or the grid.  The MPS
    backend breaks this silently when the chunk is too large; this is the test
    that says :func:`chunk_size_for` is small enough here.
    """
    coords, sigmas, values, (lo, hi) = _scene(20_000, (60.0, 30.0, 30.0), zdim=True)
    grid = GaussianGrid.covering(
        ((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])), 15.0, z_step_nm=30.0
    )
    volume = voxel_render(coords, sigmas, values, grid, device)

    expected = (
        values
        * (2.0 * np.pi) ** 1.5
        * np.prod(sigmas, axis=1)
        / np.prod(grid.pixel_size_nm)
    ).sum()
    assert volume.sum() == pytest.approx(expected, rel=1e-3)


# ------------------------------------------------------- model equivalence


@pytest.mark.parametrize("device", devices())
def test_the_2d_export_raster_matches_the_voxel_reference(device):
    """Same scene, same grid: the export and the baseline agree pixel for pixel.

    This is the statement that the export is *the* classical render and not
    merely something like it.  A sigma of three pixels is well resolved, so
    the discrete mass differs from the analytic ``2π σ²`` by far less than the
    tolerance; what the tolerance absorbs is float32 accumulation order.
    """
    coords, sigmas, values, (lo, hi) = _scene(300, (0.0, 30.0, 30.0), zdim=False)
    grid = GaussianGrid.covering(((0.0, 0.0), (lo[1], hi[1]), (lo[2], hi[2])), 10.0)

    reference = rasterize(coords, sigmas, values, grid)
    baseline = voxel_render(coords, sigmas, values, grid, device)

    assert baseline.shape == reference.shape
    np.testing.assert_allclose(baseline, reference, atol=1e-3 * reference.max())


@pytest.mark.parametrize("device", devices())
def test_the_3d_export_raster_matches_the_voxel_reference(device):
    """The same in three dimensions, with a different sigma on every axis.

    Anisotropic on purpose: an axis swap between ``(z, y, x)`` and any other
    order produces a volume with the same mass and the same extent, and only a
    per-axis width can catch it.
    """
    coords, sigmas, values, (lo, hi) = _scene(
        100, (60.0, 30.0, 20.0), zdim=True, span_nm=1_000.0
    )
    grid = GaussianGrid.covering(
        ((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])), 10.0, z_step_nm=20.0
    )

    reference = rasterize(coords, sigmas, values, grid)
    baseline = voxel_render(coords, sigmas, values, grid, device)

    assert baseline.shape == reference.shape
    np.testing.assert_allclose(baseline, reference, atol=1e-3 * reference.max())

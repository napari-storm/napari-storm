"""The memory law of the two representations, pinned to the code.

Dense storage is set by the sampled volume and is independent of the number of
localizations; the production renderer's storage is set by the number of
localizations and is independent of the volume.  Both halves are arithmetic,
and both are stated here once so the documentation and the benchmarks cannot
drift from what the layer actually holds.  Decimal bytes throughout.
"""

import numpy as np
import pytest

from napari_storm.core.raster import GaussianGrid
from napari_storm.memory_budget import (
    INSTANCED_BYTES_PER_LOCALIZATION,
    INSTANCED_SHARED_BYTES,
    instanced_render_bytes_for,
)
from napari_storm.napari_particles.instanced_layer import InstancedParticles


def test_the_production_layer_costs_what_the_accounting_says():
    """One number, measured on the layer the plugin runs, pins the constant.

    The prototype layout in ``napari_particles.instanced`` costs 32 because it
    keeps a per-instance size; the production layer keeps one size per
    dataset and costs 28.  Docs, this test and the benchmark all read the
    constant, so they cannot drift apart again.
    """
    n = 1000
    layer = InstancedParticles(
        np.zeros((n, 3), dtype=np.float32), size=10.0, sigmas=(1.0, 1.0, 1.0)
    )
    assert layer.host_bytes() / n == INSTANCED_BYTES_PER_LOCALIZATION == 28
    assert instanced_render_bytes_for(n) - layer.host_bytes() == INSTANCED_SHARED_BYTES
    assert INSTANCED_SHARED_BYTES == 4 * 3 * 4 + 6 * 4


def test_voxel_memory_grows_with_the_field_and_splat_memory_with_the_data():
    """Dense against retained storage, as arithmetic.

    The numbers are the benchmark fixture's field of view at a 10 nm voxel,
    which is the sampling a 20 nm FWHM needs.  Nothing here allocates: the
    grid reports what it *would* cost, which is the point.  Decimal
    gigabytes throughout, as in the benchmark report.
    """
    field = ((0.0, 1_000.0), (10_000.0, 30_000.0), (10_000.0, 30_000.0))
    grid = GaussianGrid.covering(field, 10.0, z_step_nm=10.0)

    assert grid.shape == (100, 2000, 2000)
    assert grid.nbytes() == 1_600_000_000
    assert instanced_render_bytes_for(100_000) == pytest.approx(2_800_000, abs=100)

    # Twice the lateral field: four times the grid, the same splats.
    wider = ((0.0, 1_000.0), (10_000.0, 50_000.0), (10_000.0, 50_000.0))
    assert (
        GaussianGrid.covering(wider, 10.0, z_step_nm=10.0).nbytes() == 4 * grid.nbytes()
    )
    # Twice the axial extent: twice the grid, the same splats.
    deeper = ((0.0, 2_000.0), (10_000.0, 30_000.0), (10_000.0, 30_000.0))
    assert (
        GaussianGrid.covering(deeper, 10.0, z_step_nm=10.0).nbytes()
        == 2 * grid.nbytes()
    )

    # Ten times the data: ten times the splats, the same grid.
    assert instanced_render_bytes_for(1_000_000) == pytest.approx(
        10 * instanced_render_bytes_for(100_000), rel=1e-4
    )

    # The crossover: the grid costs what 57 million localizations would.
    assert grid.nbytes() / INSTANCED_BYTES_PER_LOCALIZATION == pytest.approx(
        57_142_857, abs=1
    )

"""The scientific core: localization data and derived quantities.

Level 2 of ``docs/modernization-review.md`` separates the science from the
application. Everything in this package depends on numpy and the standard
library only -- **no Qt, no napari, no file I/O** -- so it can be tested
headlessly and reused outside one dock widget.

That boundary is enforced by a test, not by convention: see
``_tests/test_core_is_host_free.py``. Adding a Qt or napari import anywhere
under ``napari_storm.core`` fails the suite.
"""

from .bounds import EMPTY, Bounds
from .dataset_state import (
    AppearanceChanged,
    DatasetState,
    MaskChanged,
    TransformChanged,
)
from .dataset_store import (
    BatchCommitted,
    DatasetClosed,
    DatasetOpened,
    DatasetStore,
    StoreCleared,
)
from .localization_table import (
    ACTIVE,
    AXES,
    COORDINATE_DTYPE,
    DEFAULT_PHOTON_COLUMN,
    DEFAULT_POSITION_COLUMNS,
    DEFAULT_SIGMA_COLUMNS,
    FILTERED,
    LocalizationTable,
    TableSelection,
)
from .metadata import (
    DATA_IN_NM,
    PHOTON_COUNT_PRESENT,
    PIXEL_SIZE_NM,
    SIGMA_PRESENT,
    ZDIM_PRESENT,
    MetadataProvider,
    StaticMetadataProvider,
)
from .render_planner import DatasetTraits, GaussianSettings, RenderPlanner
from .renderer import (
    Changed,
    LayerAppearance,
    LocalizationRenderer,
    NullRenderer,
    RenderRequest,
)
from .scene import (
    CameraState,
    DatasetEntry,
    ReferenceImageEntry,
    Scene,
    SceneFormatError,
    load_scene,
    save_scene,
)
from .validation import (
    InvalidLocalizationData,
    non_finite_mask,
    require_positive_maximum,
    sanitize_positive,
)
from .world_transform import IDENTITY, WorldTransform

__all__ = [
    "ACTIVE",
    "CameraState",
    "DatasetEntry",
    "ReferenceImageEntry",
    "Scene",
    "SceneFormatError",
    "load_scene",
    "save_scene",
    "AXES",
    "FILTERED",
    "TableSelection",
    "COORDINATE_DTYPE",
    "DEFAULT_PHOTON_COLUMN",
    "DEFAULT_POSITION_COLUMNS",
    "DEFAULT_SIGMA_COLUMNS",
    "LocalizationTable",
    "BatchCommitted",
    "DatasetClosed",
    "DatasetOpened",
    "DatasetStore",
    "StoreCleared",
    "MetadataProvider",
    "StaticMetadataProvider",
    "DATA_IN_NM",
    "PHOTON_COUNT_PRESENT",
    "PIXEL_SIZE_NM",
    "SIGMA_PRESENT",
    "ZDIM_PRESENT",
    "LocalizationRenderer",
    "NullRenderer",
    "RenderRequest",
    "InvalidLocalizationData",
    "non_finite_mask",
    "require_positive_maximum",
    "sanitize_positive",
    "Bounds",
    "EMPTY",
    "AppearanceChanged",
    "DatasetState",
    "MaskChanged",
    "TransformChanged",
    "Changed",
    "LayerAppearance",
    "DatasetTraits",
    "GaussianSettings",
    "RenderPlanner",
    "IDENTITY",
    "WorldTransform",
]

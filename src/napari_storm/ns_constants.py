from ._data_constants import MINFLUX_Z_CORRECTION_FACTOR as MINFLUX_Z_CORRECTION_FACTOR
from ._data_constants import STORM_DATA_DTYPE

FWHM_TO_SIGMA = 2.354

DEFAULT_FIXED_FWHM_XY_NM = 20
DEFAULT_FIXED_FWHM_Z_NM = 20
DEFAULT_VAR_PSF_FWHM_XY_NM = 300
DEFAULT_VAR_PSF_FWHM_Z_NM = 700
DEFAULT_VAR_MIN_FWHM_XY_NM = 10
DEFAULT_VAR_MIN_FWHM_Z_NM = 10
DEFAULT_SCALEBAR_NM = 500
DEFAULT_GRID_LINE_DISTANCE_UM = 1.0
PERCENTILE_CLIP = 99
SIGMA_TO_SIZE_FACTOR = 5

#: The plane flat (2-D) localizations are drawn on, in nanometres.  A reference
#: image imported with no localizations loaded is placed here too, so that the
#: two families land on the *same* plane rather than on two that happen to be
#: written in different files.  Sharing one name is the point: while the planner
#: said 1.0 and the image importer defaulted to 0.0, a 2-D dataset and its
#: reference image were one plane apart, and napari's 2-D display -- which shows
#: a single slice -- could only ever draw one of them.
FLAT_DATA_Z_NM = 1.0

# Accepted range for user-entered FWHM values, in nanometres.  The upper bound
# is a safety cap, not a physical one: Gaussian footprint drives fragment cost,
# so an unbounded value entered here can stall the GPU regardless of how many
# localizations are loaded.
MIN_FWHM_NM = 0.1
MAX_FWHM_NM = 100_000.0

# Backwards-compatible public name; the dependency-free constants module is
# the single source of truth shared with data_formats.py.
LOCS_DTYPE = STORM_DATA_DTYPE

list_of_recognized_file_formats = [
    "h5",
    "hdf5",
    "yaml",
    "csv",
    "smlm",
    "npy",
    "json",
    "test",
]

standard_colors = ["white", "red", "blue", "green", "yellow"]

standard_colormaps = ["gray", "red", "blue", "green", "yellow"]

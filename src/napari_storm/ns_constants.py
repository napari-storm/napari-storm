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

#: Which way the camera faces for each named view, as (view_direction,
#: up_direction) in napari's own (z, y, x) world order.
#:
#: Directions rather than Euler angles, because an angle triple only means
#: something against a coordinate convention.  These replace triples tuned when
#: the planner emitted coordinates as (z, x, y); the planner emits (z, y, x)
#: now, and the triples went on selecting the same *angles* and therefore a
#: different pair of data axes -- which is how every one of the three view
#: buttons came to be labelled with a view it does not show.
#:
#: Each view puts the first axis of its name to the right and the second one
#: downward, so XY agrees with what napari's own 2-D display draws.
AXIS_VIEWS = {
    "XY": ((1, 0, 0), (0, -1, 0)),  # down +z; x right, y down
    "XZ": ((0, -1, 0), (-1, 0, 0)),  # along -y; x right, z down
    "YZ": ((0, 0, 1), (-1, 0, 0)),  # along +x; y right, z down
}

#: What a dataset opens in.  Looking down the optical axis is what "the image"
#: means for a localization dataset; the side views are something you ask for.
DEFAULT_AXIS_VIEW = "XY"

#: The extensions an HDF5 localization file turns up under.  Which reader one
#: needs is decided by looking inside it, not by which of these it happens to
#: carry: Picasso tables and daxview molecule sets use them interchangeably.
HDF5_EXTENSIONS = ("h5", "hdf5", "hdf")

#: Every suffix `open_known_filetype_and_import_dataset` will dispatch on, and
#: the single source of truth for what napari is told this plugin reads.  The
#: manifest is checked against this list by a test, because a format the
#: dispatcher handles but the manifest omits is one that File -> Open and
#: drag-and-drop cannot find -- which is what happened to .mat, .pmx and .mfx.
#:
#: Deliberately absent: tif/tiff/dat/raw, which the dispatcher recognizes only
#: in order to explain that raw movies are not what this plugin renders.
list_of_recognized_file_formats = [
    "h5",
    "hdf5",
    "hdf",
    "yaml",
    "csv",
    "smlm",
    "npy",
    "json",
    "mfx",
    "mat",
    "pmx",
    "ns",
    "test",
]

standard_colors = ["white", "red", "blue", "green", "yellow"]

standard_colormaps = ["gray", "red", "blue", "green", "yellow"]

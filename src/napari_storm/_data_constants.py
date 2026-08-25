"""Dependency-free constants shared by import and dataset modules."""

STORM_DATA_DTYPE = [
    ("frame_number", "i4"),
    ("x_pos_pixels", "f4"),
    ("y_pos_pixels", "f4"),
    ("z_pos_pixels", "f4"),
    ("sigma_x_pixels", "f4"),
    ("sigma_y_pixels", "f4"),
    ("sigma_z_pixels", "f4"),
    ("photon_count", "f4"),
]

# Empirical axial correction used by every MINFLUX importer.  This previously
# had conflicting values (0.7 and 0.8) in two modules.
MINFLUX_Z_CORRECTION_FACTOR = 0.8

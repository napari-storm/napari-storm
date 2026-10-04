"""Write applied data and correction provenance without display offsets."""

import json
from pathlib import Path

import h5py


def save_localizations(path, dataset, drift=None):
    path = Path(path).with_suffix(".ns")
    table = dataset.table
    with h5py.File(path, "w") as f:
        output = f.create_dataset("dataset", data=table.records[table.excluded == 0])
        output.attrs["name"] = dataset.name
        output.attrs["zdim_present"] = dataset.zdim_present
        output.attrs["dataset_class"] = type(dataset).__name__
        for name in ("pixelsize_nm", "sigma_present", "photon_count_present", "itr"):
            value = getattr(dataset, name, None)
            if value is not None:
                output.attrs[name] = value
        if drift is not None:
            output.attrs["drift_applied"] = drift.applied
            model = drift.model
            group = f.create_group("postprocessing")
            group["knots"] = model.knots
            group["drift_nm"] = model.displacements_nm
            group["anchor_nm"] = model.anchor_nm
            group["frame_range"] = model.frame_range
            group["scale"] = drift.scale
            if model.rank_times is not None:
                group["rank_times_s"] = model.rank_times
            group.attrs["clock"] = model.clock
            group.attrs["interpolation"] = model.interpolation
            group.attrs["provenance"] = json.dumps(model.provenance)
    return path

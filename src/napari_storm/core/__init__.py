"""The scientific core, now the napari-free ``storm_drawing.core``.

``napari_storm.core`` and every ``napari_storm.core.<module>`` are the same
module objects as ``storm_drawing.core`` and its modules, so existing imports,
isinstance checks and module state are unchanged.  This module enters them in
``sys.modules`` under the old names, and is itself replaced there by
``storm_drawing.core``.
"""

import importlib
import pkgutil
import sys

import storm_drawing.core as _core

for _module in pkgutil.iter_modules(_core.__path__):
    sys.modules[f"{__name__}.{_module.name}"] = importlib.import_module(
        f"{_core.__name__}.{_module.name}"
    )
sys.modules[__name__] = _core

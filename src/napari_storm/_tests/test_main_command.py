"""The ``napari-storm`` command's arguments, and how it opens the files named.

No viewer here: what the command line asks for, and the order and merge flags
the files are opened with, are checked against a stand-in dock.  The desktop
installers' smoke test (installer/launcher.py) runs the whole thing.
"""

from napari_storm.__main__ import _open_in_turn, _parse_args


def test_files_are_positional():
    assert _parse_args(["a.h5", "b.csv"]).files == ["a.h5", "b.csv"]
    assert _parse_args([]).files == []


def test_unknown_options_are_ignored_rather_than_refused():
    # macOS has passed -psn_0_<n> to apps started from the Finder; the viewer
    # should start rather than exit with a usage error.
    assert _parse_args(["-psn_0_12345", "a.h5"]).files == ["a.h5"]


class _Dock:
    """Records each open, and completes it when told to."""

    def __init__(self):
        self.calls = []

    def open_localization_data_file_and_get_dataset(self, **kwargs):
        self.calls.append(kwargs)
        return True

    def finish(self, index):
        self.calls[index]["on_loaded"](["dataset"])


def test_files_open_one_after_another_and_merge_after_the_first():
    dock = _Dock()
    _open_in_turn(dock, ["a.h5", "b.h5", "c.h5"], merge=False)
    # Background loads must not overlap: only the first has started.
    assert [call["file_path"] for call in dock.calls] == ["a.h5"]
    assert dock.calls[0]["merge"] is False
    assert dock.calls[0]["background"] is True

    dock.finish(0)
    dock.finish(1)
    assert [call["file_path"] for call in dock.calls] == ["a.h5", "b.h5", "c.h5"]
    assert [call["merge"] for call in dock.calls] == [False, True, True]

    dock.finish(2)
    assert len(dock.calls) == 3


def test_no_files_opens_nothing():
    dock = _Dock()
    _open_in_turn(dock, [], merge=False)
    assert dock.calls == []

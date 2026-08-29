"""Facts a reader needs but cannot find in the file.

Several formats are missing something the plugin requires — a Picasso HDF5 with
no pixel size, an ``.npy`` with no dimensionality flag. Those readers used to
construct a ``QInputDialog`` themselves, which is what made the data layer
depend on Qt and made a reader untestable without a running application.

A reader now *asks its provider* and does not know or care how the answer is
obtained. The application supplies one that opens a dialog; a script or a test
supplies :class:`StaticMetadataProvider` with the answers already known; a
headless caller gets the default, which answers nothing and lets the reader fail
the way it already failed when a user pressed Cancel.

Questions are addressed by a stable *key* rather than by their prompt text, so a
provider can answer ``"pixelsize_nm"`` without matching on an English sentence
that may be reworded at any time.
"""

from __future__ import annotations

__all__ = [
    "MetadataProvider",
    "StaticMetadataProvider",
    "PIXEL_SIZE_NM",
    "ZDIM_PRESENT",
    "DATA_IN_NM",
    "SIGMA_PRESENT",
    "PHOTON_COUNT_PRESENT",
]

#: Keys the bundled readers ask about.  A provider may answer any subset.
PIXEL_SIZE_NM = "pixelsize_nm"
ZDIM_PRESENT = "zdim_present"
DATA_IN_NM = "data_in_nm"
SIGMA_PRESENT = "sigma_present"
PHOTON_COUNT_PRESENT = "photon_count_present"


class MetadataProvider:
    """Answers nothing.  The default, and what a headless caller gets.

    ``None`` means "no answer", which is the same thing a dismissed dialog has
    always meant. Readers already handle it: they either leave the key absent or
    raise, and this class deliberately does not change which.
    """

    def ask_yes_no(self, key, question):
        """Return True, False, or None if the question cannot be answered."""
        return None

    def ask_text(self, key, question):
        """Return a string, or None if the question cannot be answered."""
        return None


class StaticMetadataProvider(MetadataProvider):
    """Canned answers, for scripted loading and for tests.

    >>> provider = StaticMetadataProvider({PIXEL_SIZE_NM: "100"})
    >>> provider.ask_text(PIXEL_SIZE_NM, "Enter the pixelsize [nm]")
    '100'
    """

    def __init__(self, answers=None):
        self.answers = dict(answers or {})

    def ask_yes_no(self, key, question):
        answer = self.answers.get(key)
        return None if answer is None else bool(answer)

    def ask_text(self, key, question):
        answer = self.answers.get(key)
        return None if answer is None else str(answer)

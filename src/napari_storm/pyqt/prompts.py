"""Modal questions importers may ask, safe to call from a background loader.

Importers run on a worker thread once loading moves off the GUI thread (P1-09),
and a ``QDialog`` cannot be constructed there.  Each helper marshals the whole
construct-and-exec onto the GUI thread and blocks the caller until answered, so
importers keep asking their questions exactly where they always did.

Each returns ``None`` when the user dismissed the dialog without answering.
"""
from qtpy import QtCore
from qtpy.QtWidgets import QDialog, QInputDialog

from ..background_loading import run_on_main_thread
from ..core import MetadataProvider
from .get_string_dialog import GetStringWrapper
from .yes_or_no_dialog import YesNoWrapper

__all__ = ["ask_yes_no", "ask_string", "ask_line_edit", "QtMetadataProvider"]


def _exec_wrapper(wrapper_class, question):
    window = wrapper_class(question)
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    if window.exec_() == QDialog.Accepted:
        return window.tobereturned
    return None


def ask_yes_no(question):
    """True/False from a yes-no dialog, or None if it was dismissed."""
    return run_on_main_thread(_exec_wrapper, YesNoWrapper, question)


def ask_string(question):
    """A string from a single-field dialog, or None if it was dismissed."""
    return run_on_main_thread(_exec_wrapper, GetStringWrapper, question)


def _exec_input_dialog(title, label):
    text, accepted = QInputDialog.getText(None, title, label)
    return text if accepted else None


def ask_line_edit(title, label):
    """A string from ``QInputDialog``, or None if the user cancelled."""
    return run_on_main_thread(_exec_input_dialog, title, label)


class QtMetadataProvider(MetadataProvider):
    """Answers a reader's questions by asking the user.

    This is the only place the two halves meet: readers depend on the abstract
    provider in ``napari_storm.core`` and never on Qt, and the application
    supplies this implementation at the point where it opens a file.
    """

    def ask_yes_no(self, key, question):
        return ask_yes_no(question)

    def ask_text(self, key, question):
        # A single-field dialog for everything.  The question already carries
        # its own units and phrasing; the key is for programmatic providers.
        return ask_string(question)

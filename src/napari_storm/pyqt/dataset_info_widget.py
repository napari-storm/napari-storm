"""
Structured card-based info panel for loaded localization datasets.
Replaces the old TestListView plain-text list.
"""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..CustomErrors import ParentError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TYPE_LABELS = {
    "LocalizationDataBaseClass": ("Localizations", "#5a8a5a"),
    "StormDataClass(LocalizationDataBaseClass)": ("STORM / PALM", "#b06030"),
    "MinfluxDataBaseClass(LocalizationDataBaseClass)": ("MINFLUX", "#3070b0"),
    "MinfluxDataAIIterationClass(MinfluxDataBaseClass)": ("MINFLUX AI", "#5050b0"),
}

_CARD_STYLE = """
QFrame#DatasetCard {
    background-color: rgb(45, 52, 60);
    border: 1px solid rgb(70, 82, 94);
    border-radius: 4px;
}
"""

_KEY_STYLE = "color: rgb(140, 152, 164); font-size: 11px;"
_VAL_STYLE = "color: rgb(220, 228, 236); font-size: 11px;"
_NAME_STYLE = "color: rgb(240, 245, 250); font-size: 13px; font-weight: bold;"


def _fmt_um(nm_value: float) -> str:
    return f"{nm_value / 1000:.2f} µm"


def _fmt_n(n: int) -> str:
    return f"{n:,}"


def _type_badge(dataset_type: str):
    label_text, color = _TYPE_LABELS.get(dataset_type, (dataset_type, "#888888"))
    lbl = QLabel(label_text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet(
        f"background-color: {color}; color: white; border-radius: 3px;"
        f" font-size: 10px; font-weight: bold; padding: 1px 5px;"
    )
    lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return lbl


# ---------------------------------------------------------------------------
# Info card
# ---------------------------------------------------------------------------


def _make_card(dataset) -> QFrame:
    """Build a styled QFrame card containing metadata for *dataset*."""
    card = QFrame()
    card.setObjectName("DatasetCard")
    card.setStyleSheet(_CARD_STYLE)
    card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    layout = QVBoxLayout()
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(4)

    # --- header: name + type badge ---
    header = QWidget()
    h_layout = QGridLayout()
    h_layout.setContentsMargins(0, 0, 0, 0)

    name_lbl = QLabel(dataset.name)
    name_lbl.setStyleSheet(_NAME_STYLE)
    name_lbl.setWordWrap(True)
    h_layout.addWidget(name_lbl, 0, 0)

    badge = _type_badge(dataset.dataset_type)
    h_layout.addWidget(badge, 0, 1, Qt.AlignRight | Qt.AlignVCenter)
    h_layout.setColumnStretch(0, 1)
    header.setLayout(h_layout)
    layout.addWidget(header)

    # --- thin separator ---
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet("background-color: rgb(70, 82, 94); max-height: 1px;")
    layout.addWidget(sep)

    # --- key-value grid ---
    grid = QGridLayout()
    grid.setContentsMargins(0, 2, 0, 0)
    grid.setVerticalSpacing(2)
    grid.setHorizontalSpacing(8)
    grid.setColumnStretch(1, 1)

    rows = _collect_rows(dataset)
    for r, (key, val) in enumerate(rows):
        k_lbl = QLabel(key)
        k_lbl.setStyleSheet(_KEY_STYLE)
        v_lbl = QLabel(val)
        v_lbl.setStyleSheet(_VAL_STYLE)
        v_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        grid.addWidget(k_lbl, r, 0)
        grid.addWidget(v_lbl, r, 1)

    layout.addLayout(grid)
    card.setLayout(layout)
    return card


def _collect_rows(dataset):
    """Return list of (key, value-string) pairs for the card body."""
    rows = []

    # Localizations.  Filtering and render-budget thinning are reported
    # separately: one is the user's choice about their data, the other is an
    # accommodation to the GPU, and calling the second "filtered" would be a lie.
    n_all = dataset.number_of_entries()
    n_filtered = dataset.number_of_filtered_entries()
    n_drawn = dataset.number_of_active_entries()
    if n_filtered == n_all:
        rows.append(("Localizations", _fmt_n(n_all)))
    else:
        rows.append(
            ("Localizations", f"{_fmt_n(n_filtered)} / {_fmt_n(n_all)} (filtered)")
        )
    if n_drawn != n_filtered:
        rows.append(("Drawn", f"{_fmt_n(n_drawn)} (render budget)"))

    # Dimensions
    rows.append(("Dimensions", "3D" if dataset.zdim_present else "2D"))

    # Spatial extents (all locs)
    x_all = dataset.x_pos_nm_all
    y_all = dataset.y_pos_nm_all
    x_ext = float(np.max(x_all) - np.min(x_all))
    y_ext = float(np.max(y_all) - np.min(y_all))
    rows.append(("X extent", _fmt_um(x_ext)))
    rows.append(("Y extent", _fmt_um(y_ext)))

    if dataset.zdim_present:
        z_all = dataset.z_pos_nm_all
        z_ext = float(np.max(z_all) - np.min(z_all))
        rows.append(("Z extent", _fmt_um(z_ext)))

    # STORM/PALM specific
    if hasattr(dataset, "pixelsize_nm") and dataset.pixelsize_nm is not None:
        rows.append(("Pixel size", f"{dataset.pixelsize_nm:.0f} nm"))

    # Uncertainty info
    if hasattr(dataset, "sigma_present"):
        parts = []
        if dataset.sigma_present:
            parts.append("sigma")
        if getattr(dataset, "photon_count_present", False):
            parts.append("photons")
        if parts:
            rows.append(("Uncertainties", ", ".join(parts)))
        elif dataset.uncertainty_defined is False:
            rows.append(("Uncertainties", "none"))

    # MINFLUX AI iteration
    if hasattr(dataset, "itr") and dataset.itr is not None:
        rows.append(("Iteration", str(dataset.itr)))

    return rows


# ---------------------------------------------------------------------------
# Main panel widget
# ---------------------------------------------------------------------------


class DatasetInfoPanel(QWidget):
    """
    Scrollable panel that displays a structured info card for each loaded
    localization dataset.  Drop-in replacement for the old TestListView.
    """

    def __init__(self, datasets, parent=None):
        super().__init__(parent)
        self._parent_ref = parent
        self.datasets = datasets
        self._cards = []

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Placeholder shown when no dataset is loaded
        self._placeholder = QLabel(
            "STATISTICS\nWaiting for Data...\nImport or drag a file here"
        )
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            "color: rgb(130, 142, 154); font-style: italic; padding: 16px;"
        )
        outer.addWidget(self._placeholder)

        # Scroll area for cards
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea { border: none; }")
        self._scroll.hide()

        self._cards_widget = QWidget()
        self._cards_layout = QVBoxLayout()
        self._cards_layout.setContentsMargins(4, 4, 4, 4)
        self._cards_layout.setSpacing(8)
        self._cards_layout.addStretch(1)
        self._cards_widget.setLayout(self._cards_layout)
        self._scroll.setWidget(self._cards_widget)
        outer.addWidget(self._scroll, 1)

        self.setLayout(outer)

        # Keep drag-and-drop working (forwarded to parent)
        self.setAcceptDrops(True)

    # ------------------------------------------------------------------
    # Parent property (mirrors old TestListView API)
    # ------------------------------------------------------------------

    @property
    def parent(self):
        return self._parent_ref

    @parent.setter
    def parent(self, value):
        raise ParentError("Cannot change parent of existing Widget")

    # ------------------------------------------------------------------
    # Public API matching old TestListView
    # ------------------------------------------------------------------

    def update_dataset_ref(self):
        self.datasets = self._parent_ref.localization_datasets

    def show_infos(self, filename, idx):
        """Add a card for the dataset at *idx* in self.datasets."""
        self.update_dataset_ref()
        dataset = self.datasets[idx]
        card = _make_card(dataset)
        # Insert before the trailing stretch (last item)
        insert_at = max(self._cards_layout.count() - 1, 0)
        self._cards_layout.insertWidget(insert_at, card)
        self._cards.append(card)
        self._placeholder.hide()
        self._scroll.show()

    def clear(self):
        """Remove all dataset cards and show the placeholder again."""
        for card in self._cards:
            self._cards_layout.removeWidget(card)
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        self._scroll.hide()
        self._placeholder.show()

    def remove_dataset(self, dataset_index):
        """Remove the card matching an unloaded dataset."""
        if not 0 <= dataset_index < len(self._cards):
            return
        card = self._cards.pop(dataset_index)
        self._cards_layout.removeWidget(card)
        card.setParent(None)
        card.deleteLater()
        self.update_dataset_ref()
        if not self._cards:
            self._scroll.hide()
            self._placeholder.show()

    # ------------------------------------------------------------------
    # Drag-and-drop forwarding
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            urls = event.mimeData().urls()
            file_path = urls[0].toString()[8:]
            self._parent_ref.open_localization_data_file_and_get_dataset(
                file_path=file_path, background=True
            )
        else:
            event.ignore()

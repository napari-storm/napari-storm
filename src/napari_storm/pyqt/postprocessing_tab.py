"""Dataset-specific post-processing controls."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class PostProcessingWindow(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        content = QWidget()
        self.setWidget(content)
        root_layout = QVBoxLayout(content)
        layout = root_layout
        self.dataset = QComboBox()
        layout.addWidget(self.dataset)
        self.status = QLabel("Load localizations to begin.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.actions = {}
        self.inputs = {}

        def section(title, *, collapsible=False):
            nonlocal layout
            body = QWidget()
            layout = QVBoxLayout(body)
            layout.setContentsMargins(0, 0, 0, 0)
            if collapsible:
                title_button = QToolButton()
                title_button.setText(title)
                title_button.setCheckable(True)
                title_button.setToolButtonStyle(
                    Qt.ToolButtonStyle.ToolButtonTextBesideIcon
                )
                title_button.setArrowType(Qt.ArrowType.RightArrow)
                title_button.setStyleSheet(
                    "QToolButton { font-weight: bold; border: none; padding: 12px 0 4px 0; }"
                )

                def toggle(expanded):
                    body.setVisible(expanded)
                    title_button.setArrowType(
                        Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
                    )

                title_button.toggled.connect(toggle)
                root_layout.addWidget(title_button)
                body.hide()
            else:
                label = QLabel(title)
                label.setStyleSheet("font-weight: bold; margin-top: 12px")
                root_layout.addWidget(label)
            root_layout.addWidget(body)

        def button(key, label):
            b = QPushButton(label)
            self.actions[key] = b
            layout.addWidget(b)
            return b

        def number(key, label, value, low, high, integer=False):
            control = QSpinBox() if integer else QDoubleSpinBox()
            control.setRange(low, high)
            control.setValue(value)
            row = QFormLayout()
            row.addRow(label, control)
            layout.addLayout(row)
            self.inputs[key] = control

        section("Fiducials", collapsible=True)
        button("detect", "Detect fiducials")
        self.candidates = QListWidget()
        self.candidates.setMaximumHeight(140)
        layout.addWidget(self.candidates)
        button("exclude", "Exclude ticked fiducials")
        button("restore", "Restore fiducials")
        note = QLabel(
            "Detection uses raw positions. Exclusion discs span the full z range."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        section("Grouping (STORM)", collapsible=True)
        number("distance", "Max distance [nm]", 30, 0.1, 10000)
        number("dark", "Max dark frames", 1, 0, 10000, True)
        number("duration", "Max frames", 50, 1, 100000, True)
        button("group", "Group localizations")
        section("Drift correction")
        number("max_drift", "Max drift [nm]", 300, 3, 100000)
        self.comet_note = QLabel("")
        self.comet_note.setWordWrap(True)
        self.comet_note.setStyleSheet("color: grey")
        layout.addWidget(self.comet_note)
        self.input_mode = QComboBox()
        self.input_mode.addItems(["Localizations", "Group means"])
        layout.addWidget(self.input_mode)
        number("window", "Localizations per time window", 60, 1, 1000000, True)
        note = QLabel(
            "Target sigma: 10 nm. With group or trace means, the count is means per window."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        for key, label in [
            ("estimate", "Estimate memory"),
            ("run", "Run COMET"),
            ("cancel", "Cancel"),
            ("undo", "Undo"),
            ("reapply", "Re-apply"),
            ("discard", "Discard drift"),
            ("save", "Save drift…"),
            ("load", "Load drift…"),
            ("transfer", "Apply to other datasets…"),
            ("export", "Save corrected localizations…"),
        ]:
            button(key, label)
        self.plot_layout = QVBoxLayout()
        layout.addLayout(self.plot_layout)
        section("Explore", collapsible=True)
        self.fraction_label = QLabel("Correction: 100 % (applied data is exported)")
        layout.addWidget(self.fraction_label)
        self.fraction = QSlider(Qt.Orientation.Horizontal)
        self.fraction.setRange(0, 10)
        self.fraction.setValue(10)
        layout.addWidget(self.fraction)
        button("play", "Play raw → corrected")
        self.time_view = QCheckBox("x, y, time view (2D)")
        layout.addWidget(self.time_view)
        self.pairs = QCheckBox("Show pair network")
        layout.addWidget(self.pairs)
        number("radius", "Pair radius [nm]", 35, 0.1, 100000)
        self.pair_mode = QComboBox()
        self.pair_mode.addItems(["All pairs in region", "Chains"])
        layout.addWidget(self.pair_mode)
        self.pair_status = QLabel(
            "At most 200,000 vectors. Pairs are selected on corrected positions; this is not independent validation."
        )
        self.pair_status.setWordWrap(True)
        layout.addWidget(self.pair_status)
        root_layout.addStretch()

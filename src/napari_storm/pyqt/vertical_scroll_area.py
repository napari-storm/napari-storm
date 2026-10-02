"""A scroll area that scrolls vertically and is never narrower than its content."""

from qtpy.QtCore import QEvent, QSize, Qt
from qtpy.QtWidgets import QScrollArea


class VerticalScrollArea(QScrollArea):
    """Scrolls vertically only, and asks for the width its content needs.

    A QScrollArea keeps its content's size to itself -- that is what lets it
    scroll.  With the horizontal scrollbar off, content wider than the dock was
    simply cut off on the right: a loaded dataset's channel controls need about
    500 px, and the dock stayed at the width it had been given while empty.
    This reports the content's minimum width as its own, so the dock grows to
    fit, and re-announces it whenever the content's layout changes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def setWidget(self, widget):
        super().setWidget(widget)
        widget.installEventFilter(self)
        self.updateGeometry()

    def eventFilter(self, watched, event):
        if watched is self.widget() and event.type() == QEvent.LayoutRequest:
            # The content's minimum may have changed; let the layouts above
            # ask again.  QScrollArea never does: it only tracks resizes.
            self.updateGeometry()
        return super().eventFilter(watched, event)

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        widget = self.widget()
        if widget is None:
            return hint
        # Room for the vertical scrollbar whether or not it is showing, so
        # that its appearing does not squeeze the content it scrolls.
        width = (
            widget.minimumSizeHint().width()
            + self.verticalScrollBar().sizeHint().width()
            + 2 * self.frameWidth()
        )
        return QSize(max(hint.width(), width), hint.height())

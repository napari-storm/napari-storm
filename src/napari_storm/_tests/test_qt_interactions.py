"""Regression tests for custom mouse interactions under Qt5/Qt6."""

from qtpy.QtCore import QEvent, QPoint, QPointF, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QApplication, QStyle, QStyleOptionSlider, QWidget

from napari_storm.pyqt.GridPlaneSlider import GridPlaneSlider
from napari_storm.pyqt.detachable_tab import DetachableTabWidget
import napari_storm.pyqt.detachable_tab as detachable_tab_module


def _move_event(x, y=10):
    local = QPointF(x, y)
    return QMouseEvent(
        QEvent.MouseMove,
        local,
        local,
        Qt.NoButton,
        Qt.LeftButton,
        Qt.NoModifier,
    )


def test_tab_drag_uses_threshold_and_qt6_mouse_position(qtbot, monkeypatch):
    tabs = DetachableTabWidget()
    tabs.addTab(QWidget(), "Data")
    qtbot.addWidget(tabs)
    tabs.resize(400, 200)
    tabs.show()

    created_drags = []

    class _Drag:
        def __init__(self, parent):
            created_drags.append(parent)

        def setMimeData(self, _mime_data):
            pass

        def setPixmap(self, _pixmap):
            pass

        def exec_(self, _actions):
            # Keep the test inside the tab bar and avoid opening a dialog.
            return Qt.MoveAction

    monkeypatch.setattr(detachable_tab_module, "QDrag", _Drag)

    bar = tabs.tabBar
    bar.dragStartPos = QPoint(10, 10)
    bar.dragInitiated = False

    # A normal tiny movement while selecting a tab must not begin a drag.
    bar.mouseMoveEvent(_move_event(10 + QApplication.startDragDistance() - 1))
    assert not bar.dragInitiated
    assert created_drags == []

    # Crossing the threshold constructs the synthetic QPointF move event and
    # completes without the PyQt6 QPoint overload error.
    bar.mouseMoveEvent(_move_event(10 + QApplication.startDragDistance()))
    assert bar.dragInitiated
    assert created_drags == [bar]


class _GridInterface:
    def __init__(self):
        self.calls = []

    def update_grid_plane(self, **kwargs):
        self.calls.append(kwargs)


def _slider_handle_center(slider):
    option = QStyleOptionSlider()
    slider.initStyleOption(option)
    return slider.style().subControlRect(
        QStyle.CC_Slider,
        option,
        QStyle.SC_SliderHandle,
        slider,
    ).center()


def test_grid_slider_clicking_handle_does_not_jump(qtbot):
    interface = _GridInterface()
    slider = GridPlaneSlider(
        data_to_layer_interface=interface,
        type_of_slider="line_thickness",
        init_range=(1, 100),
        init_value=50,
    )
    qtbot.addWidget(slider)
    slider.resize(400, 30)
    slider.show()

    interface.calls.clear()
    qtbot.mouseClick(slider, Qt.LeftButton, pos=_slider_handle_center(slider))
    assert slider.value() == 50
    assert interface.calls == []

    slider.setValue(60)
    assert interface.calls[-1] == {"line_thickness": 60}

    qtbot.mouseClick(slider, Qt.MiddleButton, pos=_slider_handle_center(slider))
    assert slider.value() == 50
    assert interface.calls[-1] == {"line_thickness": 50}

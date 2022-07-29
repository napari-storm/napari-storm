from PyQt5.QtWidgets import QFrame, QSizePolicy


class QHSeperationLine(QFrame):

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(1)
        self.setFixedHeight(2)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setStyleSheet("background-color: rgb(65, 72, 81);")
        return

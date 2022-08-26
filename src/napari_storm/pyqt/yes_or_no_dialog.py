from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QDialog, QFormLayout


class YesNoWrapper(QDialog):
    """Dialog wrapper for QWidget that should be shown in a dialog,
     also has the option to return a value from said widget"""
    def __init__(self, question):
        super().__init__()
        self.widget = YesNoDialogWidget(question, parent=self)
        self.layout = QFormLayout()
        self.layout.addRow(self.widget)
        self.setLayout(self.layout)

        self.tobereturned = None


class YesNoDialogWidget(QWidget):
    """Dialog which asks a yes-no-question"""
    def __init__(self, question, parent):
        super().__init__()

        self.parent = parent

        self.layout = QFormLayout()

        self.Lquestion = QLabel(question)
        self.Byes = QPushButton("yes")
        self.Bno = QPushButton("no")
        self.Byes.clicked.connect(self.yes)
        self.Bno.clicked.connect(self.no)

        self.layout.addRow(self.Lquestion)
        self.layout.addRow(self.Byes)
        self.layout.addRow(self.Bno)

        self.setLayout(self.layout)

    def yes(self):
        self.parent.tobereturned = True
        self.parent.accept()

    def no(self):
        self.parent.tobereturned = False
        self.parent.accept()



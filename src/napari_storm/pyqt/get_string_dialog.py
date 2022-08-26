from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QDialog, QFormLayout, QLineEdit


class GetStringWrapper(QDialog):
    """Dialog wrapper for QWidget that should be shown in a dialog,
     also has the option to return a value from said widget"""

    def __init__(self, question):
        super().__init__()
        self.widget = GetStringDialogWidget(question, parent=self)
        self.layout = QFormLayout()
        self.layout.addRow(self.widget)
        self.setLayout(self.layout)

        self.tobereturned = None


class GetStringDialogWidget(QWidget):
    """Dialog which asks for a string"""

    def __init__(self, question, parent):
        super().__init__()

        self.parent = parent

        self.layout = QFormLayout()

        self.Lquestion = QLabel(question)
        self.Eanswer = QLineEdit()
        self.Byes = QPushButton("accept")
        self.Byes.clicked.connect(self.yes)

        self.layout.addRow(self.Lquestion)
        self.layout.addRow(self.Eanswer)
        self.layout.addRow(self.Byes)

        self.setLayout(self.layout)

    def yes(self):
        self.parent.tobereturned = self.Eanswer.text()
        self.parent.accept()




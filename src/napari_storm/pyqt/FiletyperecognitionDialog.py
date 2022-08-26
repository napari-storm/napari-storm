from PyQt5.QtWidgets import QDialog, QFormLayout, QWidget, QLabel, QComboBox, QPushButton


class MainWindowWrapper(QDialog):
    """Dialog wrapper for QWidget that should be shown in a dialog,
     also has the option to return a value from said widget"""
    def __init__(self, description, options, accept_function, abort_function):
        super().__init__()
        self.widget = FiletypeRecognitionsChooseDialoge(description, options, accept_function, abort_function,
                                                        parent=self)
        self.layout = QFormLayout()
        self.layout.addRow(self.widget)
        self.setLayout(self.layout)

        self.tobereturned = None


class FiletypeRecognitionsChooseDialoge(QWidget):
    """Multi Purpose QWidget which gives the user a set of choices and then continues to do what is specified
    in the 'accept' and 'abort' functions"""
    def __init__(self, description, options, accept_function, abort_function, parent):
        super().__init__()

        self.options_index = 0
        self.parent = parent

        self.accept_function = accept_function
        self.abort_function = abort_function
        self.layout = QFormLayout()

        self.label = QLabel()
        self.label.setText(description)

        self.options_cb = QComboBox()
        self.options_cb.addItems(options)
        self.options_cb.setCurrentIndex(self.options_index)
        self.options_cb.currentIndexChanged.connect(self.options_index_changed)

        self.accept_button = QPushButton()
        self.accept_button.setText("accept choice")
        self.abort_button = QPushButton("abort")

        self.layout.addRow(self.label)
        self.layout.addRow(self.options_cb)
        self.layout.addRow(self.accept_button)
        self.layout.addRow(self.abort_button)

        self.accept_button.clicked.connect(self.accepted)
        self.abort_button.clicked.connect(self.abort)

        self.setLayout(self.layout)

    def options_index_changed(self):
        self.options_index = self.options_cb.currentIndex()

    def accepted(self):
        self.parent.tobereturned = self.accept_function(self.options_index)
        self.parent.accept()

    def abort(self):
        self.parent.tobereturned = self.abort_function()
        self.parent.accept()

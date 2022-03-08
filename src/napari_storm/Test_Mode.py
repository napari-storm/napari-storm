from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout, QLineEdit, QComboBox, QPushButton, QDialog
import numpy as np
from .ns_constants import *
from .LocalizationData import LocalizationData


class TestModeWindow(QDialog):
    def __init__(self, parent):
        super().__init__()
        print("test mode started")
        self.setWindowTitle("TestMode")
        self.parent = parent

        self.available_arrangement_modes = ["line", "grid", "lattice", "random"]
        self.current_arrangement_mode = 0

        self.available_size_modes = ["fixed", "variable", "random"]
        self.current_size_mode = 0

        self.num_of_locs = 125
        self.dist_of_locs_nm = 25
        self.sigma_x_nm = 10
        self.sigma_y_nm = 10
        self.sigma_z_nm = 10

        self.layout = QGridLayout()

        self.Lnum_locs = QLabel()
        self.Lnum_locs.setText("Number of locs")

        self.Enum_locs = QLineEdit()
        self.Enum_locs.setText(str(self.num_of_locs))
        self.Enum_locs.textChanged.connect(self.num_locs_changed)

        self.Larrangement_mode = QLabel()
        self.Larrangement_mode.setText("loc arrangement and distance [nm]")

        self.Carrangement_mode = QComboBox()
        self.Carrangement_mode.addItems(self.available_arrangement_modes)
        self.Carrangement_mode.currentIndexChanged.connect(self.arrangement_mode_changed)

        self.Edist_of_locs_nm = QLineEdit()
        self.Edist_of_locs_nm.setText(str(self.dist_of_locs_nm))

        self.Lsize_mode = QLabel()
        self.Lsize_mode.setText("Size mode")

        self.Csize_mode = QComboBox()
        self.Csize_mode.addItems(self.available_size_modes)
        self.Csize_mode.currentIndexChanged.connect(self.size_mode_changed)

        self.Lsigma_x_nm = QLabel()
        self.Lsigma_x_nm.setText("Sigma [nm]:")

        self.Lsigma_y_nm = QLabel()
        self.Lsigma_y_nm.setText("Sigma Y [nm]:")
        self.Lsigma_y_nm.hide()

        self.Lsigma_z_nm = QLabel()
        self.Lsigma_z_nm.setText("Sigma Z [nm]:")
        self.Lsigma_z_nm.hide()

        self.Esigma_x_nm = QLineEdit()
        self.Esigma_x_nm.setText(str(self.sigma_x_nm))
        self.Esigma_x_nm.textChanged.connect(self.sigma_x_changed)

        self.Esigma_y_nm = QLineEdit()
        self.Esigma_y_nm.setText(str(self.sigma_y_nm))
        self.Esigma_y_nm.hide()
        self.Esigma_y_nm.textChanged.connect(self.sigma_y_changed)

        self.Esigma_z_nm = QLineEdit()
        self.Esigma_z_nm.setText(str(self.sigma_z_nm))
        self.Esigma_z_nm.hide()
        self.Esigma_z_nm.textChanged.connect(self.sigma_z_changed)

        self.Baccept = QPushButton()
        self.Baccept.setText("accept")
        self.Baccept.clicked.connect(self.accept_evaluate_return_dataset)

        self.layout.addWidget(self.Lnum_locs, 0, 0)
        self.layout.addWidget(self.Enum_locs, 0, 1)
        self.layout.addWidget(self.Larrangement_mode, 1, 0)
        self.layout.addWidget(self.Carrangement_mode, 1, 1)
        self.layout.addWidget(self.Edist_of_locs_nm, 1, 2)
        self.layout.addWidget(self.Lsize_mode, 2, 0)
        self.layout.addWidget(self.Csize_mode, 2, 1)
        self.layout.addWidget(self.Lsigma_x_nm, 3, 0)
        self.layout.addWidget(self.Lsigma_y_nm, 3, 1)
        self.layout.addWidget(self.Lsigma_z_nm, 3, 2)
        self.layout.addWidget(self.Esigma_x_nm, 4, 0)
        self.layout.addWidget(self.Esigma_y_nm, 4, 1)
        self.layout.addWidget(self.Esigma_z_nm, 4, 2)
        self.layout.addWidget(self.Baccept, 5, 1)
        self.setLayout(self.layout)

    def num_locs_changed(self):
        self.num_of_locs = int(self.Enum_locs.text())

    def sigma_x_changed(self):
        self.sigma_x_nm = float(self.Esigma_x_nm.text())

    def sigma_y_changed(self):
        self.sigma_y_nm = float(self.Esigma_y_nm.text())

    def sigma_z_changed(self):
        self.sigma_z_nm = float(self.Esigma_z_nm.text())

    def size_mode_changed(self):
        if self.Csize_mode.currentText() == self.available_size_modes[0]:
            self.Esigma_y_nm.hide()
            self.Esigma_z_nm.hide()
            self.Lsigma_y_nm.hide()
            self.Lsigma_z_nm.hide()
            self.Lsigma_x_nm.setText("Sigma [nm]")
            self.current_size_mode = 0
        elif self.Csize_mode.currentText() == self.available_size_modes[1]:
            self.Esigma_y_nm.show()
            self.Esigma_z_nm.show()
            self.Lsigma_y_nm.show()
            self.Lsigma_z_nm.show()
            self.Lsigma_x_nm.setText("Sigma X [nm]")
            self.Lsigma_y_nm.setText("Sigma Y [nm]")
            self.Lsigma_z_nm.setText("Sigma Z [nm]")
            self.current_size_mode = 1
        else:
            self.Esigma_y_nm.show()
            self.Esigma_z_nm.show()
            self.Lsigma_y_nm.show()
            self.Lsigma_z_nm.show()
            self.Lsigma_x_nm.setText("<Sigma X> [nm]")
            self.Lsigma_y_nm.setText("<Sigma Y> [nm]")
            self.Lsigma_z_nm.setText("<Sigma Z> [nm]")
            self.current_size_mode = 2


    def arrangement_mode_changed(self):
        if self.Carrangement_mode.currentText() == self.available_arrangement_modes[0]:
            self.current_arrangement_mode = 0
        elif self.Carrangement_mode.currentText() == self.available_arrangement_modes[1]:
            self.current_arrangement_mode = 1
        elif self.Carrangement_mode.currentText() == self.available_arrangement_modes[2]:
            self.current_arrangement_mode = 2
        else:
            self.current_arrangement_mode = 3

    def accept_evaluate_return_dataset(self):

        if self.current_arrangement_mode == 0:
            locs_pos_nm = self.create_line_dataset()
        elif self.current_arrangement_mode == 1:
            locs_pos_nm = self.create_grid_dataset()
        elif self.current_arrangement_mode == 2:
            locs_pos_nm = self.create_lattice_dataset()
        else:
            locs_pos_nm = self.create_random_dataset()

        if self.current_size_mode == 0:
            sigmas = np.ones((3, self.num_of_locs))
        elif self.current_size_mode == 1:
            sigmas = np.ones((3, self.num_of_locs))
            sigmas[0, :] *= float(self.Esigma_x_nm.text())
            sigmas[1, :] *= float(self.Esigma_y_nm.text())
            sigmas[2, :] *= float(self.Esigma_z_nm.text())

        else:
            sigmas = np.random.rand(3, self.num_of_locs) * 2
            sigmas[0, :] *= float(self.Esigma_x_nm.text())
            sigmas[1, :] *= float(self.Esigma_y_nm.text())
            sigmas[2, :] *= float(self.Esigma_z_nm.text())

        locs = np.rec.array(
            (np.arange(self.num_of_locs),
             locs_pos_nm[0, :],
             locs_pos_nm[1, :],
             locs_pos_nm[2, :],
             sigmas[0, :],
             sigmas[1, :],
             sigmas[2, :],
             np.ones(self.num_of_locs),)
            , dtype=LOCS_DTYPE)

        self.parent.get_dataset_from_test_mode(
            datasets=[LocalizationData(locs=locs, name="Generated in Testmode", pixelsize_nm=1,
                                      offset_pixels=(0, 0, 0), zdim_present=True,
                                      sigma_present=True, photon_count_present=False,
                                      parent=self.parent)])
        self.hide()

    def create_line_dataset(self):
        locs = np.ones((3, self.num_of_locs))
        locs[0, :] = np.arange(self.num_of_locs) * self.dist_of_locs_nm
        return locs

    def create_grid_dataset(self):
        self.num_of_locs = int(np.sqrt(self.num_of_locs)) ** 2
        locs = np.ones((3, self.num_of_locs))
        x = np.arange(int(np.sqrt(self.num_of_locs)))
        y = np.arange(int(np.sqrt(self.num_of_locs)))
        xv, yv = np.meshgrid(x, y)
        locs[0, :] = xv.flatten() * self.dist_of_locs_nm
        locs[1, :] = yv.flatten() * self.dist_of_locs_nm
        return locs

    def create_lattice_dataset(self):
        self.num_of_locs = int(np.cbrt(self.num_of_locs)) ** 3
        locs = np.ones((3, self.num_of_locs))
        x = np.arange(int(np.cbrt(self.num_of_locs)))
        y = np.arange(int(np.cbrt(self.num_of_locs)))
        z = np.arange(int(np.cbrt(self.num_of_locs)))
        xv, yv, zv = np.meshgrid(x, y, z)
        locs[0, :] = xv.flatten() * self.dist_of_locs_nm
        locs[1, :] = yv.flatten() * self.dist_of_locs_nm
        locs[2, :] = zv.flatten() * self.dist_of_locs_nm
        return locs

    def create_random_dataset(self):
        locs = np.random.rand(3, self.num_of_locs) * self.dist_of_locs_nm * np.cbrt(self.num_of_locs) * 2
        return locs


import sys
from PyQt5.QtWidgets import QApplication

"""if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TestModeWindow(parent=None)
    window.show()
    sys.exit(app.exec())"""

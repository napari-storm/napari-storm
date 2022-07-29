from PyQt5.QtWidgets import QMessageBox


def create_yes_no_dialog(message, additional_info=None):
    msg=QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(message)
    if additional_info is not None:
        msg.setInformativeText(additional_info)
        msg.setDetailedText("The details are as follows:")
    msg.setWindowTitle(message)
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    return_value = msg.exec_()
    if return_value == QMessageBox.Yes:
        return True
    elif return_value == QMessageBox.No:
        return False
    else:
        raise RuntimeError('The Messagebox does not like what you did here...')

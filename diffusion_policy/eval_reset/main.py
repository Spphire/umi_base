import sys
from PyQt6.QtWidgets import QApplication
from gui.robot_control_gui import RobotControlGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = RobotControlGUI()
    gui.show()
    sys.exit(app.exec())
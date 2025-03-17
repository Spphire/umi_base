import os
import threading
import time
from PyQt6.QtWidgets import (
    QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QMessageBox, QTextEdit, QGridLayout,
    QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QPainter
from PIL import Image, ImageFilter
from PIL.ImageQt import ImageQt
from robot.robot_controller import RobotController

class RobotControlGUI(QWidget):
    show_msg_signal = pyqtSignal(str, str)
    log_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.initUI()
        self.robot_thread = None
        self.settings = QSettings("YourCompany", "YourApp")
        self.last_transform_file = ""
        self.last_target_positions_file = ""

        self.load_settings()

    def initUI(self):
        self.resize(407, 601)

        self.setWindowTitle('RobotMove')
        self.setWindowIcon(QIcon('resources/icon.png'))

        self.robot_ip_label = QLabel('➢ Robot IP:   ')
        self.robot_ip_input = QLineEdit()

        self.robot_port_label = QLabel('➢ Robot Port:')
        self.robot_port_input = QLineEdit()
        self.robot_port_input.setText('8080')  # 默认端口

        self.transform_file_label = QLabel('📂 World Matrix:')
        self.transform_file_button = QPushButton('选择文件')
        self.transform_file_button.clicked.connect(self.select_transform_file)

        self.target_positions_file_label = QLabel('📂 Target Position:')
        self.target_positions_file_button = QPushButton('选择文件')
        self.target_positions_file_button.clicked.connect(self.select_target_positions_file)

        self.position_index_label = QLabel('🍀 Select target position number:')

        self.position_index_group = QButtonGroup(self)
        self.position_index_layout = QGridLayout()
        self.position_index_layout.setSpacing(5)
        row = 0
        col = 0
        for i in range(1, 21):
            radio_button = QRadioButton(str(i))
            radio_button.setStyleSheet("""
                QRadioButton {
                    font-size: 14px;
                    color: #000000;
                }
                QRadioButton::indicator {
                    width: 20px;
                    height: 20px;
                }
                QRadioButton::indicator:checked {
                    background-color: #87CEFA;
                    border-radius: 10px;
                }
            """)
            self.position_index_group.addButton(radio_button, i)
            self.position_index_layout.addWidget(radio_button, row, col)
            col += 1
            if col == 5:
                col = 0
                row += 1

        first_button = self.position_index_group.button(1)
        if first_button:
            first_button.setChecked(True)

        self.start_button = QPushButton('🏃   Start Moving')
        self.start_button.clicked.connect(self.start_robot_move)

        self.get_pos_button = QPushButton('✍   Get Pos')
        self.get_pos_button.clicked.connect(self.get_robot_position)

        self.log_label = QLabel('✍️  Logging:')
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.log_text_edit.setStyleSheet("background: transparent;")

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        h_layout_robot_ip = QHBoxLayout()
        h_layout_robot_ip.addWidget(self.robot_ip_label)
        h_layout_robot_ip.addWidget(self.robot_ip_input)
        main_layout.addLayout(h_layout_robot_ip)

        h_layout_robot_port = QHBoxLayout()
        h_layout_robot_port.addWidget(self.robot_port_label)
        h_layout_robot_port.addWidget(self.robot_port_input)
        main_layout.addLayout(h_layout_robot_port)

        h_layout_transform_file = QHBoxLayout()
        h_layout_transform_file.addWidget(self.transform_file_label)
        h_layout_transform_file.addWidget(self.transform_file_button)
        main_layout.addLayout(h_layout_transform_file)

        h_layout_target_positions_file = QHBoxLayout()
        h_layout_target_positions_file.addWidget(self.target_positions_file_label)
        h_layout_target_positions_file.addWidget(self.target_positions_file_button)
        main_layout.addLayout(h_layout_target_positions_file)

        main_layout.addWidget(self.position_index_label)
        main_layout.addLayout(self.position_index_layout)

        h_layout_buttons = QHBoxLayout()
        h_layout_buttons.addWidget(self.start_button)
        h_layout_buttons.addWidget(self.get_pos_button)
        main_layout.addLayout(h_layout_buttons)

        main_layout.addWidget(self.log_label)
        main_layout.addWidget(self.log_text_edit)

        self.setLayout(main_layout)

        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                font-family: 'Palatino Linotype', 'Book Antiqua', 'Garamond', serif;
                font-style: italic;
                font-weight: bold;
                color: #FF0800;
                font-size: 14px;
            }
            QPushButton {
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #FF7F7F;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
                min-width: 80px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #005F99;
            }
            QLineEdit {
                font-family: 'Segoe UI', Arial, sans-serif;
                padding: 5px;
                border: 2px solid #87CEFA;
                border-radius: 5px;
                min-width: 200px;
                background-color: rgba(255, 255, 255, 200);
                color: #000000;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #FF1493;
                background-color: rgba(255, 255, 255, 230);
            }
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                background: transparent;
                color: #D4D4D4;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
                min-height: 150px;
                font-size: 13px;
            }
            QRadioButton {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
                color: #000000;
            }
            QRadioButton::indicator {
                width: 20px;
                height: 20px;
            }
            QRadioButton::indicator:unchecked {
                border: 1px solid #87CEFA;
                border-radius: 10px;
                background-color: white;
            }
            QRadioButton::indicator:checked {
                background-color: #87CEFA;
                border: 1px solid #87CEFA;
                border-radius: 10px;
            }
        """)

        pil_image = Image.open("resources/background.jpg")
        blurred_pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=3))
        qt_image = ImageQt(blurred_pil_image)
        self.background_image = QPixmap.fromImage(qt_image)
        self.background_image = self.background_image.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.show_msg_signal.connect(self.display_message)
        self.log_signal.connect(self.append_log)

    def resizeEvent(self, event):
        self.background_image = self.background_image.scaled(self.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.background_image)
        super().paintEvent(event)

    def select_transform_file(self):
        file_dialog = QFileDialog()
        directory = os.path.dirname(self.last_transform_file) if self.last_transform_file else ""
        file_path, _ = file_dialog.getOpenFileName(self, '选择转换矩阵文件', directory, 'Text Files (*.txt);;All Files (*)')
        if file_path:
            self.last_transform_file = file_path
            file_name = os.path.basename(file_path)
            self.transform_file_button.setText(file_name)
            self.settings.setValue("last_transform_file", self.last_transform_file)

    def select_target_positions_file(self):
        file_dialog = QFileDialog()
        directory = os.path.dirname(self.last_target_positions_file) if self.last_target_positions_file else ""
        file_path, _ = file_dialog.getOpenFileName(self, '选择目标位置信息文件', directory, 'Text Files (*.txt);;All Files (*)')
        if file_path:
            self.last_target_positions_file = file_path
            file_name = os.path.basename(file_path)
            self.target_positions_file_button.setText(file_name)
            self.settings.setValue("last_target_positions_file", self.last_target_positions_file)

    def start_robot_move(self):
        if self.robot_thread and self.robot_thread.is_alive():
            QMessageBox.warning(self, '❗', '机器人正在移动，请勿重复启动。')
            return

        robot_ip = self.robot_ip_input.text()
        robot_port = self.robot_port_input.text()
        transform_file = self.last_transform_file
        target_positions_file = self.last_target_positions_file
        selected_button = self.position_index_group.checkedButton()
        position_index = self.position_index_group.id(selected_button) if selected_button else None

        if not robot_ip or not robot_port or not transform_file or not target_positions_file or not position_index:
            QMessageBox.warning(self, '❗', '请填写所有必填项并选择文件。')
            return

        self.settings.setValue("last_position_index", str(position_index))

        self.robot_controller = RobotController(robot_ip, robot_port, self)
        self.robot_thread = threading.Thread(target=self.robot_controller.move_robot_to_position,
                                             args=(transform_file, target_positions_file, position_index))
        self.robot_thread.start()

    def get_robot_position(self):
        if self.robot_thread and self.robot_thread.is_alive():
            QMessageBox.warning(self, '❗', '机器人正在运行，请稍后再试。')
            return

        robot_ip = self.robot_ip_input.text()
        robot_port = self.robot_port_input.text()
        target_positions_file = self.last_target_positions_file
        selected_button = self.position_index_group.checkedButton()
        position_index = self.position_index_group.id(selected_button) if selected_button else None

        if not robot_ip or not robot_port or not target_positions_file or not position_index:
            QMessageBox.warning(self, '❗', '请填写所有必填项并选择文件。')
            return

        self.settings.setValue("last_position_index", str(position_index))

        self.robot_controller = RobotController(robot_ip, robot_port, self)
        self.robot_thread = threading.Thread(target=self.robot_controller.get_robot_current_position,
                                             args=(target_positions_file, position_index))
        self.robot_thread.start()

    def show_message(self, title, message):
        self.show_msg_signal.emit(title, message)

    def display_message(self, title, message):
        QMessageBox.information(self, title, message)

    def log_message(self, message, level="info"):
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime())
        self.log_signal.emit(f"{timestamp} {message}", level)

    def append_log(self, message, level):
        if level == "info":
            color = "#87CEFA"
        elif level == "warning":
            color = "#FFA500"
        elif level == "error":
            color = "#FF0000"
        else:
            color = "#D4D4D4"

        html_message = f'<span style="color:{color};">{message}</span>'
        self.log_text_edit.append(html_message)

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def load_settings(self):
        self.robot_ip_input.setText(self.settings.value("robot_ip", ""))
        self.robot_port_input.setText(self.settings.value("robot_port", "8080"))
        self.last_transform_file = self.settings.value("last_transform_file", "")
        self.last_target_positions_file = self.settings.value("last_target_positions_file", "")
        last_position_index = self.settings.value("last_position_index", "")

        if last_position_index and last_position_index.isdigit():
            index = int(last_position_index)
            button = self.position_index_group.button(index)
            if button:
                button.setChecked(True)

        if self.last_transform_file:
            self.transform_file_button.setText(os.path.basename(self.last_transform_file))
        if self.last_target_positions_file:
            self.target_positions_file_button.setText(os.path.basename(self.last_target_positions_file))

    def save_settings(self):
        self.settings.setValue("robot_ip", self.robot_ip_input.text())
        self.settings.setValue("robot_port", self.robot_port_input.text())
        self.settings.setValue("last_transform_file", self.last_transform_file)
        self.settings.setValue("last_target_positions_file", self.last_target_positions_file)
        selected_button = self.position_index_group.checkedButton()
        position_index = self.position_index_group.id(selected_button) if selected_button else ""
        self.settings.setValue("last_position_index", str(position_index))
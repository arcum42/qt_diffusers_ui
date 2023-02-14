import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame
from PySide6 import QtCore

from PySide6.QtGui import QPixmap
from PIL.ImageQt import ImageQt


class output_ui(QWidget):
    def __init__(self):
        print("Init window")
        super().__init__()
        self.resize(512, 512)
        self.setWindowTitle("Output")
        self.setWindowFlags(
            QtCore.Qt.Window | QtCore.Qt.WindowTitleHint | QtCore.Qt.CustomizeWindowHint)

        self.image = list()

        self.layout = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.inner = QFrame(self.scroll)
        self.layout.addWidget(self.scroll, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.layout_scroll = QVBoxLayout(self.scroll)

    def clear(self):
        for x in self.image:
            x.destroy()
        self.image.clear()

    def add_image(self):
        list_size = len(self.image)
        self.image.append(QLabel("None"))
        self.image[list_size].setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image[list_size].resize(512, 512)
        self.layout_scroll.addWidget(self.image[list_size])

    def set_image(self, img, img_num):
        while len(self.image) < (img_num + 1):
            self.add_image()
        self.image[img_num].setPixmap(QPixmap.fromImage(ImageQt(img)))

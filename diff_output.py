import os
from PySide6.QtWidgets import QWidget,QVBoxLayout,QLabel
from PySide6 import QtCore

from PySide6.QtGui import QImage, QPixmap
from PIL import Image
from PIL.ImageQt import ImageQt

class output_ui(QWidget):
    def __init__(self):
        print("Init window")
        super().__init__()
        self.resize(512, 512)
        self.setWindowTitle("Output")
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowTitleHint | QtCore.Qt.CustomizeWindowHint)
        layout = QVBoxLayout()
        self.image = QLabel("None")
        self.image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setLayout(layout)


    def open(self):
        print("Open window.")


    def close(self):
        print("Close window.")


    def clear(self):
        print("Clear images.")


    def add_image(self, img):
        print("Add image")
        self.image.setPixmap(QPixmap.fromImage(ImageQt(img)))

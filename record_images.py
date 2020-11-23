# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi
from pathlib import Path


class record_images(QWidget):
    def __init__(self,ws,thread):
        super().__init__()
        loadUi('record_images.ui', self)
        self.btn_record.clicked.connect(self.record)
        self.ind = 0 # Indice of images
        self.ws = ws
        self.thread = thread
        self.mode_create = True # True means we have to create the categorie and its directories
        self.mode_record = False # True : record the image, False : stop recording

    def record(self):
        if self.mode_create:
            # create the 4 directories for this category
            for dir in ["images","data/train","data/val","data/test"]:
                (self.ws/dir/self.le_category_name.text()).mkdir()
            self.le_category_name.setEnabled(False)
            self.btn_record.setText("Start")
            self.mode_create = False
            self.mode_record = True
        elif self.mode_record:
            self.thread.start_recording(self.ind,self.ws/"images"/self.le_category_name.text())
            self.btn_record.setText("Stop")
            self.mode_record = False
        else:
            self.thread.stop_recording()
            self.ind = self.thread.ind_image
            self.lbl_nb_images.setText(str(self.ind))
            self.btn_record.setText("Start")
            self.mode_record = True
# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
import random
import time, os
from pathlib import Path
from thread_camera import *
from thread_command import *
from record_images import record_images
from trainer import Trainer
from onnx_export import ONNX_export
#
# Goal :
#

class GUI_hackathon_ai(QWidget):
    """
    GUI
    """

    def __init__(self, parent=None):
        super().__init__()
        loadUi('hackathon_ai.ui', self)
        self.ws = None
        # Change font, colour of text entry box
        self.txt_log.setStyleSheet(
            """QPlainTextEdit {background-color: #333;
                               color: #00FF00;
                               font-size: 8;
                               font-family: Courier;}""")
        # Check if there're some model directories
        models = next(os.walk('/home/nano/hackathon_AI'))[1]
        for d in ['.git','.idea','__pycache__']:  # We aren't interested by this folders (it's not models)
            if d in models:
                models.remove(d)
        if models:  # if there're some models on the Nano computer
            for m in models:
                self.cb_select_model.addItem(m)
            self.btn_inference.setEnabled(True)
            self.temp_name = models[0]  # we select the first one (but the user can replace it)
        # Definition of the threads
        self.thread_camera = Thread_Camera()
        self.thread_camera.signalAfficherImage.connect(self.display_image_from_camera)
        if self.thread_camera.sourceVideo and self.thread_camera.sourceVideo.isOpened():
            self.thread_camera.start()
        self.thread_command = Thread_Command()
        self.thread_command.display_msg_signal.connect(self.update_log)
        # Event handlers
        self.btn_working_space.clicked.connect(self.select_ws)
        self.sb_camera_id.valueChanged.connect(self.camera_changed)
        self.btn_project_ok.clicked.connect(self.create_project)
        self.btn_split_image.clicked.connect(self.split_images)
        self.btn_train_model.clicked.connect(self.train_model)
        self.btn_convert_onnx.clicked.connect(self.convert_to_onnx)
        self.btn_inference.clicked.connect(self.inference)
        self.cb_select_model.currentIndexChanged.connect(self.change_model)

    def change_model(self):
        self.ws = Path(self.cb_select_model.currentText())

    def select_ws(self):
        self.ws = Path(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.update_log('Selection of {} as working directory'.format(self.ws))

    def camera_changed(self):
        self.thread_camera.changeCamera(self.sb_camera_id.value())

    def display_image_from_camera(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        (height, width, _) = img.shape
        mQImage = QImage(img, width, height, QImage.Format_RGB888)
        pix = QPixmap.fromImage(mQImage)
        self.lbl_image.setPixmap(pix)

    def create_project(self):
        # Create directories in working space
        for d in ["images","data/val","data/train","data/test","model"]:
            dir = self.ws / d
            dir.mkdir(parents=True)
        nb_categories = self.sb_nb_categories.value()
        for i in range(nb_categories):
            ri = record_images(self.ws, self.thread_camera)
            self.vl_record.addWidget(ri)
        self.update_log('Project creation')
        self.gb_recording.setEnabled(True)
        self.btn_split_image.setEnabled(True)

    def split_images(self):
        # Creation of the labels.txt file
        lst_labels = []
        for cat_dir in (self.ws / 'images').iterdir():
            category = cat_dir.name
            lst_labels.append(category)
            files = [f for f in cat_dir.iterdir()]
            random.shuffle(files)
            ind_80 = int(len(files) * 0.8)  # 80 %
            ind_10 = int(len(files) * 0.1)  # 10 %
            self.split(files[:ind_80], 'train', category)
            self.split(files[ind_80:ind_80 + ind_10], 'val', category)
            self.split(files[ind_80 + ind_10:], 'test', category)
        labels_file = open(str(self.ws / 'model' / 'labels.txt'), 'w')
        lst_labels.sort()
        for l in lst_labels:
            labels_file.write(l + '\n')
        labels_file.close()
        self.update_log('Images split to train, val and test directories')
        self.btn_train_model.setEnabled(True)

    def split(self, files, dir_name, category):
        the_path = self.ws / "data" / dir_name / category
        for f in files:
            f.replace(the_path / f.name)

    def train_model(self):
        self.update_log("Begin training model")
        model_dir = self.ws / 'model'
        data_dir = self.ws / 'data'
        trainer = Trainer(str(model_dir),str(data_dir))
        trainer.main()
        self.btn_convert_onnx.setEnabled(True)

    def convert_to_onnx(self):
        # Now convert the model to a ONNX model
        self.update_log("Begin conversion to ONNX.")
        model_dir = self.ws / 'model'
        exporter = ONNX_export(model_dir)
        exporter.export()
        self.btn_inference.setEnabled(True)

    def inference(self):
        self.update_log("Begin inference. It may takes a long time. Be patient.")
        if not self.ws:  # Direct inference, no training before
            self.change_model()
        model_dir = self.ws / 'model'

        cmd = 'python3 classification.py --log-level=info --headless '
        cmd += '--model=' + str(model_dir / 'resnet18.onnx') + ' --input_blob=input_0 --output_blob=output_0 '
        cmd += '--labels=' + str(model_dir / 'labels.txt') + ' csi://0'
        self.exec_cmd(cmd, 'Starting inference')

    def exec_cmd(self, cmd, msg):
        print(cmd)
        self.thread_command.exec_command(cmd, msg)

    def update_log(self, txt):
        self.txt_log.appendPlainText(txt)

    def closeEvent(self, event):
        self.thread_camera.sourceVideo.release()
        cmd = 'python3 switch_off_led.py'
        self.exec_cmd(cmd, 'Switch off the LEDs')
        time.sleep(1)
        event.accept() # let the window close

#
# Main program
#
if __name__ == '__main__':
    # GUI
    app = QApplication([])
    gui = GUI_hackathon_ai()
    gui.show()
    app.exec_()

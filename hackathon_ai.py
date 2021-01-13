# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUi
from PyQt5.QtCore import QTimer
import random
import time, os
from pathlib import Path
from thread_camera import *
from record_images import record_images
from thread_trainer import Thread_Trainer
from thread_classification import Thread_Classification
from thread_onnx_export import Thread_ONNX_export
from io import StringIO
import sys

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
        # Redirection of stdout (print)
        sys.stdout = self.mystdout = StringIO()
        self.ws = None
        # Change font, colour of text entry box
        self.txt_log.setStyleSheet(
            """QPlainTextEdit {background-color: #333;
                               color: #00FF00;
                               font-size: 8;
                               font-family: Courier;}""")
        # Check if there're some project directories
        projects = next(os.walk('/home/nano/hackathon_AI/Projects'))[1]
        # for d in ['.git','.idea','__pycache__']:  # We aren't interested by this folders (it's not models)
        #     if d in models:
        #         models.remove(d)
        if projects:  # if there're some models on the Nano computer
            for m in projects:
                self.cb_select_model.addItem(m)
            self.btn_inference.setEnabled(True)
            self.temp_name = projects[0]  # we select the first one (but the user can replace it)
        # Definition of the threads
        self.thread_camera = Thread_Camera()
        self.thread_camera.signalAfficherImage.connect(self.display_image_from_camera)
        if self.thread_camera.sourceVideo:
            self.thread_camera.start()
        self.timer = QTimer()
        self.timer.timeout.connect(self.write)
        # Event handlers
        self.btn_working_space.clicked.connect(self.select_ws)
        self.btn_project_ok.clicked.connect(self.create_project)
        self.btn_split_image.clicked.connect(self.split_images)
        self.btn_train_model.clicked.connect(self.train_model)
        self.btn_convert_onnx.clicked.connect(self.convert_to_onnx)
        self.btn_inference.clicked.connect(self.inference)
        self.cb_select_model.currentIndexChanged.connect(self.change_model)
        self.lst_epochs = []
        self.lst_accuracys = []
        self.lst_loss = []
        self.generate_plots()
        self.timer.start(500)

    def generate_plots(self):
        # generate the plots
        self.accuracy_ax = self.accuracy_plot.canvas.ax
        self.loss_ax = self.loss_plot.canvas.ax
        # set specific limits for X axes
        self.accuracy_ax.set_xlim(1, self.sb_epochs)
        self.accuracy_ax.set_autoscale_on(True)
        self.loss_ax.set_xlim(1, self.sb_epochs)
        self.loss_ax.set_autoscale_on(True)
        self.plot_accuracy, = self.accuracy_ax.plot(self.lst_epochs, self.lst_accuracys,label="Accuracy")
        self.plot_loss, = self.loss_ax.plot(self.lst_epochs, self.lst_loss,label="Loss")
        # generate the canvas to display the plots
        self.gv_accuracy.canvas.draw_idle()  # draw_idle() à la place de draw() pour corriger un pb de refresh avec MAC
        self.gv_loss.canvas.draw_idle()  # draw_idle() à la place de draw() pour corriger un pb de refresh avec MAC

    def change_model(self):
        self.ws = Path('Projects/' + self.cb_select_model.currentText())

    def select_ws(self):
        self.ws = Path(QFileDialog.getExistingDirectory(self, caption="Select Directory",directory='Projects'))
        self.lbl_working_space.setText(str(self.ws))
        print('Selection of {} as working directory'.format(self.ws))

    def display_image_from_camera(self, img, ind):
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
        print('End : Project creation')
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
        print('End : Images split to train, val and test directories')
        self.btn_train_model.setEnabled(True)

    def split(self, files, dir_name, category):
        the_path = self.ws / "data" / dir_name / category
        for f in files:
            f.replace(the_path / f.name)

    def train_model(self):
        print("Begin : Training model")
        model_dir = self.ws / 'model'
        data_dir = self.ws / 'data'
        nb_epochs = self.sb_epochs.value()
        self.trainer = Thread_Trainer(str(model_dir), str(data_dir), nb_epochs)
        self.trainer.signalEndTraining.connect(self.end_training)
        self.trainer.signalAccuracyLossData.connect(self.add_data_to_plots)
        self.trainer.start()

    def end_training(self):
        print("End : Training model")
        self.btn_convert_onnx.setEnabled(True)

    def add_data_to_plots(self,epoch,loss,accuray):
        print('Epoch {}, loss {}, accuracy {}'.format(epoch,loss,accuray))
        self.lst_epochs.append(epoch)
        self.lst_loss.append(loss)
        self.lst_accuracys.append(accuray)
        self.plot_loss.set_data(self.lst_epochs, self.lst_loss) # Update loss plot
        self.gv_loss.canvas.draw_idle()
        self.plot_accuracy.set_data(self.lst_epochs, self.lst_accuracys) # Update loss plot
        self.gv_accuracy.canvas.draw_idle()

    def convert_to_onnx(self):
        # Now convert the model to a ONNX model
        print("Begin : Conversion to ONNX.")
        model_dir = self.ws / 'model'
        self.exporter = Thread_ONNX_export(model_dir)
        self.exporter.signalEndExport.connect(self.end_convert)
        self.exporter.start()

    def end_convert(self):
        print("End : Conversion to ONNX.")
        self.btn_inference.setEnabled(True)

    def inference(self):
        print("Begin : Inference. It may takes a long time. Be patient.")
        if not self.ws:  # Direct inference, no training before
            self.change_model()
        model_dir = self.ws / 'model'
        self.thread_classification = Thread_Classification(str(model_dir),self.thread_camera.sourceVideo)
        self.thread_classification.start()

    def update_log(self, txt):
        print('\n'+txt)

    def write(self):
        txt = str(self.mystdout.getvalue())
        if txt:
            self.txt_log.appendPlainText(txt)
            self.mystdout.close()
            sys.stdout = self.mystdout = StringIO()

    def closeEvent(self, event):
        print("STOP")
        self.thread_camera.camera_running = False
        if self.thread_classification:
            self.thread_classification.classification_OK = False
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

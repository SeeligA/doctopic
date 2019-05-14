import sys

from PyQt5 import uic
from PyQt5.QtWidgets import qApp, QApplication, QFileDialog, QMainWindow, QWidget
from PyQt5 import QtCore


from PyQt5.QtGui import QIcon
import os.path
from doctopic import MyCorpus, load_from_folder, file_to_query, build_index
#from GUI.restore import guisave, guirestore
import logging
import tempfile
from shutil import copy2

TEMP_FOLDER = tempfile.gettempdir()

class MyWindow(QMainWindow):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.initUI()

        self.settings = QtCore.QSettings("DocFirm Inc.", "docfind")
        # Retrieve user input from settings
        save_folder = self.settings.value("savedFolder", "")
        self.input_line_edit_project.setText(save_folder)
        save_doc = self.settings.value("savedDoc", "")
        self.input_line_edit_doc.setText(save_doc)
        # Write new user input to settings
        self.input_line_edit_project.textChanged.connect(self.new_folder_changed)
        self.input_line_edit_doc.textChanged.connect(self.new_doc_changed)

    def initUI(self):

        uic.loadUi(os.path.join('GUI', 'doctopic.ui'), self)
        self.setWindowTitle('DocTopic')
        self.setContentsMargins(7,0,7,0)
        self.statusBar().showMessage('Ready')

        self.show()

    def new_folder_changed(self, newFolder):
        '''Store dirpath to settings'''
        self.settings.setValue("savedFolder", w.input_line_edit_project.text())

    def new_doc_changed(self, newDoc):
        '''Store filepath to settings'''
        self.settings.setValue("savedDoc", w.input_line_edit_doc.text())

    def open_project(self):
        '''Open file dialog and write dirpath to line edit'''
        dirpath = QFileDialog.getExistingDirectory()
        w.input_line_edit_project.setText(dirpath)

    def open_doc(self):
        '''Open file dialog and write filepath to line edit'''
        fp = QFileDialog.getOpenFileName(filter = 'TXT-Datei (*.txt)')[0]
        w.input_line_edit_doc.setText(fp)

    def run_query(self):
        '''
        '''
        # Check
        if hasattr(w, 'model'):
            pass
        else:
            w.model, w.dictionary, w.index, w.labels = load_from_folder(w.input_line_edit_project.text())

            if all((w.model, w.dictionary, w.index, w.labels)) == False:
                logging.info('UnboundLocalError: Unable to find local variable')
                w.textOutput.setText('UnboundLocalError: Unable to find local variable')
                return

        if os.path.isfile(w.input_line_edit_doc.text()):
            vec_bow = file_to_query(w.input_line_edit_doc.text(), w.dictionary)
            vec_lsi = w.model[vec_bow]
            sims_lsi = w.index[vec_lsi]
            sims_lsi = sorted(enumerate(sims_lsi), key=lambda item: -item[1])

            topic = w.model.print_topic(max(vec_lsi, key=lambda item: abs(item[1]))[0])

            w.print_details(sims_lsi, topic)
        else:
            w.textOutput.setText('Please select a valid file')

    def print_details(self, sims, topic):
        '''Print query results to text output'''
        dash = '-' * 119

        self.textOutput.setText(dash)
        self.textOutput.append('{:<5s}{:>8s}{:>12}{:>12s}'.format('RANK', 'SIM', 'CLIENT', 'FILE'))
        self.textOutput.append(dash)

        for i in range(10):
            labels = w.labels[str(sims[i][0])]
            self.textOutput.append('{:^10}{:.4}\t{:>12.7}\t{:<12s}'.format(i+1, sims[i][1], labels[0], labels[1]))

        self.textOutput.append('\n')
        self.textOutput.append('{} Most Prominent LSI topic {}'.format('-'*43,'-'*43))
        self.textOutput.append(topic)
        self.textOutput.append(dash)

    def reset_model(self):
        '''Delete attribute when changing the project folder'''
        logging.info('Reset model')
        if hasattr(w, 'model'):
            del w.model


    def train_model(self):
        if os.path.isdir(w.input_line_edit_doc.text()):
            corpus = MyCorpus(w.input_line_edit_doc.text())
            _ = corpus.save_to_temp()
            _ = corpus.get_labels()

            tfidf, corpus_tfidf, lsi, corpus_lsi = corpus.build_lsi(topics=250)
            index_lsi = build_index(list(corpus_lsi))
            w.textOutput_train.setText('Training complete!')

        else:
            w.textOutput_train.setText('Please select a valid folder')

    def save_model(self):
        src = TEMP_FOLDER
        dst = w.input_line_edit_project.text()
        files = ['tmp.index', 'tmp.lsi', 'tmp.lsi.projection', 'tmp.dict', 'tmp.json']
        for file in files:
            copy2(os.path.join(src, file), os.path.join(dst, file))
            logging.info('Copying {} to {}'.format(file, dst))
        w.textOutput_train.append('Files saved')


if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = MyWindow()
    w.open_project_button.clicked.connect(MyWindow.open_project)
    w.open_doc_button.clicked.connect(MyWindow.open_doc)
    w.run_query_button.clicked.connect(MyWindow.run_query)
    w.actionOpen_Project.triggered.connect(MyWindow.open_project)
    w.actionExit.triggered.connect(qApp.quit)
    w.input_line_edit_project.textChanged.connect(MyWindow.reset_model)
    w.train_model_button.clicked.connect(MyWindow.train_model)
    w.save_project_button.clicked.connect(MyWindow.save_model)


    sys.exit(app.exec_())

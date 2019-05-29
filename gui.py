import sys

from qtpy import QtCore, uic
from qtpy.QtWidgets import qApp, QApplication, QFileDialog, QMainWindow

import os.path
from doctopic import MyCorpus, load_from_folder, file_to_query, build_index

import logging
import tempfile
from shutil import copy2

TEMP_FOLDER = tempfile.gettempdir()


class MyWindow(QMainWindow):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.initUI()
        # Load settings
        self.settings = QtCore.QSettings("DocFirm Inc.", "docfind")
        # Retrieve user input from settings
        save_folder = self.settings.value("savedFolder", "")
        self.input_line_edit_model.setText(save_folder)
        save_doc = self.settings.value("savedDoc", "")
        self.input_line_edit_doc.setText(save_doc)
        # Write new user input to settings
        self.input_line_edit_model.textChanged.connect(self.new_folder_changed)
        self.input_line_edit_doc.textChanged.connect(self.new_doc_changed)

    def initUI(self):

        uic.loadUi(os.path.join('GUI', 'doctopic.ui'), self)
        self.setWindowTitle('DocTopic')
        self.setContentsMargins(7, 0, 7, 0)
        self.statusBar().showMessage('Ready')

        self.show()

    def new_folder_changed(self, newFolder):
        """Store dirpath to settings"""
        self.settings.setValue("savedFolder", w.input_line_edit_model.text())

    def new_doc_changed(self, newDoc):
        """Store filepath to settings"""
        self.settings.setValue("savedDoc", w.input_line_edit_doc.text())

    def open_model(self):
        """Open file dialog and write dirpath to line edit."""
        dirpath = QFileDialog.getExistingDirectory()
        w.input_line_edit_model.setText(dirpath)

    def open_doc(self):
        """Open file dialog and write filepath to line edit."""
        fp = QFileDialog.getOpenFileName(filter='TXT-Datei (*.txt)')[0]
        w.input_line_edit_doc.setText(fp)

    def run_query(self):
        """Run query against model."""

        # Check if a model has been already loaded
        if hasattr(w, 'model'):
            pass
        # Load relevant files from model directory
        else:
            w.model, w.dictionary, w.tfidf, w.sparse_index, w.index, w.labels = load_from_folder(
                w.input_line_edit_model.text())
            # Abort query if one or more files are missing
            if all((w.model, w.dictionary, w.tfidf, w.sparse_index, w.index, w.labels)) == False:
                logging.info('UnboundLocalError: Unable to find local variable')
                w.textOutput.setText('UnboundLocalError: Unable to find local variable')
                return

        # Check if a query document has been specified
        if os.path.isfile(w.input_line_edit_doc.text()):
            # Convert file to BOW vector
            vec_bow = file_to_query(w.input_line_edit_doc.text(), w.dictionary)
            # Serialize tfidf transformation and convert search vector to LSI space
            # Note: When using transformed search vectors, apply same transformation when building the index
            vec_lsi = w.model[w.tfidf[vec_bow]]
            # Apply search vector to indexed LSI corpus and sort resulting index-similarity tuples.
            sims_lsi = w.index[vec_lsi]
            sims_lsi = sorted(enumerate(sims_lsi), key=lambda item: -item[1])
            # Retrieve most prominent topic from search vector
            topic = w.model.print_topic(max(vec_lsi, key=lambda item: abs(item[1]))[0])
            # doc_topics, word_topics, phi_values, = model.get_document_topics(vec_bow, per_word_topics=True)
            # Apply search vector to transformed tfidf corpus and sort resulting index-similarity tuples
            sims_tfidf = w.sparse_index[w.tfidf[vec_bow]]
            sims_tfidf = sorted(enumerate(sims_tfidf), key=lambda item: -item[1])

            w.print_details(sims_lsi, sims_tfidf, topic)
        else:
            w.textOutput.setText('Please select a valid file')

    def print_details(self, sims, sims_tfidf, topic):
        """Print query results to text output."""

        sections = [' LSI Similarity ', ' Most Prominent LSI topic ', ' tf-idf Similarity ']
        headers = '{:<5s}{:>8s}{:>20s}{:>20s}{:>27s}'.format('RANK', 'SIM', 'CLIENT', 'PROJECT', 'FILE')
        line = 120

        self.textOutput.setText('{:-{align}{width}}'.format(sections[0], align='^', width=line))
        self.textOutput.append(headers)
        self.textOutput.append('{:-{align}{width}}'.format('', align='^', width=line))
        #
        for i in range(min(len(sims), 10)):
            labels = w.labels[str(sims[i][0])]
            self.textOutput.append('{:<10d}{: 8.3f}\t{:>14.12}{:>14.12}\t{:<12s}'.format(i + 1, sims[i][1], labels[0], labels[1], labels[-1]))

        self.textOutput.append('\n')
        self.textOutput.append('{:-{align}{width}}'.format(sections[1], align='^', width=line))
        self.textOutput.append(topic)

        self.textOutput.append('\n')
        self.textOutput.append('{:-{align}{width}}'.format(sections[2], align='^', width=line))
        self.textOutput.append(headers)
        self.textOutput.append('{:-{align}{width}}'.format('', align='^', width=line))

        for i in range(min(len(sims_tfidf), 10)):
            labels = w.labels[str(sims_tfidf[i][0])]
            self.textOutput.append(
                '{:<10d}{: 8.3f}\t{:>14.12}{:>14.12}\t{:<12s}'.format(i + 1, sims_tfidf[i][1], labels[0], labels[1], labels[-1]))


    def reset_model(self):
        """Delete attribute when changing the model folder."""

        if hasattr(w, 'model'):
            del w.model
            logging.info('Model reset')


    def train_model(self):
        if os.path.isdir(w.input_line_edit_doc.text()):
            corpus = MyCorpus(w.input_line_edit_doc.text())
            corpus.save_to_temp()
            #_ = corpus.get_labels()

            tfidf, corpus_tfidf, lsi, corpus_lsi = corpus.build_lsi(topics=250)
            # build LSI index
            _ = build_index(list(corpus_lsi))
            # build tfidf index
            _ = build_index(list(corpus_tfidf), num_features=len(corpus.dictionary))
            w.textOutput_train.setText('Training complete!')

        else:
            w.textOutput_train.setText('Please select a valid folder')

    def add_documents(self):
        pass

    @staticmethod
    def save_model():
        src = TEMP_FOLDER
        dst = w.input_line_edit_model.text()  # TODO: Check if the path is a valid folder
        # list of files expected in your TEMP folder after training a model
        files = ['sparse.index',
                 'tmp.index',
                 'tmp.lsi',
                 'tmp.tfidf',
                 'tmp.lsi.projection',
                 'tmp.dict',
                 'tmp.json']
        for file in files:
            copy2(os.path.join(src, file), os.path.join(dst, file))
            logging.info('Copying {} to {}'.format(file, dst))
        w.textOutput_train.append('Files saved')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    w.open_model_button.clicked.connect(MyWindow.open_model)
    w.open_doc_button.clicked.connect(MyWindow.open_doc)
    w.run_query_button.clicked.connect(MyWindow.run_query)
    w.actionOpen_Model.triggered.connect(MyWindow.open_model)
    w.actionExit.triggered.connect(qApp.quit)
    w.input_line_edit_model.textChanged.connect(MyWindow.reset_model)
    w.train_model_button.clicked.connect(MyWindow.train_model)
    w.save_model_button.clicked.connect(MyWindow.save_model)
    w.add_documents_button.clicked.connect(MyWindow.add_documents)

    sys.exit(app.exec_())

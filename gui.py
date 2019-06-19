import logging
import os.path
import sys
from collections import Counter

import numpy as np

from qtpy import QtCore, uic
from qtpy.QtCore import Qt
from qtpy.QtWidgets import qApp, QApplication, QFileDialog, QMainWindow, QMessageBox, QTreeWidgetItem, QHeaderView, \
    QTreeWidgetItemIterator

from sources.doctopic import MyCorpus, load_from_folder, file_to_query, build_index, merge_labels, iter_documents, \
    update_indices, TEMP_FOLDER

from sources.utils import make_model_archive, move_from_temp


class MyWindow(QMainWindow):

    def __init__(self):
        super(MyWindow, self).__init__()
        self.init_ui()
        # Load settings
        self.settings = QtCore.QSettings("DocFirm Inc.", "docfind")
        # Retrieve user input from settings
        save_folder = self.settings.value("savedFolder", "")
        self.input_line_edit_model.setText(save_folder)
        # save_doc = self.settings.value("savedDoc", "")
        # self.input_line_edit_query.setText(save_doc)
        # Write new user input to settings
        self.input_line_edit_model.textChanged.connect(self.new_folder_changed)
        # self.input_line_edit_query.textChanged.connect(self.new_doc_changed)

    def new_folder_changed(self, newFolder):
        """Store dirpath to settings"""
        self.settings.setValue("savedFolder", w.input_line_edit_model.text())

    def init_ui(self):

        uic.loadUi(os.path.join('GUI', 'doctopic.ui'), self)
        self.setWindowTitle('DocTopic')
        self.setContentsMargins(7, 0, 7, 0)
        self.statusBar().showMessage('Ready')

        self.show()

    # def new_doc_changed(self):
        # """Store filepath to settings"""
        # self.settings.setValue("savedDoc", w.input_line_edit_query.text())

    @staticmethod
    def open_model():
        """Open file dialog and write dirpath to line edit."""
        dirpath = QFileDialog.getExistingDirectory()
        w.input_line_edit_model.setText(dirpath)

    @staticmethod
    def open_folder():
        """Open file dialog and write dirpath to line edit."""
        dirpath = QFileDialog.getExistingDirectory()
        w.input_line_edit_train.setText(dirpath)

    @staticmethod
    def open_doc():
        """Open file dialog and write filepath(s) to list widget."""
        w.query_list_widget.clear()
        fp = QFileDialog.getOpenFileNames(filter='TXT-Datei (*.txt)')[0]
        w.query_list_widget.addItems(fp)

    @staticmethod
    def load_model():
        # Check if a model has been already loaded
        if hasattr(w, 'lsi'):
            pass
        # Load relevant files from model directory
        else:
            logging.info('Loading model parameters')
            params = ['lsi', 'dictionary', 'tfidf', 'tfidf_index', 'lsi_index', 'labels', 'corpus']
            load = load_from_folder(params, w.input_line_edit_model.text())

            for k, v in load.items():
                setattr(w, k, v)

    def run_index_query(self):
        """ Run cross-comparison on checked files."""

        w.textOutput_index.clear()
        w.load_model()
        msk = np.zeros(len(w.labels), dtype='bool')

        # Retrieve index number of files checked in the tree view
        idx = w.get_checked()
        # Create a boolean mask from indeces
        for i in idx:
            msk[i] = True

        sections = [' LSI Similarity ', ' Most Prominent LSI topic ', ' tf-idf Similarity ']
        headers = '{:<5s}{:>8s}{:>20s}{:>20s}{:>27s}'.format('RANK', 'SIM', 'CLIENT', 'PROJECT', 'FILE')
        line = 120
        dashed = '{:-{align}{width}}'.format('', align='^', width=line)

        for i in idx:
            w.textOutput_index.append('Printing results for {}'.format(w.labels[str(i)]))

            # Apply mask on indexed file
            sims_lsi = list(w.lsi_index)[i][msk]
            sims_tfidf = list(w.tfidf_index)[i][msk]

            # Add index number to scores for reference purposes
            sims_lsi = zip(sims_lsi, idx)
            sims_tfidf = zip(sims_tfidf, idx)

            # Sort tuples by score
            sims_lsi = sorted(sims_lsi, key=lambda item: -item[0])
            sims_tfidf = sorted(sims_tfidf, key=lambda item: -item[0])

            w.textOutput_index.append('{:-{align}{width}}'.format(sections[0], align='^', width=line))
            w.textOutput_index.append(headers)
            w.textOutput_index.append(dashed)

            for j in range(len(sims_lsi)):
                # load label from
                labels = w.labels[str(sims_lsi[j][1])]

                w.textOutput_index.append('{:<10d}{: 8.3f}\t{:>14.12}{:>14.12}\t{:<12s}'
                                          .format(j + 1, sims_lsi[j][0], labels[0], labels[1], labels[-1]))

            w.textOutput_index.append('\n')
            w.textOutput_index.append('{:-{align}{width}}'.format(sections[2], align='^', width=line))
            w.textOutput_index.append(headers)
            w.textOutput_index.append(dashed)

            for j in range(len(sims_tfidf)):
                labels = w.labels[str(sims_tfidf[j][1])]

                w.textOutput_index.append(
                    '{:<10d}{: 8.3f}\t{:>14.12}{:>14.12}\t{:<12s}'
                    .format(j + 1, sims_tfidf[j][0], labels[0], labels[1], labels[-1]))
            w.textOutput_index.append('\n')

    @staticmethod
    def run_query():
        """Run query against model."""

        w.textOutput.clear()

        w.load_model()
        #  Todo: Implement batch querying for improved performance
        for path in [str(w.query_list_widget.item(i).text()) for i in range(w.query_list_widget.count())]:
            basename = os.path.basename(path)
            w.textOutput.append('Printing results for {}'.format(basename))

            if os.path.isfile(path):
                vec_bow = file_to_query(path, w.dictionary)
                vec_bow = w.tfidf[vec_bow]
                # Serialize tfidf transformation and convert search vector to LSI space
                # Note: When using transformed search vectors, apply same transformation when building the index
                vec_lsi = w.lsi[vec_bow]

                # Apply search vector to indexed LSI corpus and sort resulting index-similarity tuples.
                sims_lsi = w.lsi_index[vec_lsi]
                sims_lsi = sorted(enumerate(sims_lsi), key=lambda item: -item[1])
                # Retrieve most prominent topic from search vector
                topic = w.lsi.print_topic(max(vec_lsi, key=lambda item: abs(item[1]))[0])

                # Apply search vector to transformed tfidf corpus and sort resulting index-similarity tuples
                sims_tfidf = w.tfidf_index[vec_bow]
                sims_tfidf = sorted(enumerate(sims_tfidf), key=lambda item: -item[1])

                w.print_details(sims_lsi, sims_tfidf, topic)

            else:
                w.textOutput.setText('{} not found. Please select a valid file.'.format(basename))

    def load_tree(self):
        """Build a index tree view from labels items."""

        w.treeWidget.clear()
        w.treeWidget.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        # Load model parameters including json dictionary with corpus metadata
        w.load_model()
        # Create lists from corpus metadata for populating tree cells
        clients = [v[0] for v in w.labels.values()]
        projects = [v[1] for v in w.labels.values()]
        files_idx = [(v[2], k) for k, v in w.labels.items()]

        # Create unique keys from list items for reference purposes
        c_cnt = Counter(clients)
        p_cnt = Counter(zip(clients, projects))
        f_cnt = Counter(zip(clients, projects, files_idx))
        c_dict, p_dict, f_dict = dict(), dict(), dict()

        # Iterate over counter items to create client items
        for key, count in c_cnt.items():
            c_dict[key] = QTreeWidgetItem(w.treeWidget, [key, str(count)])
            c_dict[key].setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
            c_dict[key].setExpanded(True)
        # Add project children to client items
        for key, count in p_cnt.items():
            p_dict[key] = QTreeWidgetItem(c_dict[key[0]], [key[1], str(count)])
            p_dict[key].setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
            # p_dict[key].setCheckState(0, Qt.Unchecked)
        # Add file children to project items
        for key in f_cnt.keys():
            f_dict[key] = QTreeWidgetItem(p_dict[key[:2]], [key[2][0], "", key[2][1]])
            f_dict[key].setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsEditable | Qt.ItemIsSelectable)
            f_dict[key].setCheckState(0, Qt.Unchecked)

    @staticmethod
    def get_checked():
        """Iterate over tree items and return keys of checked files."""

        iterator = QTreeWidgetItemIterator(w.treeWidget)
        idx = []
        while iterator.value():
            item = iterator.value()
            if item.checkState(0) == Qt.Checked:
                # Convert string index to integer and add to list
                idx.append(int(item.text(2)))
            iterator += 1

        return sorted(idx)

    def print_details(self, sims_lsi, sims_tfidf, topic):
        """Print query results to text output."""

        sections = [' LSI Similarity ', ' Most Prominent LSI topic ', ' tf-idf Similarity ']
        headers = '{:<5s}{:>8s}{:>20s}{:>20s}{:>27s}'.format('RANK', 'SIM', 'CLIENT', 'PROJECT', 'FILE')
        line = 120
        dashed = '{:-{align}{width}}'.format('', align='^', width=line)

        self.textOutput.append('{:-{align}{width}}'.format(sections[0], align='^', width=line))
        self.textOutput.append(headers)
        self.textOutput.append(dashed)

        for i in range(min(len(sims_lsi), 10)):
            labels = w.labels[str(sims_lsi[i][0])]
            self.textOutput.append('{:<10d}{: 8.3f}\t{:>14.12}{:>14.12}\t{:<12s}'
                                   .format(i + 1, sims_lsi[i][1], labels[0], labels[1], labels[-1]))

        self.textOutput.append('\n')
        self.textOutput.append('{:-{align}{width}}'.format(sections[1], align='^', width=line))
        self.textOutput.append(topic)

        self.textOutput.append('\n')
        self.textOutput.append('{:-{align}{width}}'.format(sections[2], align='^', width=line))
        self.textOutput.append(headers)
        self.textOutput.append(dashed)

        for i in range(min(len(sims_tfidf), 10)):
            labels = w.labels[str(sims_tfidf[i][0])]
            self.textOutput.append(
                '{:<10d}{: 8.3f}\t{:>14.12}{:>14.12}\t{:<12s}'
                .format(i + 1, sims_tfidf[i][1], labels[0], labels[1], labels[-1]))
        self.textOutput.append('\n')

    @staticmethod
    def reset_model():
        """Delete attribute when changing the model folder."""

        if hasattr(w, 'lsi'):
            del w.lsi
            logging.info('Model reset')

    @staticmethod
    def train_model():
        src = w.input_line_edit_train.text()
        dst = [os.path.join(TEMP_FOLDER, name) for name in ['lsi.index', 'tfidf.index']]

        if os.path.isdir(src):
            corpus = MyCorpus(src)
            corpus.save_to_temp()
            num_features = len(corpus.dictionary)

            tfidf, corpus_tfidf, lsi, corpus_lsi = corpus.build_lsi(topics=250)

            # build LSI index and save to temp folder
            lsi_index = build_index(dst[0], corpus_lsi, num_features)
            lsi_index.save(dst[0])
            # build tfidf index and save to temp folder
            tfidf_index = build_index(dst[1], corpus_tfidf, num_features)
            tfidf_index.save(dst[1])

            w.textOutput_train.setText('Training complete! Click "Save Project" to save parameters to model folder')
            w.save_model_button.setEnabled(True)

        else:
            w.textOutput_train.setText('Please select a valid training folder')

        w.reset_model()

    @staticmethod
    def add_docs():

        dst = w.input_line_edit_model.text()
        src = w.input_line_edit_train.text()
        if os.path.isdir(dst) and os.path.isdir(src):

            # Prompt user before updating indices
            dialog = QMessageBox.question(w, 'DocTopic', 'This operation will change your search indices. Continue?',
                                          QMessageBox.Save | QMessageBox.Cancel, QMessageBox.Cancel)
            if dialog == QMessageBox.Save:

                cnt = update_indices(src, dst)

                w.textOutput_train.setText('{} documents added!'.format(cnt))
                w.save_model_button.setDisabled(True)
                w.reset_model()
            else:
                pass

        else:
            w.textOutput_train.setText('Please select a valid folder')

    @staticmethod
    def save_model():
        """Saving files created during training to model folder."""

        dst = w.input_line_edit_model.text()

        # Check for files in model dir and zip folder if any
        if os.listdir(dst):
            archive = make_model_archive(dst)
            w.textOutput_train.append('Backup of model folder created here: {}'.format(archive))
        else:
            pass

        # Copy model files to model folder
        cnt = move_from_temp(TEMP_FOLDER, dst)

        w.textOutput_train.append('{} files saved'.format(cnt))
        w.save_model_button.setDisabled(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    w.add_documents_button.clicked.connect(MyWindow.add_docs)
    w.open_model_button.clicked.connect(MyWindow.open_model)
    w.open_doc_button.clicked.connect(MyWindow.open_doc)
    w.open_folder_button.clicked.connect(MyWindow.open_folder)
    w.run_query_button.clicked.connect(MyWindow.run_query)
    w.actionOpen_Model.triggered.connect(MyWindow.open_model)
    w.actionExit.triggered.connect(qApp.quit)
    w.input_line_edit_model.textChanged.connect(MyWindow.reset_model)
    w.train_model_button.clicked.connect(MyWindow.train_model)
    w.save_model_button.clicked.connect(MyWindow.save_model)
    w.load_tree_button.clicked.connect(MyWindow.load_tree)
    w.dummy_button.clicked.connect(MyWindow.run_index_query)

    sys.exit(app.exec_())

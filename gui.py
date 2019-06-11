import sys
import logging
import os.path
import shutil

from qtpy import QtCore, uic
from qtpy.QtCore import Qt
from qtpy.QtWidgets import qApp, QApplication, QFileDialog, QMainWindow, QMessageBox, QTreeWidgetItem, QHeaderView, QTreeWidgetItemIterator

from gensim import corpora, models, similarities
from collections import Counter
import numpy as np

from doctopic import MyCorpus, load_from_folder, file_to_query, build_index, merge_labels, iter_documents, TEMP_FOLDER, load_labels



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
        #self.input_line_edit_query.setText(save_doc)
        # Write new user input to settings
        self.input_line_edit_model.textChanged.connect(self.new_folder_changed)
        #self.input_line_edit_query.textChanged.connect(self.new_doc_changed)

    def new_folder_changed(self, newFolder):
        """Store dirpath to settings"""
        self.settings.setValue("savedFolder", w.input_line_edit_model.text())

    def initUI(self):

        uic.loadUi(os.path.join('GUI', 'doctopic.ui'), self)
        self.setWindowTitle('DocTopic')
        self.setContentsMargins(7, 0, 7, 0)
        self.statusBar().showMessage('Ready')

        self.show()

    def new_doc_changed(self):
        """Store filepath to settings"""
        #self.settings.setValue("savedDoc", w.input_line_edit_query.text())

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
            w.lsi, w.dictionary, w.tfidf, w.tfidf_index, w.lsi_index, w.labels, w.corpus = load_from_folder(
                w.input_line_edit_model.text())
            # Abort query if one or more files are missing
            if not all((w.lsi, w.dictionary, w.tfidf, w.tfidf_index, w.lsi_index, w.labels, w.corpus)):
                w.textOutput.setText('UnboundLocalError: Unable to find local variable')
                return None


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
            c_dict[key].setExpanded(1)
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

            dialog = QMessageBox.question(w, 'DocTopic', 'This operation will change your search indices. Continue?',
                                          QMessageBox.Save | QMessageBox.Cancel, QMessageBox.Cancel)
            if dialog == QMessageBox.Save:

                # Load dictionary and create corpus
                dictionary = corpora.Dictionary.load(os.path.join(dst, 'tmp.dict'))
                # Create list of BOW from training documents
                train_corpus = [dictionary.doc2bow(tokens) for tokens in iter_documents(src)]

                corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'tmp.mm'), train_corpus)
                train_corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'tmp.mm'))


                # Fold training dictionary into model dictionary
                #train_to_dict = dictionary.merge_with(train_dict)

                # Merge corpora
                #dictionary.save(os.path.join(dst, 'tmp.dict'))
                #dictionary.save(os.path.join(dst, 'tmp.dict'))

                #trained_corpus = itertools.chain(corpus, train_to_dict[train_corpus])

                # Overwrite model corpus
                #corpora.MmCorpus.serialize(os.path.join(dst, 'tmp.mm'), trained_corpus)

                # Load pre-trained LSI models
                tfidf = models.TfidfModel.load(os.path.join(dst, 'tmp.tfidf'))
                lsi = models.LsiModel.load(os.path.join(dst, 'tmp.lsi'))

                # Load indices
                lsi_index = similarities.Similarity.load(os.path.join(dst, 'lsi.index'))
                tfidf_index = similarities.Similarity.load(os.path.join(dst, 'tfidf.index'))

                # Update indices with training corpus
                lsi_index.add_documents(lsi[tfidf[list(train_corpus)]])
                tfidf_index.add_documents(tfidf[list(train_corpus)])

                # Save updated indices
                lsi_index.save(os.path.join(dst, 'lsi.index'))
                tfidf_index.save(os.path.join(dst, 'tfidf.index'))

                # Update labels and save to file
                cnt = merge_labels(dst)
                w.textOutput_train.setText('{} documents added!'.format(cnt))
                w.save_model_button.setDisabled(True)
            else:
                pass

        else:
            w.textOutput_train.setText('Please select a valid folder')

    @staticmethod
    def save_model():

        dst = w.input_line_edit_model.text()

        files = os.listdir(dst)
        if len(files) > 0:
            basename = os.path.join(dst)
            root = os.path.dirname(dst)
            basedir = os.path.basename(dst)

            # Create ZIP archive in root directory
            shutil.make_archive(basename, 'zip', root, basedir)
            w.textOutput_train.append('Backup of {} folder created here: {}'.format(basedir, root))

            # Remove files in basedir
            for file in files:
                os.unlink(os.path.join(dst, file))

        else:
            pass

        src = TEMP_FOLDER
        files = os.listdir(src)

        cnt = 0
        for file in files:
            fp = os.path.join(src, file)
            fp_new = os.path.join(dst, file)

            if file.endswith('lsi.index') or file.endswith('tfidf.index'):
                index = similarities.Similarity.load(fp)
                # Set internal path to index
                index.output_prefix = fp_new
                # Update path to index shards so that it corresponds to the output_prefix
                index.check_moved()
                # Save index with updated paths to file
                index.save(fp_new)

            else:
                # Copy file to model parameters dir
                shutil.copy2(fp, fp_new)

            # Increment file count
            cnt += 1
            logging.info('Copying {} to {}'.format(file, dst))
            # Remove file in temporary folder
            os.unlink(fp)

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

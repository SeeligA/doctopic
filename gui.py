import sys

from qtpy import QtCore, uic
from qtpy.QtWidgets import qApp, QApplication, QFileDialog, QMainWindow, QMessageBox

import os.path

from doctopic import MyCorpus, load_from_folder, file_to_query, build_index, merge_labels, iter_documents, TEMP_FOLDER
from gensim import corpora, models, similarities
import logging

import shutil


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
        self.input_line_edit_query.setText(save_doc)
        # Write new user input to settings
        self.input_line_edit_model.textChanged.connect(self.new_folder_changed)
        self.input_line_edit_query.textChanged.connect(self.new_doc_changed)

    def new_folder_changed(self, newFolder):
        """Store dirpath to settings"""
        self.settings.setValue("savedFolder", w.input_line_edit_model.text())

    def initUI(self):

        uic.loadUi(os.path.join('GUI', 'doctopic.ui'), self)
        self.setWindowTitle('DocTopic')
        self.setContentsMargins(7, 0, 7, 0)
        self.statusBar().showMessage('Ready')

        self.show()

    def new_doc_changed(self, newDoc):
        """Store filepath to settings"""
        self.settings.setValue("savedDoc", w.input_line_edit_query.text())

    def open_model(self):
        """Open file dialog and write dirpath to line edit."""
        dirpath = QFileDialog.getExistingDirectory()
        w.input_line_edit_model.setText(dirpath)

    def open_folder(self):
        """Open file dialog and write dirpath to line edit."""
        dirpath = QFileDialog.getExistingDirectory()
        w.input_line_edit_train.setText(dirpath)

    def open_doc(self):
        """Open file dialog and write filepath to line edit."""
        fp = QFileDialog.getOpenFileName(filter='TXT-Datei (*.txt)')[0]
        w.input_line_edit_query.setText(fp)

    def load_model(self):
        # Check if a model has been already loaded
        if hasattr(w, 'lsi'):
            pass
        # Load relevant files from model directory
        else:
            logging.info('Loading model parameters')
            w.lsi, w.dictionary, w.tfidf, w.tfidf_index, w.lsi_index, w.labels, w.corpus = load_from_folder(
                w.input_line_edit_model.text())
            # Abort query if one or more files are missing
            if all((w.lsi, w.dictionary, w.tfidf, w.tfidf_index, w.lsi_index, w.labels, w.corpus)) == False:
                w.textOutput.setText('UnboundLocalError: Unable to find local variable')
                return None

    def run_query(self):
        """Run query against model."""

        # Check if a query document has been specified
        if os.path.isfile(w.input_line_edit_query.text()):
            w.load_model()
            # Convert file to BOW vector
            vec_bow = file_to_query(w.input_line_edit_query.text(), w.dictionary)
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
            w.textOutput.setText('Please select a valid file')

    def print_details(self, sims, sims_tfidf, topic):
        """Print query results to text output."""

        sections = [' LSI Similarity ', ' Most Prominent LSI topic ', ' tf-idf Similarity ']
        headers = '{:<5s}{:>8s}{:>20s}{:>20s}{:>27s}'.format('RANK', 'SIM', 'CLIENT', 'PROJECT', 'FILE')
        line = 120
        dashed = '{:-{align}{width}}'.format('', align='^', width=line)

        self.textOutput.setText('{:-{align}{width}}'.format(sections[0], align='^', width=line))
        self.textOutput.append(headers)
        self.textOutput.append(dashed)

        for i in range(min(len(sims), 10)):
            labels = w.labels[str(sims[i][0])]
            self.textOutput.append('{:<10d}{: 8.3f}\t{:>14.12}{:>14.12}\t{:<12s}'.format(i + 1, sims[i][1], labels[0], labels[1], labels[-1]))

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
                '{:<10d}{: 8.3f}\t{:>14.12}{:>14.12}\t{:<12s}'.format(i + 1, sims_tfidf[i][1], labels[0], labels[1], labels[-1]))

    def reset_model(self):
        """Delete attribute when changing the model folder."""

        if hasattr(w, 'lsi'):
            del w.lsi
            logging.info('Model reset')

    def train_model(self):
        src = w.input_line_edit_train.text()
        dst = [os.path.join(TEMP_FOLDER, name) for name in ['lsi.index', 'tfidf.index']]

        if os.path.isdir(src):
            corpus = MyCorpus(src)
            corpus.save_to_temp()
            num_features = len(corpus.dictionary)

            tfidf, corpus_tfidf, lsi, corpus_lsi = corpus.build_lsi(topics=250)

            # build LSI index and save to temp folder
            lsi_index = build_index(corpus_lsi, num_features, dst[0])
            lsi_index.save(dst[0])
            # build tfidf index and save to temp folder
            tfidf_index = build_index(corpus_tfidf, num_features, dst[1])
            tfidf_index.save(dst[1])

            w.textOutput_train.setText('Training complete! Click "Save Project" to save parameters to model folder')
            w.save_model_button.setEnabled(True)

        else:
            w.textOutput_train.setText('Please select a valid training folder')

        w.reset_model()

    def add_docs(self):

        dst = w.input_line_edit_model.text()
        src = w.input_line_edit_train.text()
        if os.path.isdir(dst) and os.path.isdir(src):

            dialog = QMessageBox.question(w, 'DocTopic', 'This operation will change your search indices. Continue?',
                                          QMessageBox.Save | QMessageBox.Cancel, QMessageBox.Cancel)
            if dialog == QMessageBox.Save:

                # Load dictionary and create corpus
                dictionary = corpora.Dictionary.load(os.path.join(dst, 'tmp.dict'))

                train_corpus = []
                for tokens in iter_documents(src):
                    train_corpus.append(dictionary.doc2bow(tokens))
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
            else:
                pass

        else:
            w.textOutput_train.setText('Please select a valid folder')

    @staticmethod
    def save_model(self):

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
                index.output_prefix = fp_new
                index.check_moved()
                index.save(fp_new)
                cnt += 1

            else:
                shutil.copy2(fp, fp_new)
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

    sys.exit(app.exec_())

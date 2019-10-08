import os
import os.path

from qtpy import QtCore
import ipywidgets as widgets
from ipywidgets import Button, Layout, GridspecLayout, GridBox

from sources.doctopic import (MyCorpus, load_from_folder, data_to_query, build_index, update_indices, TEMP_FOLDER)

from sources.utils import make_model_archive, move_from_temp
from sources.log import out, logger


class MyControlWidget(GridBox):
    def __init__(self):
        super(MyControlWidget, self).__init__()

        self.layouts = dict()

        self.init_ui()
        # Load settings
        self.settings = QtCore.QSettings("DocFirm Inc.", "docfind")
        # Retrieve user input from settings
        save_folder = self.settings.value("savedFolder", "")
        self.input_line_edit_model.value = save_folder
        self.input_line_edit_model.observe(self.new_folder_changed)
        self.input_widget = GridBox()

    def init_ui(self):

        self.layouts = self.create_layouts()
        self.build_widgets()
        self.build_tab_grids()
        self.apply_layouts()

        self.children = [self.input_line_edit_model, self.tabs, out]

        self.add_documents_button.on_click(self.add_docs)
        # w.open_model_button.clicked.connect(MyWindow.open_model)
        self.open_doc_button.observe(self.open_doc)
        self.train_folder_button.observe(self.check_train_folder, names='value')
        self.run_query_button.on_click(self.run_query)
        # w.actionOpen_Model.triggered.connect(MyWindow.open_model)
        # w.actionExit.triggered.connect(qApp.quit)

        self.train_model_button.on_click(self.train_model)
        self.save_model_button.on_click(self.save_model)
        # w.load_tree_button.clicked.connect(MyWindow.load_tree)
        # w.compare_checked_button.clicked.connect(MyWindow.run_index_query)

    def save_model(self, b):
        """Saving files created during training to model folder."""

        dst = self.input_line_edit_model.value

        # Check for files in model dir and zip folder if any
        if os.listdir(dst):
            archive = make_model_archive(dst)
            with self.text_output_train:
                print(f'Backup of model folder created here: {archive}')
        else:
            pass

        # Copy model files to model folder
        cnt = move_from_temp(TEMP_FOLDER, dst)
        with self.text_output_train:
            print(f'{cnt} files saved')
        self.save_model_button.disabled = True

    def train_model(self, b):
        src = self.input_line_edit_train.value
        dst = [os.path.join(TEMP_FOLDER, name) for name in ['lsi.index', 'tfidf.index']]

        self.text_output_train.clear_output()

        if os.path.isdir(src):

            corpus = MyCorpus(src)
            corpus.save_to_temp()
            num_features = len(corpus.dictionary)

            tfidf, corpus_tfidf, lsi, corpus_lsi = corpus.build_lsi(topics=300)

            # build LSI index and save to temp folder
            lsi_index = build_index(dst[0], corpus_lsi, num_features)
            lsi_index.save(dst[0])
            # build tfidf index and save to temp folder
            tfidf_index = build_index(dst[1], corpus_tfidf, num_features)
            tfidf_index.save(dst[1])

            with self.text_output_train:
                print('Training complete! Click "Save Project" to save parameters to model folder.')
            self.save_model_button.disabled = False

        else:
            with self.text_output_train:
                print('Please select a valid training folder!')

        self.reset_model()

    def reset_model(self):
        """Delete attribute when changing the model folder."""

        if hasattr(self, 'lsi'):
            del self.lsi
            logger.info('Model reset')

    def run_query(self, b):
        self.text_output_query.value = ''

        self.load_model()
        #  Todo: Implement batch querying for improved performance
        for idx in self.query_list_widget.index:
            file = self.query_list_widget.options[idx]

            self.text_output_query.value += f'Printing results for {file}:\n'

            data = self.open_doc_button.data[idx]
            vec_bow = data_to_query(data, self.dictionary)
            vec_bow = self.tfidf[vec_bow]
            # Serialize tfidf transformation and convert search vector to LSI space
            # Note: When using transformed search vectors, apply same transformation when building the index
            vec_lsi = self.lsi[vec_bow]

            # Apply search vector to indexed LSI corpus and sort resulting index-similarity tuples.
            sims_lsi = self.lsi_index[vec_lsi]
            sims_lsi = sorted(enumerate(sims_lsi), key=lambda item: -item[1])
            # Retrieve most prominent topic from search vector
            topic = self.lsi.print_topic(max(vec_lsi, key=lambda item: abs(item[1]))[0])

            # Apply search vector to transformed tfidf corpus and sort resulting index-similarity tuples
            sims_tfidf = self.tfidf_index[vec_bow]
            sims_tfidf = sorted(enumerate(sims_tfidf), key=lambda item: -item[1])

            def print_details(self, sims_lsi, sims_tfidf, topic):
                """Print query results to text output."""
                sections = [' LSI Similarity ', ' Most Prominent LSI topic ', ' tf-idf Similarity ']
                headers = '{:<5s}{:>6s}{:>14s}{:>18s}{:>20s}'.format('RANK', 'SIM', 'CLIENT', 'PROJECT', 'FILE')
                line = 120
                dashed = '{:-{align}{width}}'.format('', align='^', width=line)

                self.text_output_query.value += '{:-{align}{width}}\n'.format(sections[0], align='^', width=line)
                self.text_output_query.value += f'{headers}\n{dashed}\n'

                for i in range(min(len(sims_lsi), 10)):
                    labels = self.labels[str(sims_lsi[i][0])]
                    self.text_output_query.value += '\n{:<10d}{: 8.3f}\t{:>14.12}{:>14.12}\t{:<12s}'.format(
                        i + 1, sims_lsi[i][1], labels[0], labels[1], labels[-1])

                self.text_output_query.value += '\n\n{:-{align}{width}}\n'.format(sections[1], align='^', width=line)
                self.text_output_query.value += topic

                self.text_output_query.value += '\n{:-{align}{width}}\n'.format(sections[2], align='^', width=line)
                self.text_output_query.value += f'{headers}\n{dashed}\n'

                for i in range(min(len(sims_tfidf), 10)):
                    labels = self.labels[str(sims_tfidf[i][0])]
                    self.text_output_query.value += '\n{:<10d}{: 8.3f}\t{:>14.12}{:>14.12}\t{:<12s}'.format(
                        i + 1, sims_tfidf[i][1], labels[0], labels[1], labels[-1])

                self.text_output_query.value += '\n\n'

            print_details(self, sims_lsi, sims_tfidf, topic)

    def load_model(self):
        """Load model parameters.

        This method controls which model files will be loaded from disk and added to the
        current object as attributes.
        """

        # Check if a model has been already loaded
        if hasattr(self, 'lsi'):
            pass
        # Load relevant files from model directory
        else:
            logger.info('Loading model parameters')
            params = ['lsi', 'dictionary', 'tfidf', 'tfidf_index', 'lsi_index', 'labels', 'corpus']
            load = load_from_folder(params, self.input_line_edit_model.value)

            for k, v in load.items():
                setattr(self, k, v)
            logger.info('Model loaded')

    def apply_layouts(self):
        self.layout = self.layouts['main']
        self.button_controls.layout = self.layouts['controls']
        self.train_folder_button.layout = Layout(width='auto')

    @staticmethod
    def create_layouts():
        layouts = dict()
        layouts['main'] = {'grid_template_columns': 'repeat(1, 99.5%)'}
        layouts['controls'] = {'grid_template_columns': 'repeat(1, 99.5%)', 'height': '100px'}
        layouts['outputs'] = {'border': '1px solid grey',
                              'height': '99.5%', 'width': '98.5%',
                              'overflow': 'auto'}
        layouts['grids'] = {'n_columns': 6, 'n_rows': 8, 'height': '500px'}
        return layouts

    def build_widgets(self):

        self.input_line_edit_model = self.file_selector(placeholder="Enter path to Model folder", desc="Model: ")

        # Query tab widgets
        self.query_list_widget = widgets.SelectMultiple(
            options=[], rows=2, description="Query File(s)",
            layout=Layout(width='98.5%', height='95.5%'))

        self.open_doc_button = widgets.FileUpload(accept='.txt', multiple=True)

        self.run_query_button = self.create_expanded_button('Run Query', 'warning', disabled=True)

        self.input_line_edit_train = self.file_selector(placeholder="Enter path to training folder",
                                                        desc="Training Data")

        self.train_folder_button = widgets.ToggleButton(
            value=False,
            description='Check path',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Check path for .txt training files',
            icon='question'
        )

        self.train_model_button = self.create_expanded_button('Train Model', 'warning')
        self.save_model_button = self.create_expanded_button('Save Project', 'warning', disabled=True)
        self.add_documents_button = self.create_expanded_button('Add Documents', 'warning')
        self.button_controls = widgets.GridBox([self.train_model_button, self.save_model_button,
                                                self.add_documents_button])

        self.load_tree_button = self.create_expanded_button('Load Tree', 'warning')
        self.compare_checked_button = self.create_expanded_button('Compare checked files', 'warning')

        self.text_output_query = widgets.Textarea(layout=self.layouts['outputs'])
        self.tree_text_area = widgets.Textarea(layout=self.layouts['outputs'])
        self.text_output_index = widgets.Textarea(layout=self.layouts['outputs'])
        self.text_output_train = widgets.Output(layout=self.layouts['outputs'])

    def check_train_folder(self, b):

        directory = self.input_line_edit_train.value

        self.text_output_train.clear_output()

        if os.path.isdir(directory):

            with self.text_output_train:
                print('Checking...', end='')
            count = 0
            for root, dirs, files in os.walk(directory):
                for file in filter(lambda file: file.endswith('.txt'), files):
                    count += 1

            self.train_folder_button.description = f'{count} file(s) found'
            self.train_folder_button.icon = ''
            if count > 0:
                self.train_folder_button.button_style = 'success'
                with self.text_output_train:
                    print(' Done!')
        else:
            with self.text_output_train:
                print('Please select a valid training folder!')

    def open_doc(self, b):
        """Write filepath(s) to list widget."""

        query_items = [x['name'] for x in self.open_doc_button.metadata]
        self.open_doc_button._counter = len(query_items)
        self.query_list_widget.options = query_items
        self.query_list_widget.value = query_items
        self.run_query_button.disabled = False

    def new_folder_changed(self, newFolder):
        """Store dirpath to settings"""
        self.settings.setValue("savedFolder", self.input_line_edit_model.value)
        self.reset_model()

    def build_tab_grids(self):
        """Build grids for app tabs."""
        layout = self.layouts['grids']
        self.query_grid = self.build_query_grid(layout)
        self.train_grid = self.build_train_grid(layout)
        self.index_grid = self.build_index_grid(layout)

        self.tabs = widgets.Tab()

        self.tabs.children = [self.query_grid, self.train_grid, self.index_grid]
        self.tabs.set_title(0, 'Query')
        self.tabs.set_title(1, 'Train')
        self.tabs.set_title(2, 'Index')


    def build_index_grid(self, layout):
        grid = GridspecLayout(**layout)
        grid[0, :2] = self.load_tree_button
        grid[0, 2:] = self.compare_checked_button
        grid[1:, :2] = self.tree_text_area
        grid[1:, 2:5] = self.text_output_index
        return grid

    def build_train_grid(self, layout):
        grid = GridspecLayout(**layout)
        grid[0, :5] = self.input_line_edit_train
        grid[0, 5] = self.train_folder_button
        grid[1:, :5] = self.text_output_train
        grid[1:4, 5] = self.button_controls
        return grid

    def build_query_grid(self, layout):
        grid = GridspecLayout(**layout)
        grid[0:2, 0:5] = self.query_list_widget
        grid[0, 5] = self.open_doc_button
        grid[2:, 0:5] = self.text_output_query
        grid[2, 5] = self.run_query_button
        return grid

    def add_docs(self, b):

        dst = self.input_line_edit_model.value
        src = self.input_line_edit_train.value

        if os.path.isdir(dst) and os.path.isdir(src):

            # Prompt user before updating indices
            if self.add_documents_button.button_style == 'warning':
                with self.text_output_train:
                    print('This operation will change your search indices. Click Add Documents again to continue.')
                self.add_documents_button.button_style = 'danger'

            else:
                cnt = update_indices(src, dst)
                with self.text_output_train:
                    print(f'{cnt} documents added!')
                self.save_model_button.disabled = True
                self.reset_model()

        else:
            with self.text_output_train:
                print('Please select a valid folder')

    @staticmethod
    def file_selector(placeholder, desc, disabled=False):
        return widgets.Text(value=None,
                            placeholder=placeholder,
                            description=desc,
                            layout=Layout(width='98.5%', height='30px'),
                            disabled=disabled
                            )

    @staticmethod
    def create_expanded_button(description, button_style, disabled=False):
        return Button(description=description,
                      button_style=button_style,
                      disabled=disabled,
                      layout=Layout(height='auto', width='auto')
                      )

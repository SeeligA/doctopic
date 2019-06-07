from gensim import corpora, models, similarities, utils
from gensim.test.utils import get_tmpfile
import os
import tempfile
import json
# from smart_open import open
import logging
import zipfile
from contextlib import contextmanager


TEMP_FOLDER = tempfile.mkdtemp()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logging.info('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))


# See: https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ
#      --> Q1 How many times does a feature with id 123 appear in a corpus?


class MyCorpus(object):
    def __init__(self, top_dir):
        self.top_dir = top_dir
        self.dictionary = corpora.Dictionary(iter_documents(top_dir))
        self.dictionary.filter_extremes(no_below=1, keep_n=100000)  # check API docs for pruning params

    def __iter__(self):
        for tokens in iter_documents(self.top_dir):
            yield self.dictionary.doc2bow(tokens)

    def save_to_temp(self):

        self.dictionary.save(os.path.join(TEMP_FOLDER, 'tmp.dict'))
        corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'tmp.mm'), self)

    def build_tfidf(self):
        tfidf = models.TfidfModel(self)  # step 1 -- initialize a model
        tfidf.save(os.path.join(TEMP_FOLDER, 'tmp.tfidf'))
        return tfidf, tfidf[self]

    def build_lsi(self, topics=250):
        """
        Build LSI model from corpus and sparse word vector.

        Args:
            topics: integer value determining the number of dimensions of the LSI space

        Returns:
            tfidf: initialized tfidf model from self
            corpus_tfidf: transformation of tfidf model and self
            lsi: initialized lsi model from self
            corpus_lsi: transformation of lsi model and self
        """

        # When building indices from transformed vectors, apply same transformation to search vectors
        tfidf, corpus_tfidf = self.build_tfidf()
        lsi = models.LsiModel(corpus_tfidf, id2word=self.dictionary, num_topics=topics)
        corpus_lsi = lsi[corpus_tfidf]
        lsi.save(os.path.join(TEMP_FOLDER, 'tmp.lsi'))
        return tfidf, corpus_tfidf, lsi, corpus_lsi

    def build_lda(self, topics=10, passes=100):

        lda = models.LdaModel(corpus, id2word=self.dictionary, num_topics=topics, passes=passes)
        tfidf.save(os.path.join(TEMP_FOLDER, 'tmp.tfidf'))
        return lda[self]

    def get_labels(self):
        """
        Retrieve document ids with filename and directory for reference purposes.

        Returns:
            labels: dict() in JSON format with integer ids mapped to list entries
        """

        idx = 0
        labels = {}
        for root, dirs, files in os.walk(self.top_dir):
            client = os.path.split(os.path.split(root)[0])[1]
            project = os.path.split(root)[1]
            for file in filter(lambda file: file.endswith('.txt'), files):
                labels[str(idx)] = [client, project, file]
                idx += 1

            for file in filter(lambda file: file.endswith('.zip'), files):
                with zipfile.ZipFile(os.path.join(root, file), 'r') as doczip:
                    for name in doczip.namelist():
                        labels[str(idx)] = [client, project, name]
                        idx += 1

        # save labels to file
        save_labels(labels, os.path.join(TEMP_FOLDER, 'tmp.json'))

        return labels


def build_index(corpus, num_features, fp):
    """Build indices for different models.

    Args:
        corpus: either a regular MyCorpus object or a transformation
        num_features: number of dimensions required for creating sparse vectors
        fp: path/to/file

    Returns:
        index: index of documents comprised in a corpus
    """

    return similarities.Similarity(fp, corpus, num_features)


def file_to_query(filepath, dictionary):
    """Generate BOW vector from document for querying purposes."""

    if os.path.exists(filepath):
        document = open(filepath, 'rb').read()
        tokens = tokenize(document)
        vec_bow = dictionary.doc2bow(tokens)
        return vec_bow


def iter_documents(top_directory):
    """Iterate over all documents, yielding a document (=list of utf8 tokens) at a time."""
    idx = 0
    labels = {}

    for root, dirs, files in os.walk(top_directory):
        client = os.path.split(os.path.split(root)[0])[1]
        project = os.path.split(root)[-1]
        logging.debug(client)
        logging.debug(project)

        for file in filter(lambda file: file.endswith('.txt'), files):  # TODO: Add filter for PDF files

            try:
                with open(os.path.join(root, file), 'rb') as document:
                    logging.debug(file)

                    document = document.read()  # read the entire document, as one big string
                    labels[str(idx)] = [client, project, file]
                    idx += 1
                    yield tokenize(document)  # or whatever tokenization suits you
            except UnicodeDecodeError:
                idx -= 1
                continue

        for file in filter(lambda file: file.endswith('.zip'), files):
            with zipfile.ZipFile(os.path.join(root, file), 'r') as doczip:
                for name in doczip.namelist():
                    logging.debug(name)
                    if name.endswith('.txt'):
                        try:
                            with doczip.open(name) as document:
                                document = document.read()  # read the entire document, as one big string
                                labels[str(idx)] = [client, project, name]
                                logging.debug([idx, client, project, name])
                                idx += 1
                                yield tokenize(document)
                        except UnicodeDecodeError:
                            idx -= 1
                            continue

    save_labels(labels, os.path.join(TEMP_FOLDER, 'tmp.json'))


def load_from_folder(folder):
    lsi, dictionary, tfidf, tfidf_index, lsi_index, labels, corpus = None, None, None, None, None, None, None
    for file in os.listdir(folder):
        logging.info('Scanning: {}'.format(file))
        if file.endswith('.lsi'):
            lsi = models.LsiModel.load(os.path.join(folder, file))
        elif file.endswith('.tfidf'):
            tfidf = models.TfidfModel.load(os.path.join(folder, file))
        elif file.endswith('.dict'):
            dictionary = corpora.Dictionary.load(os.path.join(folder, file))
        elif file.endswith('tfidf.index'):
            tfidf_index = similarities.Similarity.load(os.path.join(folder, file))
        elif file.endswith('lsi.index'):
            lsi_index = similarities.Similarity.load(os.path.join(folder, file))
        elif file.endswith('.json'):
            labels = load_labels(os.path.join(folder, file))
        elif file.endswith('.mm'):
            corpus = corpora.MmCorpus(os.path.join(folder, file))

    return lsi, dictionary, tfidf, tfidf_index, lsi_index, labels, corpus


def load_labels(fp):
    with open(fp) as f:
        labels = json.load(f)
    return labels


def merge_labels(fp):
    """Merge model labels with training doc labels

    Args:
        fp: path to model parameters
    Returns:
        None
    """
    fp = os.path.join(fp, 'tmp.json')

    labels = load_labels(fp)
    # Models labels need to be indexed consecutively
    idx = len(labels)

    new_labels = load_labels(os.path.join(TEMP_FOLDER, 'tmp.json'))
    for k, v in new_labels.items():
        labels[str(idx)] = v
        idx += 1
    save_labels(labels, fp)
    return len(new_labels)


def save_labels(labels, fp):
    """Store document ids with grandparent & parent directory name and filename for reference purposes."""

    with open(fp, 'w') as f:
        json.dump(labels, f, sort_keys=True)


def tokenize(document):
    # TODO: Replace with custom tokenizer and customizable stop word list
    return utils.tokenize(document, lower=True)

import json
import logging
import os
import random
import tempfile
import zipfile

from gensim import corpora, models, similarities, utils
from smart_open import open

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

        # When building indices from transformed vectors, apply the same transformation to search vectors
        tfidf, corpus_tfidf = self.build_tfidf()
        lsi = models.LsiModel(corpus_tfidf, id2word=self.dictionary, num_topics=topics)
        corpus_lsi = lsi[corpus_tfidf]
        lsi.save(os.path.join(TEMP_FOLDER, 'tmp.lsi'))
        return tfidf, corpus_tfidf, lsi, corpus_lsi

    # def build_lda(self, topics=10, passes=100):
    #
    #     lda = models.LdaModel(corpus, id2word=self.dictionary, num_topics=topics, passes=passes)
    #     tfidf.save(os.path.join(TEMP_FOLDER, 'tmp.tfidf'))
    #     return lda[self]


def build_index(fp, corpus, num_features):
    """Build indices for different models.

    Args:
        fp: path/to/file
        corpus: either a regular MyCorpus object or a transformation
        num_features: number of dimensions required for creating sparse vectors

    Returns:
        index of documents comprised in a corpus
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

        # Shuffle so that client dirs are picked at random and project dirs within a client dir are processed in a
        # random sequence
        random.seed(2)
        dirs = random.shuffle(dirs)

        client = os.path.split(os.path.split(root)[0])[1]
        project = os.path.split(root)[-1]
        logging.debug(client)
        logging.debug(project)

        for file in filter(lambda file: file.endswith('.txt'), files):  # TODO: Add filter for PDF files
            logging.info(os.path.join(root, file))
            try:
                with open(os.path.join(root, file), 'rb') as document:
                    logging.debug(file)

                    document = document.read()  # read the entire document, as one big string
                    labels[str(idx)] = [client, project, file]
                    logging.debug([idx, client, project, file])
                    idx += 1
                    yield tokenize(document)  # or whatever tokenization suits you
            except UnicodeDecodeError:
                idx -= 1
                continue

        for file in filter(lambda file: file.endswith('.zip'), files):
            with zipfile.ZipFile(os.path.join(root, file), 'r') as doczip:
                for name in doczip.namelist():

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


def load_from_folder(params, arg):
    """Dynamically load parameters from model folder.

    Arguments:
        params -- List of required model parameters
        arg -- path/to/model/parameters
    Returns:
        load -- Dictionary of loaded parameters
    """

    load = dict()

    dispatch = {
         ('dictionary', 'tmp.dict'): corpora.Dictionary.load,
         ('tfidf', 'tmp.tfidf'): models.TfidfModel.load,
         ('lsi', 'tmp.lsi'): models.LsiModel.load,
         ('tfidf_index', 'tfidf.index'): similarities.Similarity.load,
         ('lsi_index', 'lsi.index'): similarities.Similarity.load,
         ('labels', 'tmp.json'): load_labels,
         ('corpus', 'tmp.mm'): corpora.MmCorpus
    }
    # Use dispatch mapping to call load functions for parameters
    for k, v in dispatch.items():
        if k[0] in params:
            load[k[0]] = dispatch[k](os.path.join(arg, k[1]))

    return load


def load_labels(fp):
    with open(fp) as f:
        labels = json.load(f)
    return labels


def merge_labels(fp):
    """Merge model labels with training doc labels.

    Args:
        fp: path to model parameters
    Returns:
        New labels count for information purposes
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


def update_indices(src, dst):
    """Update search indices with additional training documents.
    This feature is best suited for small updates and models that are already quite balanced.
    LSI and TFIDF weights will not be updated and the training files will not be included
    in neither the corpus or the dictionary.

    Arguments:
        src -- path/to/new/docs
        dst -- path/to/model
    Returns:
        cnt -- New labels count for information purposes
    """

    # Load model parameters
    params = ['lsi', 'dictionary', 'tfidf', 'tfidf_index', 'lsi_index']
    load = load_from_folder(params, dst)
    lsi, dictionary, tfidf, tfidf_index, lsi_index = map(lambda x: load[x], params)

    # Create list of BOW from training documents
    train_corpus = [dictionary.doc2bow(tokens) for tokens in iter_documents(src)]

    corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'tmp.mm'), train_corpus)
    train_corpus = corpora.MmCorpus(os.path.join(TEMP_FOLDER, 'tmp.mm'))

    # Update indices with training corpus
    lsi_index.add_documents(lsi[tfidf[list(train_corpus)]])
    tfidf_index.add_documents(tfidf[list(train_corpus)])

    # Save updated indices
    lsi_index.save(os.path.join(dst, 'lsi.index'))
    tfidf_index.save(os.path.join(dst, 'tfidf.index'))

    # Update labels and save to file
    cnt = merge_labels(dst)
    return cnt


def update_index_path(fp, fp_new):
    index = similarities.Similarity.load(fp)
    # Set internal path to index
    index.output_prefix = fp_new
    # Update path to index shards so that it corresponds to the output_prefix
    index.check_moved()
    # Save index with updated paths to file
    index.save(fp_new)

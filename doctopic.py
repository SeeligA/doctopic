from gensim import corpora, models, similarities, utils
import os
import tempfile
import json
from smart_open import open
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TEMP_FOLDER = tempfile.gettempdir()
logging.info('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))

#See: https://github.com/RaRe-Technologies/gensim/wiki/Recipes-&-FAQ#q1-how-many-times-does-a-feature-with-id-123-appear-in-a-corpus


class MyCorpus(object):
    def __init__(self, top_dir):
        self.top_dir = top_dir
        self.dictionary = corpora.Dictionary(iter_documents(top_dir))
        self.dictionary.filter_extremes(no_below=1, keep_n=30000) # check API docs for pruning params

    def __iter__(self):
        for tokens in iter_documents(self.top_dir):
            yield self.dictionary.doc2bow(tokens)

    def save_to_temp(self):

        self.dictionary.save(os.path.join(TEMP_FOLDER, 'tmp.dict'))
        corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'tmp.mm'), self)

    def build_tfidf(self):

        tfidf = models.TfidfModel(self) # step 1 -- initialize a model
        tfidf.save(os.path.join(TEMP_FOLDER, 'tmp.tfidf'))
        return tfidf, tfidf[self]

    def build_lsi(self, topics=250):

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
        '''
        Retrieve document ids with filename and directory for reference purposes
        Returns:
        labels -- dict() in JSON format with integer ids mapped to list entries
        '''

        idx = 0
        labels = {}
        for root, dirs, files in os.walk(self.top_dir):
            for file in filter(lambda file: file.endswith('.txt'), files):
                labels[str(idx)] = [os.path.split(root)[1], file]
                idx+=1
        # save labels to file
        save_labels(labels, os.path.join(TEMP_FOLDER, 'tmp.json'))

        return labels

    def train_lsi(self, filepath):

        _, tfidf_train = self.build_tfidf()
        logging.info('new tfidf corpus built')
        lsi = models.LsiModel.load(filepath)
        logging.info('existing LSI model loaded')
        lsi.add_documents(tfidf_train) # TODO: check if there are limitations as to how many documents can be added at a time
        logging.info('additional docs added')
        trained_lsi = lsi[tfidf_train]
        logging.info('training corpus fitted to model')
        lsi.save(os.path.join(TEMP_FOLDER, 'train.lsi'))
        return lsi, trained_lsi



def build_index(corpus):
    index = similarities.MatrixSimilarity(corpus) # TODO: include option to specify num_best
    index.save(os.path.join(TEMP_FOLDER, 'tmp.index'))
    return index

def file_to_query(filepath, dictionary):
    '''Generate BOW vector from document for querying
    Arguments:
    filepath -- path to document
    dictionary -- mapping of (normalized) words to integer ids
    '''
    if (os.path.exists(filepath)):
        document = open(filepath, 'rb').read()
        tokens = tokenize(document)
        vec_bow = dictionary.doc2bow(tokens)
        return vec_bow

def iter_documents(top_directory):
    '''Iterate over all documents, yielding a document (=list of utf8 tokens) at a time.'''
    for root, dirs, files in os.walk(top_directory):
        for file in filter(lambda file: file.endswith('.txt'), files): # TODO: Add filter for PDF files
            with open(os.path.join(root, file), 'rb') as document:
                document = document.read() # read the entire document, as one big string
                logging.info(file)
                yield tokenize(document) # or whatever tokenization suits you


def load_from_folder(folder):

    model, dictionary, index, labels = None, None, None, None
    for file in os.listdir(folder):
        logging.info('Scanning: {}'.format(file))

        if file.endswith('.lsi'):
            model = models.LsiModel.load(os.path.join(folder, file))
        elif file.endswith('.dict'):
            dictionary = corpora.Dictionary.load(os.path.join(folder, file))
        elif file.endswith('.index'):
            index = similarities.MatrixSimilarity.load(os.path.join(folder, file))
        elif file.endswith('.json'):
            labels = load_labels(os.path.join(folder, file))

    return model, dictionary, index, labels



def load_labels(fp):

    with open(fp) as f:
        labels = json.load(f)
    return labels


def save_labels(labels, fp):
    '''Store document ids with filename and directory for reference purposes'''
    with open(fp, 'w') as f:
        json.dump(labels, f, sort_keys=True)

def tokenize(document):
    # TODO: Replace with custom tokenizer and customizable stopword list
    return utils.tokenize(document, lower=True)

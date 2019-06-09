# DocTopic

## What is DocTopic?
A topic modelling and similarity retrieval interface that helps you managing your documents. DocTopic uses [Gensim](https://github.com/RaRe-Technologies/gensim/ "Gensim on GitHub"), a popular Python library designed for implementing key NLP algorithms at scale.
* Some excellent tutorials can be found on their [website](https://radimrehurek.com/gensim/tutorial.html "Gensim tutorials"). They also offer support and professional services.
* An interactive introduction to similarity search can be found [here](https://www.oreilly.com/learning/how-do-i-compare-document-similarity-using-python "Document Similarity using Python").

## Features
* Create a searchable corpus from your multilingual documents with 2 clicks.
* Use unsupervised training algorithms such as _Latent Semantic Analysis_ and _Latent Dirichlet Allocation_ for topic modelling purposes.  
* Query your corpus to retrieve documents that are structurally similar or belong to a similar domain.
* Update your search indices with new files so that they can be retrieved later.

## Use cases for translation service providers
Identify relevant resources from __historical project data__ such as:
* previous translations to be used as templates
* translation vendors who are experts in their field
* project parameters such as turn-around times, pre-processing steps, etc.

Quickly assess the similarity of __files within a project__ to help with:
* staggered/cascading deliveries
* assigning files to multiple vendors

__Classify documents automatically__ and create topic clusters to better understand:
* the translation needs of your customer segments
* your level of specialization and how you can use it to build your brand

## Installation
DocTopic has been created with Python 3.7. It requires Gensim in addition to Numpy, Scipy and PyQt5/qtpy. You will probably want to us a virtual environment like conda. The [Anaconda](https://www.anaconda.com/distribution/) distribution comes with the latter packages already installed. Then:

    pip install -U gensim

## Questions
If you found any of the content from this repo helpful, confusing or missing, I would like to hear from you.

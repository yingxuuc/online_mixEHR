"""
pyLDAvis mixEHR  
===============
Helper functions to visualize LDA models trained by ldaEHR. 
Ref: https://github.com/bmabey/pyLDAvis/blob/master/pyLDAvis/gensim_models.py
"""

import funcy as fp
import numpy as np
from scipy.sparse import issparse
import pyLDAvis._prepare


def _extract_data(topic_model, corpus, dictionary, doc_topic_dists=None):
    import gensim

    num_topics = topic_model.num_topics
    topic = topic_model.state.get_lambda()

    if doc_topic_dists is None:
        gamma, _ = topic_model.inference(corpus)
        doc_topic_dists = gamma / gamma.sum(axis=1)[:, None]
    else:
        if isinstance(doc_topic_dists, list):
            doc_topic_dists = gensim.matutils.corpus2dense(doc_topic_dists, num_topics).T
        elif issparse(doc_topic_dists):
            doc_topic_dists = doc_topic_dists.T.todense()
        doc_topic_dists = doc_topic_dists / doc_topic_dists.sum(axis=1)
    
    assert doc_topic_dists.shape[1] == num_topics,\
        'Document topics and number of topics do not match {} != {}'.format(
        doc_topic_dists.shape[1], num_topics)

    tf_all = []
    doc_lst = []
    vocab = []
    topic_term_dists = []
    for i in range(len(corpus)):
        if not gensim.matutils.ismatrix(corpus[i]):
            corpus_csc = gensim.matutils.corpus2csc(corpus[i], num_terms=len(dictionary[i]))
        else:
            corpus_csc = corpus[i]
 
        vocab.extend(list(dictionary[i].token2id.keys()))
        beta = 0.01
        fnames_argsort = np.asarray(list(dictionary[i].token2id.values()), dtype=np.int_)
        term_freqs = corpus_csc.sum(axis=1).A.ravel()[fnames_argsort]
        term_freqs[term_freqs == 0] = beta
        doc_lengths = corpus_csc.sum(axis=0).A.ravel()

        assert term_freqs.shape[0] == len(dictionary[i]),\
            'Term frequencies and dictionary have different shape {} != {}'.format(
            term_freqs.shape[0], len(dictionary[i]))
        assert doc_lengths.shape[0] == len(corpus[i]),\
            'Document lengths and corpus have different sizes {} != {}'.format(
            doc_lengths.shape[0], len(corpus[i]))

        tf_lst.append(term_freqs)
        doc_lst.append(doc_lengths)

        topic_i = topic[i] / topic[i].sum(axis=1)[:, None]
        topic_term_dists_i = topic_i[:, fnames_argsort]
        topic_term_dists.append(topic_term_dists_i)

    doc_lengths = sum(doc_lst)
    topic_term_dists = np.concatenate(topic_term_dists, axis = 1)

    return {'topic_term_dists': topic_term_dists, 'doc_topic_dists': doc_topic_dists,
            'doc_lengths': doc_lengths, 'vocab': vocab, 'term_frequency': np.concatenate(tf_all, axis = None)}


def prepare(topic_model, corpus, dictionary, doc_topic_dist=None, **kwargs):
    """Transforms the LdaEHR model and related corpus and dictionary into
    the data structures needed for the visualization.
    Parameters
    ----------
    topic_model : LdaEHR model
        An already trained LdaModel. 
    corpus : list of corpus, the same format as the input requirement of LdaEHR model.
    dictionary: list of gensim.corpora.Dictionary
    doc_topic_dist (optional): Document topic distribution from LdaEHR (default=None)
        The document topic distribution that is eventually visualised, if you will
        be calling `prepare` multiple times it's a good idea to explicitly pass in
        `doc_topic_dist` as inferring this for large corpora can be quite
        expensive. (most time cosuming part)
    **kwargs :
        additional keyword arguments are passed through to :func:`pyldavis.prepare`.
    Returns
    -------
    prepared_data : PreparedData
        the data structures used in the visualization
    Example
    --------
    For example usage please see this notebook:
    http://nbviewer.ipython.org/github/bmabey/pyLDAvis/blob/master/notebooks/Gensim%20Newsgroup.ipynb
    See
    ------
    See `pyLDAvis.prepare` for **kwargs.
    """
    opts = fp.merge(_extract_data(topic_model, corpus, dictionary, doc_topic_dist), kwargs)
    return pyLDAvis.prepare(**opts)


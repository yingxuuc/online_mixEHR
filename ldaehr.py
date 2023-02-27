

""" The code is based on ldamodel from gensim package.
LdaEHR: extend the LDA model to multiple types of documents. In the
EHR world, different types can be diagnosis codes, medicines, labs, etc.
"""

import logging
import numbers
import os
import time
from collections import defaultdict

import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma

from gensim import interfaces, utils, matutils
from gensim.matutils import (
    kullback_leibler, hellinger, jaccard_distance, jensen_shannon,
    dirichlet_expectation, logsumexp, mean_absolute_difference,
)
from gensim.models import basemodel, CoherenceModel
from gensim.models.callbacks import Callback


logger = logging.getLogger(__name__)


def update_dir_prior(prior, N, logphat, rho):
    """Update a given prior using Newton's method, described in
    `J. Huang: "Maximum Likelihood Estimation of Dirichlet Distribution Parameters"
    <http://jonathan-huang.org/research/dirichlet/dirichlet.pdf>`_.

    Parameters
    ----------
    prior : list of float
        The prior for each possible outcome at the previous iteration (to be updated).
    N : int
        Number of observations.
    logphat : list of float
        Log probabilities for the current estimation, also called "observed sufficient statistics".
    rho : float
        Learning rate.

    Returns
    -------
    list of float
        The updated prior.

    """
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)

    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))

    dprior = -(gradf - b) / q

    updated_prior = rho * dprior + prior
    if all(updated_prior > 0):
        prior = updated_prior
    else:
        logger.warning("updated prior is not positive")
    return prior


class LdaEHRState(utils.SaveLoad):
    """Encapsulate information for distributed computation of 

    Objects of this class are sent over the network, so try to keep them lean to
    reduce traffic.

    """

    def __init__(self, eta, shape_lst, dtype=np.float32):
        """

        Parameters
        ----------
        eta : numpy.ndarray
            The prior probabilities assigned to each term.
        shape : list of tuples (int, int)
            Shape of the sufficient statistics: (number of topics to be found, number of terms in the vocabulary).
        dtype : type
            Overrides the numpy array default types.

        """
        self.eta = np.empty(len(eta), dtype=object)
        self.sstats = np.empty(len(shape_lst), dtype=object)
        for i in range(len(shape_lst)):
            self.sstats[i] = np.zeros(shape_lst[i], dtype=dtype)
            self.eta[i] = eta[i].astype(dtype, copy=False)
        self.numdocs = 0
        self.dtype = dtype

    def reset(self):
        """Prepare the state for a new EM iteration (reset sufficient stats)."""
        for x in self.sstats:
            x[:] = 0.0
        self.numdocs = 0

    def merge(self, other):
        """Merge the result of an E step from one node with that of another node (summing up sufficient statistics).

        The merging is trivial and after merging all cluster nodes, we have the
        exact same result as if the computation was run on a single node (no
        approximation).

        Parameters
        ----------
        other : :class:`~LdaEHRState`
            The state object with which the current one will be merged.

        """
        assert other is not None
        self.sstats += other.sstats
        self.numdocs += other.numdocs

    def blend(self, rhot, other, targetsize=None):
        """Merge the current state with another one using a weighted average for the sufficient statistics.

        The number of documents is stretched in both state objects, so that they are of comparable magnitude.
        This procedure corresponds to the stochastic gradient update from
        `'Online Learning for LDA' by Hoffman et al.`_, see equations (5) and (9).

        Parameters
        ----------
        rhot : float
            Weight of the `other` state in the computed average. A value of 0.0 means that `other`
            is completely ignored. A value of 1.0 means `self` is completely ignored.
        other : :class:`~LdaEHRState`
            The state object with which the current one will be merged.
        targetsize : int, optional
            The number of documents to stretch both states to.

        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # stretch the current model's expected n*phi counts to target size
        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats *= (1.0 - rhot) * scale

        # stretch the incoming n*phi counts to target size
        if other.numdocs == 0 or targetsize == other.numdocs:
            scale = 1.0
        else:
            logger.info(
                "merging changes from %i documents into a model of %i documents", other.numdocs, targetsize)
            scale = 1.0 * targetsize / other.numdocs
        self.sstats += rhot * scale * other.sstats

        self.numdocs = targetsize

    def blend2(self, rhot, other, targetsize=None):
        """Merge the current state with another one using a weighted sum for the sufficient statistics.

        In contrast to :meth:`~LdaEHRState.blend`, the sufficient statistics are not scaled
        prior to aggregation.

        Parameters
        ----------
        rhot : float
            Unused.
        other : :class:`~LdaEHRState`
            The state object with which the current one will be merged.
        targetsize : int, optional
            The number of documents to stretch both states to.

        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # merge the two matrices by summing
        self.sstats += other.sstats
        self.numdocs = targetsize

    def get_lambda(self):
        """Get the parameters of the posterior over the topics, also referred to as "the topics".

        Returns
        -------
        numpy.ndarray
            Parameters of the posterior probability over topics.

        """
        return self.eta + self.sstats

    def get_Elogbeta(self):
        """Get the log (posterior) probabilities for each topic.

        Returns
        -------
        numpy.ndarray
            Posterior probabilities for each topic.
        """
        lambda_lst = self.get_lambda()
        res = np.empty(len(lambda_lst), dtype=object)
        for t in range(len(res)):
            res[t] = dirichlet_expectation(lambda_lst[t])
        return res

    @classmethod
    def load(cls, fname, *args, **kwargs):
        """Load a previously stored state from disk.

        Overrides :class:`~gensim.utils.SaveLoad.load` by enforcing the `dtype` parameter
        to ensure backwards compatibility.

        Parameters
        ----------
        fname : str
            Path to file that contains the needed object.
        args : object
            Positional parameters to be propagated to class:`~gensim.utils.SaveLoad.load`
        kwargs : object
            Key-word parameters to be propagated to class:`~gensim.utils.SaveLoad.load`

        Returns
        -------
        :class:`~LdaEHRState`
            The state loaded from the given file.

        """
        result = super(LdaEHRState, cls).load(fname, *args, **kwargs)

        # dtype could be absent in old models
        if not hasattr(result, 'dtype'):
            # float64 was implicitly used before (because it's the default in numpy)
            result.dtype = np.float64
            logging.info("dtype was not set in saved %s file %s, assuming np.float64",
                         result.__class__.__name__, fname)

        return result


class LdaEHR(interfaces.TransformationABC, basemodel.BaseTopicModel):
    """Train and use Online Latent Dirichlet Allocation model as
    presented in `'Online Learning for LDA' by Hoffman et al.`_

    Extend the standard LDA model to multiple data types.
    Ref: Inferring multimodal latent topics from electronic health records
    https://www.nature.com/articles/s41467-020-16378-3
    """

    def __init__(self, corpus=None, num_topics=100, id2word=None,
                 distributed=False, chunksize=2000, passes=1, update_every=1,
                 alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
                 iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
                 random_state=None, ns_conf=None, minimum_phi_value=0.01,
                 per_word_topics=False, callbacks=None, dtype=np.float32):
        """

        Parameters
        ----------
        corpus : list of corpus, corpus definition refer to LdaModel.
        num_topics : int, optional
            The number of requested latent topics to be extracted from the training corpus.
        id2word : list of `gensim.corpora.dictionary.Dictionary`
        distributed : bool, optional
            Whether distributed computing should be used to accelerate training.
        chunksize :  int, optional
            Number of documents to be used in each training chunk.
        passes : int, optional
            Number of passes through the corpus during training.
        update_every : int, optional
            Number of documents to be iterated through for each update.
            Set to 0 for batch learning, > 1 for online iterative learning.
        alpha : {float, numpy.ndarray of float, list of float, str}, optional
            A-priori belief on document-topic distribution, this can be:
                * scalar for a symmetric prior over document-topic distribution,
                * 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.

            Alternatively default prior selecting strategies can be employed by supplying a string:
                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,
                * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / (topic_index + sqrt(num_topics))`,
                * 'auto': Learns an asymmetric prior from the corpus (not available if `distributed==True`).
        eta : {float, numpy.ndarray of float, list of float, str}, optional
            A-priori belief on topic-word distribution, this can be:
                * scalar for a symmetric prior over topic-word distribution,
                * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,
                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.

            Alternatively default prior selecting strategies can be employed by supplying a string:
                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,
                * 'auto': Learns an asymmetric prior from the corpus.
        decay : float, optional
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten
            when each new document is examined.
            Corresponds to :math:`\\kappa` from `'Online Learning for LDA' by Hoffman et al.`_
        offset : float, optional
            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
            Corresponds to :math:`\\tau_0` from `'Online Learning for LDA' by Hoffman et al.`_
        eval_every : int, optional
            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
        iterations : int, optional
            Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
        gamma_threshold : float, optional
            Minimum change in the value of the gamma parameters to continue iterating.
        minimum_probability : float, optional
            Topics with a probability lower than this threshold will be filtered out.
        random_state : {np.random.RandomState, int}, optional
            Either a randomState object or a seed to generate one. Useful for reproducibility.
        ns_conf : dict of (str, object), optional
            Key word parameters propagated to :func:`gensim.utils.getNS` to get a Pyro4 nameserver.
            Only used if `distributed` is set to True.
        minimum_phi_value : float, optional
            if `per_word_topics` is True, this represents a lower bound on the term probabilities.
        per_word_topics : bool
            If True, the model also computes a list of topics, sorted in descending order of most likely topics for
            each word, along with their phi values multiplied by the feature length (i.e. word count).
        callbacks : list of :class:`~gensim.models.callbacks.Callback`
            Metric callbacks to log and visualize evaluation metrics of the model during training.
        dtype : {numpy.float16, numpy.float32, numpy.float64}, optional
            Data-type to use during calculations inside model. All inputs are also converted.

        """
        self.dtype = np.finfo(dtype).dtype

        # store user-supplied parameters
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise ValueError(
                'at least one of corpus/id2word must be specified, to establish input space dimensionality'
            )

        if self.id2word is None:
            logger.warning(
                "no word id mapping provided; initializing from corpus, assuming identity")
            self.id2word = [utils.dict_from_corpus(x) for x in corpus]

        self.num_terms = [len(x) for x in self.id2word]

        if any([x == 0 for x in self.num_terms]):
            raise ValueError(
                "cannot compute LDA over an empty collection (no terms)")

        self.distributed = bool(distributed)
        self.num_topics = int(num_topics)
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.num_updates = 0

        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics
        self.callbacks = callbacks

        self.alpha, self.optimize_alpha = self.init_dir_prior(alpha, 'alpha')
        assert self.alpha.shape == (self.num_topics,), \
            "Invalid alpha shape. Got shape %s, but expected (%d, )" % (
                str(self.alpha.shape), self.num_topics)

        self.eta, self.optimize_eta = self.init_dir_prior(eta, 'eta')
        for i in range(len(self.eta)):
            assert self.eta[i].shape == (self.num_terms[i],) or self.eta[i].shape == (self.num_topics, self.num_terms[i]), (
                "Invalid eta shape. Got shape %s, but expected (%d, 1) or (%d, %d)" %
                (str(self.eta[i].shape), self.num_terms[i], self.num_topics, self.num_terms[i]))

        self.random_state = utils.get_random_state(random_state)

        # VB constants
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold

        # set up distributed environment if necessary
        if not distributed:
            logger.info("using serial LDA version on this node")
            self.dispatcher = None
            self.numworkers = 1
        else:
            if self.optimize_alpha:
                raise NotImplementedError(
                    "auto-optimizing alpha not implemented in distributed LDA")
            # set up distributed version
            try:
                import Pyro4
                if ns_conf is None:
                    ns_conf = {}

                with utils.getNS(**ns_conf) as ns:
                    from gensim.models.lda_dispatcher import LDA_DISPATCHER_PREFIX
                    self.dispatcher = Pyro4.Proxy(
                        ns.list(prefix=LDA_DISPATCHER_PREFIX)[LDA_DISPATCHER_PREFIX])
                    logger.debug("looking for dispatcher at %s" %
                                 str(self.dispatcher._pyroUri))
                    self.dispatcher.initialize(
                        id2word=self.id2word, num_topics=self.num_topics, chunksize=chunksize,
                        alpha=alpha, eta=eta, distributed=False
                    )
                    self.numworkers = len(self.dispatcher.getworkers())
                    logger.info(
                        "using distributed version with %i workers", self.numworkers)
            except Exception as err:
                logger.error("failed to initialize distributed LDA (%s)", err)
                raise RuntimeError(
                    "failed to initialize distributed LDA (%s)" % err)

        # Initialize the variational distribution q(beta|lambda)
        self.state = LdaEHRState(
            self.eta, [(self.num_topics, x) for x in self.num_terms], dtype=self.dtype)
        self.expElogbeta = np.empty(len(self.num_terms), dtype=object)
        for t in range(len(self.num_terms)):
            self.state.sstats[t] = self.random_state.gamma(
                100., 1. / 100., (self.num_topics, self.num_terms[t])).astype(dtype, copy=False)
            self.expElogbeta[t] = np.exp(
                dirichlet_expectation(self.state.sstats[t]))

        # Check that we haven't accidentally fallen back to np.float64
        assert self.eta[0].dtype == self.dtype
        assert self.expElogbeta[0].dtype == self.dtype

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            use_numpy = self.dispatcher is not None
            start = time.time()
            self.update(corpus, chunks_as_numpy=use_numpy)
            self.add_lifecycle_event(
                "created",
                msg=f"trained {self} in {time.time() - start:.2f}s",
            )

    def init_dir_prior(self, prior, name):
        """Initialize priors for the Dirichlet distribution.

        Parameters
        ----------
        prior : {float, numpy.ndarray of float, list of float, str}
            A-priori belief on document-topic distribution. If `name` == 'alpha', then the prior can be:
                * scalar for a symmetric prior over document-topic distribution,
                * 1D array of length equal to num_topics to denote an asymmetric user defined prior for each topic.

            Alternatively default prior selecting strategies can be employed by supplying a string:
                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,
                * 'asymmetric': Uses a fixed normalized asymmetric prior of `1.0 / (topic_index + sqrt(num_topics))`,
                * 'auto': Learns an asymmetric prior from the corpus (not available if `distributed==True`).

            A-priori belief on topic-word distribution. If `name` == 'eta' then the prior can be:
                * scalar for a symmetric prior over topic-word distribution,
                * 1D array of length equal to num_words to denote an asymmetric user defined prior for each word,
                * matrix of shape (num_topics, num_words) to assign a probability for each word-topic combination.

            Alternatively default prior selecting strategies can be employed by supplying a string:
                * 'symmetric': (default) Uses a fixed symmetric prior of `1.0 / num_topics`,
                * 'auto': Learns an asymmetric prior from the corpus.
        name : {'alpha', 'eta'}
            Whether the `prior` is parameterized by the alpha vector (1 parameter per topic)
            or by the eta (1 parameter per unique term in the vocabulary).

        Returns
        -------
        init_prior: numpy.ndarray
            Initialized Dirichlet prior:
            If 'alpha' was provided as `name` the shape is (self.num_topics, ).
            If 'eta' was provided as `name` the shape is (len(self.id2word), ).
        is_auto: bool
            Flag that shows if hyperparameter optimization should be used or not.
        """
        if prior is None:
            prior = 'symmetric'

        if name == 'alpha':
            prior_shape = self.num_topics
        elif name == 'eta':
            prior_shape = self.num_terms
        else:
            raise ValueError("'name' must be 'alpha' or 'eta'")

        is_auto = False

        if isinstance(prior, str):
            if prior == 'symmetric':
                logger.info("using symmetric %s at %s", name, 1.0 / self.num_topics)
                if name == 'alpha':
                    init_prior = np.fromiter(
                        (1.0 / self.num_topics for i in range(prior_shape)),
                        dtype=self.dtype, count=prior_shape,
                    )
                else:
                    init_prior = np.empty(len(prior_shape), dtype=object)
                    for t in range(len(prior_shape)):
                        x = prior_shape[t]
                        init_prior[t] = np.fromiter(
                            (1.0 / self.num_topics for i in range(x)), dtype=self.dtype, count=x)
            elif prior == 'asymmetric':
                if name == 'eta':
                    raise ValueError(
                        "The 'asymmetric' option cannot be used for eta")
                init_prior = np.fromiter(
                    (1.0 / (i + np.sqrt(prior_shape))
                     for i in range(prior_shape)),
                    dtype=self.dtype, count=prior_shape,
                )
                init_prior /= init_prior.sum()
                logger.info("using asymmetric %s %s", name, list(init_prior))
            elif prior == 'auto':
                is_auto = True
                if name == 'alpha':
                    init_prior = np.fromiter((1.0 / self.num_topics for i in range(prior_shape)),
                                             dtype=self.dtype, count=prior_shape)
                    logger.info("using autotuned %s, starting with %s",
                                name, list(init_prior))
                else:
                    init_prior = np.empty(len(prior_shape), dtype=object)
                    for t in range(len(prior_shape)):
                        x = prior_shape[t]
                        init_prior[t] = np.fromiter(
                            (1.0 / self.num_topics for i in range(x)), dtype=self.dtype, count=x)
            else:
                raise ValueError(
                    "Unable to determine proper %s value given '%s'" % (name, prior))
        elif isinstance(prior, list):
            init_prior = np.asarray(prior, dtype=self.dtype)
        elif isinstance(prior, np.ndarray):
            init_prior = prior.astype(self.dtype, copy=False)
        elif isinstance(prior, (np.number, numbers.Real)):
            if name == 'alpha':
                init_prior = np.fromiter(
                    (prior for i in range(prior_shape)), dtype=self.dtype)
            else:
                init_prior = np.empty(len(prior_shape), dtype=object)
                for t in range(len(prior_shape)):
                    x = prior_shape[t]
                    init_prior[t] = np.fromiter(
                        (prior for i in range(x)), dtype=self.dtype, count=x)
        else:
            raise ValueError(
                "%s must be either a np array of scalars, list of scalars, or scalar" % name)

        return init_prior, is_auto

    def __str__(self):
        """Get a string representation of the current object.

        Returns
        -------
        str
            Human readable representation of the most important model parameters.

        """
        return "%s<num_terms=%s, num_topics=%s, decay=%s, chunksize=%s>" % (
            self.__class__.__name__, self.num_terms, self.num_topics, self.decay, self.chunksize
        )

    def sync_state(self, current_Elogbeta=None):
        """Propagate the states topic probabilities to the inner object's attribute.

        Parameters
        ----------
        current_Elogbeta: numpy.ndarray
            Posterior probabilities for each topic, optional.
            If omitted, it will get Elogbeta from state.

        """
        if current_Elogbeta is None:
            current_Elogbeta = self.state.get_Elogbeta()
        for t in range(len(current_Elogbeta)):
            self.expElogbeta[t] = np.exp(current_Elogbeta[t])
        assert self.expElogbeta[0].dtype == self.dtype

    def clear(self):
        """Clear the model's state to free some memory. Used in the distributed implementation."""
        self.state = None
        self.Elogbeta = None

    def inference(self, chunk, collect_sstats=False):
        """Given a chunk of sparse document vectors, estimate gamma (parameters controlling the topic weights)
        for each document in the chunk.

        This function does not modify the model. The whole input chunk of document is assumed to fit in RAM;
        chunking of a large corpus must be done earlier in the pipeline. Avoids computing the `phi` variational
        parameter directly using the optimization presented in
        `Lee, Seung: Algorithms for non-negative matrix factorization"
        <https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf>`_.

        Parameters
        ----------
        chunk : list of list of (int, float)
            The corpus chunk on which the inference step will be performed.
        collect_sstats : bool, optional
            If set to True, also collect (and return) sufficient statistics needed to update the model's topic-word
            distributions.

        Returns
        -------
        (numpy.ndarray, {numpy.ndarray, None})
            The first element is always returned and it corresponds to the states gamma matrix. The second element is
            only returned if `collect_sstats` == True and corresponds to the sufficient statistics for the M step.

        """
        try:
            len(chunk[0])
        except TypeError:
            # convert iterators/generators to plain list, so we have len() etc.
            chunk = [list(x) for x in chunk]
        if len(chunk[0]) > 1:
            logger.debug(
                "performing inference on a chunk of %i documents", len(chunk[0]))

        # Initialize the variational distribution q(theta|gamma) for the chunk
        gamma = self.random_state.gamma(100., 1. / 100., (len(chunk[0]), self.num_topics)).astype(
            self.dtype, copy=False)  # dim=(number of docs, number of topics)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)

        assert Elogtheta.dtype == self.dtype
        assert expElogtheta.dtype == self.dtype

        if collect_sstats:
            sstats = np.empty(len(self.expElogbeta), dtype=object)
            for t in range(len(self.expElogbeta)):
                sstats[t] = np.zeros_like(
                    self.expElogbeta[t], dtype=self.dtype)
        else:
            sstats = None
        converged = 0

        # Now, for each document d update that document's gamma and phi
        # Inference code copied from Hoffman's `onlineldavb.py` (esp. the
        # Lee&Seung trick which speeds things up by an order of magnitude, compared
        # to Blei's original LDA-C code, cool!).
        integer_types = (int, np.integer,)
        epsilon = np.finfo(self.dtype).eps
        for d in range(len(chunk[0])):  # d -- doc

            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]

            idsm = []
            cts = []
            expElogbetad = []
            phinorm = []
            for t in range(len(chunk)):  # t -- type
                doc = chunk[t][d]
                if len(doc) > 0 and not isinstance(doc[0][0], integer_types):
                    # make sure the term IDs are ints, otherwise np will get upset
                    ids = [int(idx) for idx, _ in doc]
                else:
                    ids = [idx for idx, _ in doc]
                idsm.append(ids)
                cts.append(np.fromiter((cnt for _, cnt in doc),
                           dtype=self.dtype, count=len(doc)))
                expElogbetadt = self.expElogbeta[t][:, ids]
                expElogbetad.append(expElogbetadt)

                # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_kw.
                # phinorm is the normalizer.
                # TODO treat zeros explicitly, instead of adding epsilon?
                phinorm.append(np.dot(expElogthetad, expElogbetadt) + epsilon)

                # Iterate between gamma and phi until convergence
            for _ in range(self.iterations):
                lastgamma = gammad
                # We represent phi implicitly to save memory and time.
                # Substituting the value of the optimal phi back into
                # the update for gamma gives this update. Cf. Lee&Seung 2001.
                gammad = self.alpha + expElogthetad * \
                    sum(np.dot(x[0] / x[1], x[2].T)
                        for x in zip(cts, phinorm, expElogbetad))/len(chunk)
                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = np.exp(Elogthetad)
                phinorm = [np.dot(expElogthetad, x) +
                           epsilon for x in expElogbetad]
                # If gamma hasn't changed much, we're done.
                meanchange = mean_absolute_difference(gammad, lastgamma)
                if meanchange < self.gamma_threshold:
                    converged += 1
                    break

            gamma[d, :] = gammad
            assert gammad.dtype == self.dtype
            if collect_sstats:
                # Contribution of document d to the expected sufficient
                # statistics for the M step.
                for t in range(len(chunk)):
                    sstats[t][:, idsm[t]] += np.outer(expElogthetad.T, cts[t] / phinorm[t])

        if len(chunk[0]) > 1:
            logger.debug("%i/%i documents converged within %i iterations",
                         converged, len(chunk[0]), self.iterations)

        if collect_sstats:
            # This step finishes computing the sufficient statistics for the
            # M step, so that
            # sstats[k, w] = \sum_d n_{dw} * phi_{dwk}
            # = \sum_d n_{dw} * exp{Elogtheta_{dk} + Elogbeta_{kw}} / phinorm_{dw}.
            sstats *= self.expElogbeta
            assert sstats[0].dtype == self.dtype

        assert gamma.dtype == self.dtype
        return gamma, sstats

    def do_estep(self, chunk, state=None):
        """Perform inference on a chunk of documents, and accumulate the collected sufficient statistics.

        Parameters
        ----------
        chunk : list of list of (int, float)
            The corpus chunk on which the inference step will be performed.
        state : :class:`~LdaEHRState`, optional
            The state to be updated with the newly accumulated sufficient statistics. If none, the models
            `self.state` is updated.

        Returns
        -------
        numpy.ndarray
            Gamma parameters controlling the topic weights, shape (`len(chunk)`, `self.num_topics`).

        """
        if state is None:
            state = self.state
        gamma, sstats = self.inference(chunk, collect_sstats=True)
        state.sstats += sstats
        # avoids calling len(chunk) on a generator
        state.numdocs += gamma.shape[0]
        assert gamma.dtype == self.dtype
        return gamma

    def update_alpha(self, gammat, rho):
        """Update parameters for the Dirichlet prior on the per-document topic weights.

        Parameters
        ----------
        gammat : numpy.ndarray
            Previous topic weight parameters.
        rho : float
            Learning rate.

        Returns
        -------
        numpy.ndarray
            Sequence of alpha parameters.

        """
        N = float(len(gammat))
        logphat = sum(dirichlet_expectation(gamma) for gamma in gammat) / N
        assert logphat.dtype == self.dtype

        self.alpha = update_dir_prior(self.alpha, N, logphat, rho)
        logger.info("optimized alpha %s", list(self.alpha))

        assert self.alpha.dtype == self.dtype
        return self.alpha

    def update_eta(self, lambdat, rho):
        """Update parameters for the Dirichlet prior on the per-topic word weights.

        Parameters
        ----------
        lambdat : numpy.ndarray
            Previous lambda parameters.
        rho : float
            Learning rate.

        Returns
        -------
        numpy.ndarray
            The updated eta parameters.

        """
        for i in range(lambdat.shape[0]):
            N = float(lambdat[i].shape[0])
            logphat = (sum(dirichlet_expectation(lambda_)
                       for lambda_ in lambdat[i]) / N).reshape((self.num_terms[i],))
            assert logphat.dtype == self.dtype

            self.eta[i] = update_dir_prior(self.eta[i], N, logphat, rho)

            assert self.eta[i].dtype == self.dtype

        return self.eta

    def log_perplexity(self, chunk, total_docs=None):
        """Calculate and return per-word likelihood bound, using a chunk of documents as evaluation corpus.

        Also output the calculated statistics, including the perplexity=2^(-bound), to log at INFO level.

        Parameters
        ----------
        chunk : list of list of (int, float)
            The corpus chunk on which the inference step will be performed.
        total_docs : int, optional
            Number of docs used for evaluation of the perplexity.

        Returns
        -------
        numpy.ndarray
            The variational bound score calculated for each word.

        """
        if total_docs is None:
            total_docs = len(chunk[0])
        corpus_words = sum(
            cnt for doc_m in chunk for doc in doc_m for _, cnt in doc)
        subsample_ratio = 1.0 * total_docs / len(chunk[0])
        perwordbound = self.bound(
            chunk, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
        logger.info(
            "%.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words",
            perwordbound, np.exp2(-perwordbound), len(chunk[0]), corpus_words
        )
        return perwordbound

    def update(self, corpus, chunksize=None, decay=None, offset=None,
               passes=None, update_every=None, eval_every=None, iterations=None,
               gamma_threshold=None, chunks_as_numpy=False):
        """Train the model with new documents, by EM-iterating over the corpus until the topics converge, or until
        the maximum number of allowed iterations is reached. `corpus` must be an iterable.

        In distributed mode, the E step is distributed over a cluster of machines.

        Notes
        -----
        This update also supports updating an already trained model (`self`) with new documents from `corpus`;
        the two models are then merged in proportion to the number of old vs. new documents.
        This feature is still experimental for non-stationary input streams.

        For stationary input (no topic drift in new documents), on the other hand,
        this equals the online update of `'Online Learning for LDA' by Hoffman et al.`_
        and is guaranteed to converge for any `decay` in (0.5, 1].
        Additionally, for smaller corpus sizes,
        an increasing `offset` may be beneficial (see Table 1 in the same paper).

        Parameters
        ----------
        corpus :list of corpus ( iterable of list of (int, float)), optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`) used to update the
            model.
        chunksize :  int, optional
            Number of documents to be used in each training chunk.
        decay : float, optional
            A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten
            when each new document is examined. Corresponds to :math:`\\kappa` from
            `'Online Learning for LDA' by Hoffman et al.`_
        offset : float, optional
            Hyper-parameter that controls how much we will slow down the first steps the first few iterations.
            Corresponds to :math:`\\tau_0` from `'Online Learning for LDA' by Hoffman et al.`_
        passes : int, optional
            Number of passes through the corpus during training.
        update_every : int, optional
            Number of documents to be iterated through for each update.
            Set to 0 for batch learning, > 1 for online iterative learning.
        eval_every : int, optional
            Log perplexity is estimated every that many updates. Setting this to one slows down training by ~2x.
        iterations : int, optional
            Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
        gamma_threshold : float, optional
            Minimum change in the value of the gamma parameters to continue iterating.
        chunks_as_numpy : bool, optional
            Whether each chunk passed to the inference step should be a numpy.ndarray or not. Numpy can in some settings
            turn the term IDs into floats, these will be converted back into integers in inference, which incurs a
            performance hit. For distributed computing it may be desirable to keep the chunks as `numpy.ndarray`.

        """
        # use parameters given in constructor, unless user explicitly overrode them
        if decay is None:
            decay = self.decay
        if offset is None:
            offset = self.offset
        if passes is None:
            passes = self.passes
        if update_every is None:
            update_every = self.update_every
        if eval_every is None:
            eval_every = self.eval_every
        if iterations is None:
            iterations = self.iterations
        if gamma_threshold is None:
            gamma_threshold = self.gamma_threshold

        try:
            lencorpus = len(corpus[0])
        except Exception:
            logger.warning(
                "input corpus stream has no len(); counting documents")
            lencorpus = sum(1 for _ in corpus[0])
        if lencorpus == 0:
            logger.warning("LdaModel.update() called with an empty corpus")
            return

        if chunksize is None:
            chunksize = min(lencorpus, self.chunksize)

        self.state.numdocs += lencorpus

        if update_every:
            updatetype = "online"
            if passes == 1:
                updatetype += " (single-pass)"
            else:
                updatetype += " (multi-pass)"
            updateafter = min(lencorpus, update_every *
                              self.numworkers * chunksize)
        else:
            updatetype = "batch"
            updateafter = lencorpus
        evalafter = min(lencorpus, (eval_every or 0)
                        * self.numworkers * chunksize)

        updates_per_pass = max(1, lencorpus / updateafter)
        logger.info(
            "running %s LDA training, %s topics, %i passes over "
            "the supplied corpus of %i documents, updating model once "
            "every %i documents, evaluating perplexity every %i documents, "
            "iterating %ix with a convergence threshold of %f",
            updatetype, self.num_topics, passes, lencorpus,
            updateafter, evalafter, iterations,
            gamma_threshold
        )

        if updates_per_pass * passes < 10:
            logger.warning(
                "too few updates, training might not converge; "
                "consider increasing the number of passes or iterations to improve accuracy"
            )

        # rho is the "speed" of updating; TODO try other fncs
        # pass_ + num_updates handles increasing the starting t for each pass,
        # while allowing it to "reset" on the first pass of each update
        def rho():
            return pow(offset + pass_ + (self.num_updates / chunksize), -decay)

        if self.callbacks:
            # pass the list of input callbacks to Callback class
            callback = Callback(self.callbacks)
            callback.set_model(self)
            # initialize metrics list to store metric values after every epoch
            self.metrics = defaultdict(list)

        for pass_ in range(passes):
            if self.dispatcher:
                logger.info('initializing %s workers', self.numworkers)
                self.dispatcher.reset(self.state)
            else:
                other = LdaEHRState(
                    self.eta, [x.shape for x in self.state.sstats], self.dtype)
            dirty = False

            reallen = 0
            # list of chunks of corpus
            chunks = [utils.grouper(x, chunksize, as_numpy=chunks_as_numpy,
                                    dtype=self.dtype) for x in corpus]

            for chunk in zip(*(enumerate(x) for x in chunks)):
                chunk_no = chunk[0][0]
                chunk_doc = [x[1] for x in chunk]
                # keep track of how many documents we've processed so far
                reallen += len(chunk[0][1])

                if eval_every and ((reallen == lencorpus) or ((chunk_no + 1) % (eval_every * self.numworkers) == 0)):
                    self.log_perplexity(chunk_doc, total_docs=lencorpus)

                if self.dispatcher:
                    # add the chunk to dispatcher's job queue, so workers can munch on it
                    logger.info(
                        "PROGRESS: pass %i, dispatching documents up to #%i/%i",
                        pass_, chunk_no * chunksize +
                        len(chunk[0][1]), lencorpus
                    )
                    # this will eventually block until some jobs finish, because the queue has a small finite length
                    self.dispatcher.putjob(chunk_doc)
                else:
                    logger.info(
                        "PROGRESS: pass %i, at document #%i/%i",
                        pass_, chunk_no * chunksize +
                        len(chunk[0][1]), lencorpus
                    )
                    gammat = self.do_estep(chunk_doc, other)
                    if self.optimize_alpha:
                        self.update_alpha(gammat, rho())

                dirty = True
                del chunk

                # perform an M step. determine when based on update_every, don't do this after every chunk
                if update_every and (chunk_no + 1) % (update_every * self.numworkers) == 0:
                    if self.dispatcher:
                        # distributed mode: wait for all workers to finish
                        logger.info(
                            "reached the end of input; now waiting for all remaining jobs to finish")
                        other = self.dispatcher.getstate()
                    self.do_mstep(rho(), other, pass_ > 0)
                    del other  # frees up memory

                    if self.dispatcher:
                        logger.info('initializing workers')
                        self.dispatcher.reset(self.state)
                    else:
                        other = LdaEHRState(
                            self.eta, [x.shape for x in self.state.sstats], self.dtype)
                    dirty = False

            if reallen != lencorpus:
                raise RuntimeError(
                    "input corpus size changed during training (don't use generators as input)")

            # append current epoch's metric values
            if self.callbacks:
                current_metrics = callback.on_epoch_end(pass_)
                for metric, value in current_metrics.items():
                    self.metrics[metric].append(value)

            if dirty:
                # finish any remaining updates
                if self.dispatcher:
                    # distributed mode: wait for all workers to finish
                    logger.info(
                        "reached the end of input; now waiting for all remaining jobs to finish")
                    other = self.dispatcher.getstate()
                self.do_mstep(rho(), other, pass_ > 0)
                del other
                dirty = False

    def do_mstep(self, rho, other, extra_pass=False):
        """Maximization step: use linear interpolation between the existing topics and
        collected sufficient statistics in `other` to update the topics.

        Parameters
        ----------
        rho : float
            Learning rate.
        other : :class:`~LdaEHR`
            The model whose sufficient statistics will be used to update the topics.
        extra_pass : bool, optional
            Whether this step required an additional pass over the corpus.

        """
        logger.debug("updating topics")
        # update self with the new blend; also keep track of how much did
        # the topics change through this update, to assess convergence
        previous_Elogbeta = self.state.get_Elogbeta()
        self.state.blend(rho, other)

        current_Elogbeta = self.state.get_Elogbeta()
        self.sync_state(current_Elogbeta)

        # print out some debug info at the end of each EM iteration
        self.print_topics(5)
        ntype = len(previous_Elogbeta)
        diff = np.zeros(ntype)
        for i in range(ntype):
            diff[i] = mean_absolute_difference(
                previous_Elogbeta[i].ravel(), current_Elogbeta[i].ravel())
        logger.info("topic diff=%f, rho=%f", np.mean(diff), rho)

        if self.optimize_eta:
            self.update_eta(self.state.get_lambda(), rho)

        if not extra_pass:
            # only update if this isn't an additional pass
            self.num_updates += other.numdocs

    def bound(self, corpus, gamma=None, subsample_ratio=1.0):
        """Estimate the variational bound of documents from the corpus as E_q[log p(corpus)] - E_q[log q(corpus)].

        Parameters
        ----------
        corpus : list of corpus(iterable of list of (int, float)), optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`) used to estimate the
            variational bounds.
        gamma : numpy.ndarray, optional
            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.
        subsample_ratio : float, optional
            Percentage of the whole corpus represented by the passed `corpus` argument (in case this was a sample).
            Set to 1.0 if the whole corpus was passed.This is used as a multiplicative factor to scale the likelihood
            appropriately.

        Returns
        -------
        numpy.ndarray
            The variational bound score calculated for each document.

        """
        score = 0.0
        _lambda = self.state.get_lambda()
        Elogbeta = [dirichlet_expectation(x) for x in _lambda]

        # stream the input doc-by-doc, in case it's too large to fit in RAM
        for doc_m in zip(*(enumerate(x) for x in corpus)):
            d = doc_m[0][0]
            doc = [[x[1]] for x in doc_m]
            # for t in range(corpus):
            if d % self.chunksize == 0:
                logger.debug("bound: at document #%i", d)
            if gamma is None:
                gammad, _ = self.inference(doc)
            else:
                gammad = gamma[d]
            Elogthetad = dirichlet_expectation(gammad)

            assert gammad.dtype == self.dtype
            assert Elogthetad.dtype == self.dtype

            # E[log p(doc | theta, beta)]
            for t in range(len(doc)):
                score += sum(cnt * logsumexp(Elogthetad +
                             Elogbeta[t][:, int(id)]) for id, cnt in doc[t][0])

            # E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
            score += np.sum((self.alpha - gammad) * Elogthetad)
            score += np.sum(gammaln(gammad) - gammaln(self.alpha))
            score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(gammad))

        # Compensate likelihood for when `corpus` above is only a sample of the whole corpus. This ensures
        # that the likelihood is always roughly on the same scale.
        score *= subsample_ratio

        # E[log p(beta | eta) - log q (beta | lambda)]; assumes eta is a scalar
        for i in range(len(Elogbeta)):
            score += np.sum((self.eta[i] - _lambda[i]) * Elogbeta[i])
            score += np.sum(gammaln(_lambda[i]) - gammaln(self.eta[i]))

            if np.ndim(self.eta[i]) == 0:
                sum_eta = self.eta[i] * self.num_terms[i]
            else:
                sum_eta = np.sum(self.eta[i])

            score += np.sum(gammaln(sum_eta) - gammaln(np.sum(_lambda[i], 1)))

        return score

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """Get a representation for selected topics.

        Parameters
        ----------
        num_topics : int, optional
            Number of topics to be returned. Unlike LSA, there is no natural ordering between the topics in LDA.
            The returned topics subset of all topics is therefore arbitrary and may change between two LDA
            training runs.
        num_words : int, optional
            Number of words to be presented for each topic. These will be the most relevant words (assigned the highest
            probability for each topic).
        log : bool, optional
            Whether the output is also logged, besides being returned.
        formatted : bool, optional
            Whether the topic representations should be formatted as strings. If False, they are returned as
            2 tuples of (word, probability).

        Returns
        -------
        list of {str, tuple of (str, float)}
            a list of topics, each represented either as a string (when `formatted` == True) or word-probability
            pairs.

        """
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)

            # add a little random jitter, to randomize results around the same alpha
            sort_alpha = self.alpha + 0.0001 * \
                self.random_state.rand(len(self.alpha))
            # random_state.rand returns float64, but converting back to dtype won't speed up anything

            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[:num_topics //
                                          2] + sorted_topics[-num_topics // 2:]

        shown = []

        topic = self.state.get_lambda()
        for i in chosen_topics:
            for t in range(len(topic)):
                topic_ = topic[t][i]
                topic_ = topic_ / topic_.sum()  # normalize to probability distribution
                bestn = matutils.argsort(topic_, num_words, reverse=True)
                topic_ = [(self.id2word[t][id], topic_[id]) for id in bestn]
                if formatted:
                    topic_ = ' + '.join('%.3f*"%s"' % (v, k)
                                        for k, v in topic_)

                shown.append((i, t, topic_))
                if log:
                    logger.info("topic #%i (%.3f): %s",
                                i, self.alpha[i], topic_)

        return shown

    def show_topic(self, topicid, topn=10):
        """Get the representation for a single topic. Words here are the actual strings, in constrast to
        :meth:`~LdaModel.get_topic_terms` that represents words by their vocabulary ID.

        Parameters
        ----------
        topicid : int
            The ID of the topic to be returned
        topn : int, optional
            Number of the most significant words that are associated with the topic.

        Returns
        -------
        list of (str, float)
            Word - probability pairs for the most relevant words generated by the topic.

        """

        return [[(self.id2word[t][id], value) for id, value in self.get_topic_terms(topicid, topn)[t]] for t in range(len(self.id2word))]

    def get_topics(self):
        """Get the term-topic matrix learned during inference.

        Returns
        -------
        list of T (# of types) numpy.ndarray
            The probability for each word in each topic, shape (`num_topics`, `vocabulary_size`).

        """
        topics_lst = self.state.get_lambda()
        return [topics / topics.sum(axis=1)[:, None] for topics in topics_lst]

    def get_topic_terms(self, topicid, topn=10):
        """Get the representation for a single topic. Words the integer IDs, in constrast to
        :meth:`~gensim.models.ldamodel.LdaModel.show_topic` that represents words by the actual strings.

        Parameters
        ----------
        topicid : int
            The ID of the topic to be returned
        topn : int, optional
            Number of the most significant words that are associated with the topic.

        Returns
        -------
        list of (int, float)
            Word ID - probability pairs for the most relevant words generated by the topic.

        """
        res = []
        topic_lst = self.get_topics()
        for i in range(len(topic_lst)):
            topic = topic_lst[i][topicid]
            topic = topic / topic.sum()  # normalize to probability distribution
            bestn = matutils.argsort(topic, topn, reverse=True)
            res.append([(idx, topic[idx]) for idx in bestn])
        return res

    def get_document_topics(self, bow, minimum_probability=None, minimum_phi_value=None,
                            per_word_topics=False):
        """Get the topic distribution for the given document.

        Parameters
        ----------
        bow : corpus : list of (list of (int, float))
            The document in BOW format.
        minimum_probability : float
            Topics with an assigned probability lower than this threshold will be discarded.
        minimum_phi_value : float
            If `per_word_topics` is True, this represents a lower bound on the term probabilities that are included.
             If set to None, a value of 1e-8 is used to prevent 0s.
        per_word_topics : bool
            If True, this function will also return two extra lists as explained in the "Returns" section.

        Returns
        -------
        list of (int, float)
            Topic distribution for the whole document. Each element in the list is a pair of a topic's id, and
            the probability that was assigned to it.
        list of (int, list of (int, float), optional
            Most probable topics per word. Each element in the list is a pair of a word's id, and a list of
            topics sorted by their relevance to this word. Only returned if `per_word_topics` was set to True.
        list of (int, list of float), optional
            Phi relevance values, multiplied by the feature length, for each word-topic combination.
            Each element in the list is a pair of a word's id and a list of the phi values between this word and
            each topic. Only returned if `per_word_topics` was set to True.

        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        # never allow zero values in sparse output
        minimum_probability = max(minimum_probability, 1e-8)

        if minimum_phi_value is None:
            minimum_phi_value = self.minimum_probability
        # never allow zero values in sparse output
        minimum_phi_value = max(minimum_phi_value, 1e-8)

        # if the input vector is a corpus, return a transformed corpus
        # is_corpus, corpus = utils.is_corpus(bow)
        # if is_corpus:
        #     kwargs = dict(
        #         per_word_topics=per_word_topics,
        #         minimum_probability=minimum_probability,
        #         minimum_phi_value=minimum_phi_value
        #     )
        #     return self._apply(corpus, **kwargs)

        gamma, phis = self.inference([bow], collect_sstats=per_word_topics)
        # sum(gamma)-- sum the prob from different types. Then normalize distribution
        topic_dist = sum(gamma) / sum(sum(gamma))

        document_topics = [
            (topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)
            if topicvalue >= minimum_probability
        ]

        if not per_word_topics:
            return document_topics

        word_topic = []  # contains word and corresponding topic
        word_phi = []  # contains word and phi values
        for t in range(len(bow)):
            for word_type, weight in bow[t]:
            # contains (phi_value, topic) pairing to later be sorted
                phi_values = []
                phi_topic = []  # contains topic and corresponding phi value to be returned 'raw' to user
                for topic_id in range(0, self.num_topics):
                    phi_twk = phis[t][topic_id][word_type]
                    if phi_twk >= minimum_phi_value:
                        # appends phi values for each topic for that word
                        # these phi values are scaled by feature length
                        phi_values.append((phi_twk, topic_id, t))
                        phi_topic.append((topic_id, phi_twk, t))

                # list with ({word_id => [(topic_0, phi_value), (topic_1, phi_value) ...]).
                word_phi.append((word_type, phi_topic))
                # sorts the topics based on most likely topic
                # returns a list like ({word_id => [topic_id_most_probable, topic_id_second_most_probable, ...]).
                sorted_phi_values = sorted(phi_values, reverse=True)
                topics_sorted = [x[1] for x in sorted_phi_values]
                word_topic.append((word_type, topics_sorted))

        return document_topics, word_topic, word_phi  # returns 2-tuple

    def get_term_topics(self, word_id, type_id, minimum_probability=None):
        """Get the most relevant topics to the given word.

        Parameters
        ----------
        word_id : int or str
            The word for which the topic distribution will be computed.
        type_id : int
            The type id for which the word is coming from.
        minimum_probability : float, optional
            Topics with an assigned probability below this threshold will be discarded.

        Returns
        -------
        list of (int, float)
            The relevant topics represented as pairs of their ID and their assigned probability, sorted
            by relevance to the given word.

        """
        if minimum_probability is None:
            minimum_probability = self.minimum_probability
        # never allow zero values in sparse output
        minimum_probability = max(minimum_probability, 1e-8)

        # if user enters word instead of id in vocab, change to get id
        if isinstance(word_id, str):
            word_id = self.id2word[type_id].doc2bow([word_id])[0][0]

        values = []
        for topic_id in range(0, self.num_topics):
            if self.expElogbeta[type_id][topic_id][word_id] >= minimum_probability:
                values.append(
                    (topic_id, self.expElogbeta[type_id][topic_id][word_id]))

        return values


    def __getitem__(self, bow, eps=None):
        """Get the topic distribution for the given document.

        Wraps :meth:`~gensim.models.ldamodel.LdaEHR.get_document_topics` to support an operator style call.
        Uses the model's current state (set using constructor arguments) to fill in the additional arguments of the
        wrapper method.

        Parameters
        ---------
        bow : list of (int, float)
            The document in BOW format.
        eps : float, optional
            Topics with an assigned probability lower than this threshold will be discarded.

        Returns
        -------
        list of (int, float)
            Topic distribution for the given document. Each topic is represented as a pair of its ID and the probability
            assigned to it.

        """
        return self.get_document_topics(bow, eps, self.minimum_phi_value, self.per_word_topics)

    def save(self, fname, ignore=('state', 'dispatcher'), separately=None, *args, **kwargs):
        """Save the model to a file.

        Large internal arrays may be stored into separate files, with `fname` as prefix.

        Notes
        -----
        If you intend to use models across Python 2/3 versions there are a few things to
        keep in mind:

          1. The pickled Python dictionaries will not work across Python versions
          2. The `save` method does not automatically save all numpy arrays separately, only
             those ones that exceed `sep_limit` set in :meth:`~gensim.utils.SaveLoad.save`. The main
             concern here is the `alpha` array if for instance using `alpha='auto'`.

        Please refer to the `wiki recipes section
        <https://github.com/RaRe-Technologies/gensim/wiki/
        Recipes-&-FAQ#q9-how-do-i-load-a-model-in-python-3-that-was-trained-and-saved-using-python-2>`_
        for an example on how to work around these issues.

        See Also
        --------
        :meth:`~gensim.models.ldamodel.LdaEHR.load`
            Load model.

        Parameters
        ----------
        fname : str
            Path to the system file where the model will be persisted.
        ignore : tuple of str, optional
            The named attributes in the tuple will be left out of the pickled model. The reason why
            the internal `state` is ignored by default is that it uses its own serialisation rather than the one
            provided by this method.
        separately : {list of str, None}, optional
            If None -  automatically detect large numpy/scipy.sparse arrays in the object being stored, and store
            them into separate files. This avoids pickle memory errors and allows `mmap`'ing large arrays
            back on load efficiently. If list of str - this attributes will be stored in separate files,
            the automatic check is not performed in this case.
        *args
            Positional arguments propagated to :meth:`~gensim.utils.SaveLoad.save`.
        **kwargs
            Key word arguments propagated to :meth:`~gensim.utils.SaveLoad.save`.

        """
        if self.state is not None:
            self.state.save(utils.smart_extension(
                fname, '.state'), *args, **kwargs)
        # Save the dictionary separately if not in 'ignore'.
        if 'id2word' not in ignore:
            utils.pickle(
                self.id2word, utils.smart_extension(fname, '.id2word'))

        # make sure 'state', 'id2word' and 'dispatcher' are ignored from the pickled object, even if
        # someone sets the ignore list themselves
        if ignore is not None and ignore:
            if isinstance(ignore, str):
                ignore = [ignore]
            # make sure None and '' are not in the list
            ignore = [e for e in ignore if e]
            ignore = list({'state', 'dispatcher', 'id2word'} | set(ignore))
        else:
            ignore = ['state', 'dispatcher', 'id2word']

        # make sure 'expElogbeta' and 'sstats' are ignored from the pickled object, even if
        # someone sets the separately list themselves.
        separately_explicit = ['expElogbeta', 'sstats']
        # Also add 'alpha' and 'eta' to separately list if they are set 'auto' or some
        # array manually.
        if (isinstance(self.alpha, str) and self.alpha == 'auto') or \
                (isinstance(self.alpha, np.ndarray) and len(self.alpha.shape) != 1):
            separately_explicit.append('alpha')
        if (isinstance(self.eta, str) and self.eta == 'auto') or \
                (isinstance(self.eta, np.ndarray) and len(self.eta.shape) != 1):
            separately_explicit.append('eta')
        # Merge separately_explicit with separately.
        if separately:
            if isinstance(separately, str):
                separately = [separately]
            # make sure None and '' are not in the list
            separately = [e for e in separately if e]
            separately = list(set(separately_explicit) | set(separately))
        else:
            separately = separately_explicit
        super(LdaEHR, self).save(fname, ignore=ignore,
                                 separately=separately, *args, **kwargs)




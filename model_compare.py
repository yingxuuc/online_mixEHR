from gensim.matutils import (
    kullback_leibler, hellinger, jaccard_distance, jensen_shannon,
    dirichlet_expectation, logsumexp, mean_absolute_difference,
)
import numpy as np

# rescaled dot product: 
def rescaled_dot_prod(x, y):
    d_max = np.dot(np.sort(x), np.sort(y))
    d_min = np.dot(np.sort(x), np.sort(y)[::-1])
    return((np.dot(x, y) - d_min)/(d_max - d_min))

# This function is based on the diff function here:
#  https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/ldamodel.py
def diff(m1, m2, distance="kullback_leibler", num_words=30,
             n_ann_terms=10, diagonal=False, annotation=True, normed=True):
        """Calculate the difference in topic distributions between two models: `self` and `other`.

        Parameters
        ----------
        other : :class:`~gensim.models.ldamodel.LdaEHR`
            The model which will be compared against the current object.
        distance : {'kullback_leibler', 'hellinger', 'jaccard', 'jensen_shannon'}
            The distance metric to calculate the difference with.
        num_words : int, optional
            The number of most relevant words used if `distance == 'jaccard'`. Also used for annotating topics.
        n_ann_terms : int, optional
            Max number of words in intersection/symmetric difference between topics. Used for annotation.
        diagonal : bool, optional
            Whether we need the difference between identical topics (the diagonal of the difference matrix).
        annotation : bool, optional
            Whether the intersection or difference of words between two topics should be returned.
        normed : bool, optional
            Whether the matrix should be normalized or not.

        Returns
        -------
        numpy.ndarray
            A difference matrix. Each element corresponds to the difference between the two topics,
            shape (`self.num_topics`, `other.num_topics`)
        numpy.ndarray, optional
            Annotation matrix where for each pair we include the word from the intersection of the two topics,
            and the word from the symmetric difference of the two topics. Only included if `annotation == True`.
            Shape (`self.num_topics`, `other_model.num_topics`, 2).

        Examples
        --------
        Get the differences between each pair of topics inferred by two models

        .. sourcecode:: pycon

            >>> from gensim.models.ldamulticore import LdaMulticore
            >>> from gensim.test.utils import datapath
            >>>
            >>> m1 = LdaMulticore.load(datapath("lda_3_0_1_model"))
            >>> m2 = LdaMulticore.load(datapath("ldamodel_python_3_5"))
            >>> mdiff, annotation = m1.diff(m2)
            >>> topic_diff = mdiff  # get matrix with difference for each topic pair from `m1` and `m2`

        """
        distances = {
            "kullback_leibler": kullback_leibler,
            "hellinger": hellinger,
            "jaccard": jaccard_distance,
            "jensen_shannon": jensen_shannon,
            "rescale_dot": rescaled_dot_prod
        }

        if distance not in distances:
            valid_keys = ", ".join("`{}`".format(x) for x in distances.keys())
            raise ValueError(
                "Incorrect distance, valid only {}".format(valid_keys))

        if not isinstance(m2, m1.__class__):
            raise ValueError(
                "The parameter `m2` must be of type `{}`".format(m1.__name__))

        distance_func = distances[distance]
        d1, d2 = m1.get_topics(), m2.get_topics()
        d1a = d1[0]
        d2a = d2[0]
        for i in range(len(d1)-1):
            d1a = np.append(d1a, d1[i+1], axis = 1)
            d2a = np.append(d2a, d2[i+1], axis = 1)

        t1_size, t2_size = d1a.shape[0], d2a.shape[0]
        annotation_terms = None

        fst_topics = [{w for s in m1.show_topic(
            topic, topn=num_words) for (w, _) in s} for topic in range(t1_size)]
        snd_topics = [{w for s in m2.show_topic(
            topic, topn=num_words) for (w, _) in s} for topic in range(t2_size)]

        if distance == "jaccard":
            d1a, d2a = fst_topics, snd_topics

        if diagonal:
            assert t1_size == t2_size, \
                "Both input models should have same no. of topics, " \
                "as the diagonal will only be valid in a square matrix"
            # initialize z and annotation array
            z = np.zeros(t1_size)
            if annotation:
                annotation_terms = np.zeros(t1_size, dtype=list)
        else:
            # initialize z and annotation matrix
            z = np.zeros((t1_size, t2_size))
            if annotation:
                annotation_terms = np.zeros((t1_size, t2_size), dtype=list)

        # iterate over each cell in the initialized z and annotation
        for topic in np.ndindex(z.shape):
            topic1 = topic[0]
            if diagonal:
                topic2 = topic1
            else:
                topic2 = topic[1]
            
            z[topic] = distance_func(d1a[topic1], d2a[topic2])
            
            if annotation:
                pos_tokens = fst_topics[topic1] & snd_topics[topic2]
                neg_tokens = fst_topics[topic1].symmetric_difference(
                    snd_topics[topic2])

                pos_tokens = list(pos_tokens)[:min(
                    len(pos_tokens), n_ann_terms)]
                neg_tokens = list(neg_tokens)[:min(
                    len(neg_tokens), n_ann_terms)]

                annotation_terms[topic] = [pos_tokens, neg_tokens]

        if normed:
            if np.abs(np.max(z)) > 1e-8:
                z /= np.max(z)

        return z, annotation_terms

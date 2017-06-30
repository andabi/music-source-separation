# CREATED:2015-09-16 14:46:47 by Brian McFee <brian.mcfee@nyu.edu>
# -*- encoding: utf-8 -*-
'''Evaluation criteria for hierarchical structure analysis.

Hierarchical structure analysis seeks to annotate a track with a nested
decomposition of the temporal elements of the piece, effectively providing
a kind of "parse tree" of the composition.  Unlike the flat segmentation
metrics defined in :mod:`mir_eval.segment`, which can only encode one level of
analysis, hierarchical annotations expose the relationships between short
segments and the larger compositional elements to which they belong.

Currently, there exist no metrics for evaluating hierarchical segment
labeling.  All evaluations are therefore based on boundaries between
segments (and relationships between segments across levels), and not the
labels applied to segments.


Conventions
-----------
Annotations are assumed to take the form of an ordered list of segmentations.
As in the :mod:`mir_eval.segment` metrics, each segmentation itself consists of
an n-by-2 array of interval times, so that the ``i`` th segment spans time
``intervals[i, 0]`` to ``intervals[i, 1]``.

Hierarchical annotations are ordered by increasing specificity, so that the
first segmentation should contain the fewest segments, and the last
segmentation contains the most.

Metrics
-------
* :func:`mir_eval.hierarchy.tmeasure`: Precision, recall, and F-measure of
  triplet-based frame accuracy.

References
----------
  .. [#mcfee2015] Brian McFee, Oriol Nieto, and Juan P. Bello.
    "Hierarchical evaluation of segment boundary detection",
    International Society for Music Information Retrieval (ISMIR) conference,
    2015.

'''

import numpy as np
import scipy.sparse
import collections
import itertools
import warnings

from . import util
from .segment import validate_structure


def _round(t, frame_size):
    '''Round a time-stamp to a specified resolution.

    Equivalent to ``t - np.mod(t, frame_size)``.

    Examples
    --------
    >>> _round(53.279, 0.1)
    53.2
    >>> _round(53.279, 0.25)
    53.25

    Parameters
    ----------
    t : number or ndarray
        The time-stamp to round

    frame_size : number > 0
        The resolution to round to

    Returns
    -------
    t_round : number
        The rounded time-stamp
    '''
    return t - np.mod(t, float(frame_size))


def _hierarchy_bounds(intervals_hier):
    '''Compute the covered time range of a hierarchical segmentation.

    Parameters
    ----------
    intervals_hier : list of ndarray
        A hierarchical segmentation, encoded as a list of arrays of segment
        intervals.

    Returns
    -------
    t_min : float
    t_max : float
        The minimum and maximum times spanned by the annotation
    '''
    boundaries = list(itertools.chain(*list(itertools.chain(*intervals_hier))))

    return min(boundaries), max(boundaries)


def _lca(intervals_hier, frame_size):
    '''Compute the (sparse) least-common-ancestor (LCA) matrix for a
    hierarchical segmentation.

    For any pair of frames ``(s, t)``, the LCA is the deepest level in
    the hierarchy such that ``(s, t)`` are contained within a single
    segment at that level.

    Parameters
    ----------
    intervals_hier : list of ndarray
        An ordered list of segment interval arrays.
        The list is assumed to be ordered by increasing specificity (depth).

    frame_size : number
        The length of the sample frames (in seconds)

    Returns
    -------
    lca_matrix : scipy.sparse.csr_matrix
        A sparse matrix such that ``lca_matrix[i, j]`` contains the depth
        of the deepest segment containing frames ``i`` and ``j``.
    '''

    frame_size = float(frame_size)

    # Figure out how many frames we need

    n_start, n_end = _hierarchy_bounds(intervals_hier)

    n = int((_round(n_end, frame_size) -
             _round(n_start, frame_size)) / frame_size)

    # Initialize the LCA matrix
    lca_matrix = scipy.sparse.lil_matrix((n, n), dtype=np.uint8)

    for level, intervals in enumerate(intervals_hier, 1):
        for ival in (_round(np.asarray(intervals),
                            frame_size) / frame_size).astype(int):
            idx = slice(ival[0], ival[1])
            lca_matrix[idx, idx] = level

    return lca_matrix.tocsr()


def _gauc(ref_lca, est_lca, transitive, window):
    '''Generalized area under the curve (GAUC)

    This function computes the normalized recall score for correctly
    ordering triples ``(q, i, j)`` where frames ``(q, i)`` are closer than
    ``(q, j)`` in the reference annotation.

    Parameters
    ----------
    ref_lca : scipy.sparse
    est_lca : scipy.sparse
        The least common ancestor matrices for the reference and
        estimated annotations

    transitive : bool
        If True, then transitive comparisons are counted, meaning that
        ``(q, i)`` and ``(q, j)`` can differ by any number of levels.

        If False, then ``(q, i)`` and ``(q, j)`` can differ by exactly one
        level.

    window : number or None
        The maximum number of frames to consider for each query.
        If `None`, then all frames are considered.

    Returns
    -------
    score : number [0, 1]
        The percentage of reference triples correctly ordered by
        the estimation.

    Raises
    ------
    ValueError
        If ``ref_lca`` and ``est_lca`` have different shapes
    '''
    # Make sure we have the right number of frames

    if ref_lca.shape != est_lca.shape:
        raise ValueError('Estimated and reference hierarchies '
                         'must have the same shape.')

    # How many frames?
    n = ref_lca.shape[0]

    # By default, the window covers the entire track
    if window is None:
        window = n

    # Initialize the score
    score = 0.0

    # Iterate over query frames
    num_frames = 0

    for query in range(n):

        # Find all pairs i,j such that ref_lca[q, i] > ref_lca[q, j]
        results = slice(max(0, query - window), min(n, query + window))

        ref_score = ref_lca[query, results]
        est_score = est_lca[query, results]

        # Densify the results
        ref_score = np.asarray(ref_score.todense()).squeeze()
        est_score = np.asarray(est_score.todense()).squeeze()

        if transitive:
            # Transitive: count comparisons across any level
            ref_rank = np.greater.outer(ref_score, ref_score)
        else:
            # Non-transitive: count comparisons only across immediate levels
            ref_rank = np.equal.outer(ref_score, ref_score + 1)

        est_rank = np.greater.outer(est_score, est_score)

        # Don't count the query as a result
        # when query < window, query itself is the index within the slice
        # otherwise, query is located at the center of the slice, window
        # (this also holds when the slice goes off the end of the array.)
        idx = min(query, window)
        ref_rank[idx, :] = False
        ref_rank[:, idx] = False

        # Compute normalization constant
        normalizer = float(ref_rank.sum())

        # Add up agreement for frames
        if normalizer > 0:
            score += np.sum(np.logical_and(ref_rank, est_rank)) / normalizer
            num_frames += 1

    # Normalize by the number of frames counted.
    # If no frames are counted, take the convention 0/0 -> 0
    if num_frames:
        score /= float(num_frames)
    else:
        score = 0.0

    return score


def validate_hier_intervals(intervals_hier):
    '''Validate a hierarchical segment annotation.

    Parameters
    ----------
    intervals_hier : ordered list of segmentations

    Raises
    ------
    ValueError
        If any segmentation does not span the full duration of the top-level
        segmentation.

        If any segmentation does not start at 0.
    '''

    # Synthesize a label array for the top layer.
    label_top = util.generate_labels(intervals_hier[0])

    boundaries = set(util.intervals_to_boundaries(intervals_hier[0]))

    for level, intervals in enumerate(intervals_hier[1:], 1):
        # Make sure this level is consistent with the root
        label_current = util.generate_labels(intervals)
        validate_structure(intervals_hier[0], label_top,
                           intervals, label_current)

        # Make sure all previous boundaries are accounted for
        new_bounds = set(util.intervals_to_boundaries(intervals))

        if boundaries - new_bounds:
            warnings.warn('Segment hierarchy is inconsistent '
                          'at level {:d}'.format(level))
        boundaries |= new_bounds


def tmeasure(reference_intervals_hier, estimated_intervals_hier,
             transitive=False, window=15.0, frame_size=0.1, beta=1.0):
    '''Computes the tree measures for hierarchical segment annotations.

    Parameters
    ----------
    reference_intervals_hier : list of ndarray
        ``reference_intervals_hier[i]`` contains the segment intervals
        (in seconds) for the ``i`` th layer of the annotations.  Layers are
        ordered from top to bottom, so that the last list of intervals should
        be the most specific.

    estimated_intervals_hier : list of ndarray
        Like ``reference_intervals_hier`` but for the estimated annotation

    transitive : bool
        whether to compute the t-measures using transitivity or not.

    window : float > 0
        size of the window (in seconds).  For each query frame q,
        result frames are only counted within q +- window.

    frame_size : float > 0
        length (in seconds) of frames.  The frame size cannot be longer than
        the window.

    beta : float > 0
        beta parameter for the F-measure.

    Returns
    -------
    t_precision : number [0, 1]
        T-measure Precision

    t_recall : number [0, 1]
        T-measure Recall

    t_measure : number [0, 1]
        F-beta measure for ``(t_precision, t_recall)``

    Raises
    ------
    ValueError
        If either of the input hierarchies are inconsistent

        If the input hierarchies have different time durations

        If ``frame_size > window`` or ``frame_size <= 0``
    '''

    # Compute the number of frames in the window
    if frame_size <= 0:
        raise ValueError('frame_size ({:.2f}) must be a positive '
                         'number.'.format(frame_size))

    if window is None:
        window_frames = None
    else:
        if frame_size > window:
            raise ValueError('frame_size ({:.2f}) cannot exceed '
                             'window ({:.2f})'.format(frame_size, window))

        window_frames = int(_round(window, frame_size) / frame_size)

    # Validate the hierarchical segmentations
    validate_hier_intervals(reference_intervals_hier)
    validate_hier_intervals(estimated_intervals_hier)

    # Build the least common ancestor matrices
    ref_lca = _lca(reference_intervals_hier, frame_size)
    est_lca = _lca(estimated_intervals_hier, frame_size)

    # Compute precision and recall
    t_recall = _gauc(ref_lca, est_lca, transitive, window_frames)
    t_precision = _gauc(est_lca, ref_lca, transitive, window_frames)

    t_measure = util.f_measure(t_precision, t_recall, beta=beta)

    return t_precision, t_recall, t_measure


def evaluate(ref_intervals_hier, ref_labels_hier,
             est_intervals_hier, est_labels_hier, **kwargs):
    '''Compute all hierarchical structure metrics for the given reference and
    estimated annotations.

    Examples
    --------
    A toy example with two two-layer annotations

    >>> ref_i = [[[0, 30], [30, 60]], [[0, 15], [15, 30], [30, 45], [45, 60]]]
    >>> est_i = [[[0, 45], [45, 60]], [[0, 15], [15, 30], [30, 45], [45, 60]]]
    >>> ref_l = [ ['A', 'B'], ['a', 'b', 'a', 'c'] ]
    >>> est_l = [ ['A', 'B'], ['a', 'a', 'b', 'b'] ]
    >>> scores = mir_eval.hierarchy.evaluate(ref_i, ref_l, est_i, est_l)
    >>> dict(scores)
    {'T-Measure full': 0.94822745804853459,
     'T-Measure reduced': 0.8732458222764804,
     'T-Precision full': 0.96569179094693058,
     'T-Precision reduced': 0.89939075137018787,
     'T-Recall full': 0.93138358189386117,
     'T-Recall reduced': 0.84857799953694923}

    A more realistic example, using SALAMI pre-parsed annotations

    >>> def load_salami(filename):
    ...     "load SALAMI event format as labeled intervals"
    ...     events, labels = mir_eval.io.load_labeled_events(filename)
    ...     intervals = mir_eval.util.boundaries_to_intervals(events)[0]
    ...     return intervals, labels[:len(intervals)]
    >>> ref_files = ['data/10/parsed/textfile1_uppercase.txt',
    ...              'data/10/parsed/textfile1_lowercase.txt']
    >>> est_files = ['data/10/parsed/textfile2_uppercase.txt',
    ...              'data/10/parsed/textfile2_lowercase.txt']
    >>> ref = [load_salami(fname) for fname in ref_files]
    >>> ref_int = [seg[0] for seg in ref]
    >>> ref_lab = [seg[1] for seg in ref]
    >>> est = [load_salami(fname) for fname in est_files]
    >>> est_int = [seg[0] for seg in est]
    >>> est_lab = [seg[1] for seg in est]
    >>> scores = mir_eval.hierarchy.evaluate(ref_int, ref_lab,
    ...                                      est_hier, est_lab)
    >>> dict(scores)
    {'T-Measure full': 0.66029225561405358,
     'T-Measure reduced': 0.62001868041578034,
     'T-Precision full': 0.66844764668949885,
     'T-Precision reduced': 0.63252297209957919,
     'T-Recall full': 0.6523334654992341,
     'T-Recall reduced': 0.60799919710921635}


    Parameters
    ----------
    ref_intervals_hier : list of list-like
    ref_labels_hier : list of str
    est_intervals_hier : list of list-like
    est_labels_hier : list of str
        Hierarchical annotations are encoded as an ordered list
        of segmentations.  Each segmentation itself is a list (or list-like)
        of intervals (\*_intervals_hier) and a list of lists of labels
        (\*_labels_hier).

    kwargs
        additional keyword arguments to the evaluation metrics.

    Returns
    -------
    scores :  OrderedDict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.

        T-measures are computed in both the "full" (``transitive=True``) and
        "reduced" (``transitive=False``) modes.

    Raises
    ------
    ValueError
        Thrown when the provided annotations are not valid.
    '''

    # First, find the maximum length of the reference
    _, t_end = _hierarchy_bounds(ref_intervals_hier)

    # Pre-process the intervals to match the range of the reference,
    # and start at 0
    ref_intervals_hier = [util.adjust_intervals(np.asarray(_), t_min=0.0)[0]
                          for _ in ref_intervals_hier]

    est_intervals_hier = [util.adjust_intervals(np.asarray(_), t_min=0.0,
                                                t_max=t_end)[0]
                          for _ in est_intervals_hier]

    scores = collections.OrderedDict()

    # Force the transitivity setting
    kwargs['transitive'] = False
    (scores['T-Precision reduced'],
     scores['T-Recall reduced'],
     scores['T-Measure reduced']) = util.filter_kwargs(tmeasure,
                                                       ref_intervals_hier,
                                                       est_intervals_hier,
                                                       **kwargs)

    kwargs['transitive'] = True
    (scores['T-Precision full'],
     scores['T-Recall full'],
     scores['T-Measure full']) = util.filter_kwargs(tmeasure,
                                                    ref_intervals_hier,
                                                    est_intervals_hier,
                                                    **kwargs)

    return scores

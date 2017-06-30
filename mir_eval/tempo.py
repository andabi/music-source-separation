'''
The goal of a tempo estimation algorithm is to automatically detect the tempo
of a piece of music, measured in beats per minute (BPM).

See http://www.music-ir.org/mirex/wiki/2014:Audio_Tempo_Estimation for a
description of the task and evaluation criteria.

Conventions
-----------

Reference tempi should be strictly positive, and provided in ascending order
as a numpy array of length 2.  Estimated tempi are allowed to be 0, but
otherwise are subject to the same constraints as reference.

The weighting value from the reference must be a float in the range [0, 1].

Metrics
-------
* :func:`mir_eval.tempo.detection`: Relative error, hits, and weighted
  precision of tempo estimation.

'''

import numpy as np
import collections
from . import util


def validate_tempi(tempi):
    """Checks that there are two non-negative tempi.

    Parameters
    ----------
    tempi : np.ndarray
        length-2 array of tempo, in bpm
    """

    if tempi.size != 2:
        raise ValueError('tempi must have exactly two values')

    if not np.all(np.isfinite(tempi)) or np.any(tempi <= 0):
        raise ValueError('tempi={} must be non-negative numbers'.format(tempi))


def validate(reference_tempi, reference_weight, estimated_tempi):
    """Checks that the input annotations to a metric look like valid tempo
    annotations.

    Parameters
    ----------
    reference_tempi : np.ndarray
        reference tempo values, in bpm

    reference_weight : float
        perceptual weight of slow vs fast in reference

    estimated_tempi : np.ndarray
        estimated tempo values, in bpm

    """
    validate_tempi(reference_tempi)
    validate_tempi(estimated_tempi)

    if reference_weight < 0 or reference_weight > 1:
        raise ValueError('Reference weight must lie in range [0, 1]')


def detection(reference_tempi, reference_weight, estimated_tempi, tol=0.08):
    """Compute the tempo detection accuracy metric.

    Parameters
    ----------
    reference_tempi : np.ndarray, shape=(2,)
        Two non-negative reference tempi

    reference_weight : float > 0
        The relative strength of ``reference_tempi[0]`` vs
        ``reference_tempi[1]``.

    estimated_tempi : np.ndarray, shape=(2,)
        Two non-negative estimated tempi.

    tol : float in [0, 1]:
        The maximum allowable deviation from a reference tempo to
        count as a hit.
        ``|est_t - ref_t| <= tol * ref_t``
        (Default value = 0.08)

    Returns
    -------
    p_score : float in [0, 1]
        Weighted average of recalls:
        ``reference_weight * hits[0] + (1 - reference_weight) * hits[1]``

    one_correct : bool
        True if at least one reference tempo was correctly estimated

    both_correct : bool
        True if both reference tempi were correctly estimated

    Raises
    ------
    ValueError
        If the input tempi are ill-formed

        If the reference weight is not in the range [0, 1]

        If ``tol <= 0`` or ``tol > 1``.
    """

    validate(reference_tempi, reference_weight, estimated_tempi)

    if tol <= 0 or tol > 1:
        raise ValueError('invalid tolerance {}: must lie in the range '
                         '(0, 1]'.format(tol))

    relative_errors = []
    hits = []

    for ref_t in reference_tempi:
        # Compute the relative error for this reference tempo
        relative_errors.append(np.min(
            np.abs(ref_t - estimated_tempi) / float(ref_t)))

        # Count the hits
        hits.append(bool(relative_errors[-1] <= tol))

    p_score = reference_weight * hits[0] + (1.0-reference_weight) * hits[1]

    one_correct = bool(np.max(hits))
    both_correct = bool(np.min(hits))

    return p_score, one_correct, both_correct


def evaluate(reference_tempi, reference_weight, estimated_tempi, **kwargs):
    """Compute all metrics for the given reference and estimated annotations.

    Parameters
    ----------
    reference_tempi : np.ndarray, shape=(2,)
        Two non-negative reference tempi

    reference_weight : float > 0
        The relative strength of ``reference_tempi[0]`` vs
        ``reference_tempi[1]``.

    estimated_tempi : np.ndarray, shape=(2,)
        Two non-negative estimated tempi.

    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.

    Returns
    -------
    scores : dict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.
    """
    # Compute all metrics
    scores = collections.OrderedDict()

    (scores['P-score'],
     scores['One-correct'],
     scores['Both-correct']) = util.filter_kwargs(detection, reference_tempi,
                                                  reference_weight,
                                                  estimated_tempi,
                                                  **kwargs)

    return scores

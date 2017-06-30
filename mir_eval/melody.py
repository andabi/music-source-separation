# CREATED:2014-03-07 by Justin Salamon <justin.salamon@nyu.edu>
'''
Melody extraction algorithms aim to produce a sequence of frequency values
corresponding to the pitch of the dominant melody from a musical
recording.  For evaluation, an estimated pitch series is evaluated against a
reference based on whether the voicing (melody present or not) and the pitch
is correct (within some tolerance).

For a detailed explanation of the measures please refer to:
    J. Salamon, E. Gomez, D. P. W. Ellis and G. Richard, "Melody Extraction
    from Polyphonic Music Signals: Approaches, Applications and Challenges",
    IEEE Signal Processing Magazine, 31(2):118-134, Mar. 2014.

Conventions
-----------

Melody annotations are assumed to be given in the format of a 1d array of
frequency values which are accompanied by a 1d array of times denoting when
each frequency value occurs.  In a reference melody time series, a frequency
value of 0 denotes "unvoiced".  In a estimated melody time series, unvoiced
frames can be indicated either by 0 Hz or by a negative Hz value - negative
values represent the algorithm's pitch estimate for frames it has determined as
unvoiced, in case they are in fact voiced.

Metrics are computed using a sequence of reference and estimated pitches in
cents and boolean voicing arrays, both of which are sampled to the same
timebase.  The function :func:`mir_eval.melody.to_cent_voicing` can be used to
convert a sequence of estimated and reference times and frequency values in Hz
to boolean voicing arrays and frequency arrays in the format required by the
metric functions.  By default, the convention is to resample the estimated
melody time series to the reference melody time series' timebase.

Metrics
-------

* :func:`mir_eval.melody.voicing_measures`: Voicing measures, including the
  recall rate (proportion of frames labeled as melody frames in the reference
  that are estimated as melody frames) and the false alarm
  rate (proportion of frames labeled as non-melody in the reference that are
  mistakenly estimated as melody frames)
* :func:`mir_eval.melody.raw_pitch_accuracy`: Raw Pitch Accuracy, which
  computes the proportion of melody frames in the reference for which the
  frequency is considered correct (i.e. within half a semitone of the reference
  frequency)
* :func:`mir_eval.melody.raw_chroma_accuracy`: Raw Chroma Accuracy, where the
  estimated and reference frequency sequences are mapped onto a single octave
  before computing the raw pitch accuracy
* :func:`mir_eval.melody.overall_accuracy`: Overall Accuracy, which computes
  the proportion of all frames correctly estimated by the algorithm, including
  whether non-melody frames where labeled by the algorithm as non-melody

'''

import numpy as np
import scipy.interpolate
import collections
import warnings
from . import util


def validate_voicing(ref_voicing, est_voicing):
    """Checks that voicing inputs to a metric are in the correct format.

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference boolean voicing array
    est_voicing : np.ndarray
        Estimated boolean voicing array

    """
    if ref_voicing.size == 0:
        warnings.warn("Reference voicing array is empty.")
    if est_voicing.size == 0:
        warnings.warn("Estimated voicing array is empty.")
    if ref_voicing.sum() == 0:
        warnings.warn("Reference melody has no voiced frames.")
    if est_voicing.sum() == 0:
        warnings.warn("Estimated melody has no voiced frames.")
    # Make sure they're the same length
    if ref_voicing.shape[0] != est_voicing.shape[0]:
        raise ValueError('Reference and estimated voicing arrays should '
                         'be the same length.')
    for voicing in [ref_voicing, est_voicing]:
        # Make sure they're (effectively) boolean
        if np.logical_and(voicing != 0, voicing != 1).any():
            raise ValueError('Voicing arrays must be boolean.')


def validate(ref_voicing, ref_cent, est_voicing, est_cent):
    """Checks that voicing and frequency arrays are well-formed.  To be used in
    conjunction with :func:`mir_eval.melody.validate_voicing`

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference boolean voicing array
    ref_cent : np.ndarray
        Reference pitch sequence in cents
    est_voicing : np.ndarray
        Estimated boolean voicing array
    est_cent : np.ndarray
        Estimate pitch sequence in cents

    """
    if ref_cent.size == 0:
        warnings.warn("Reference frequency array is empty.")
    if est_cent.size == 0:
        warnings.warn("Estimated frequency array is empty.")
    # Make sure they're the same length
    if ref_voicing.shape[0] != ref_cent.shape[0] or \
       est_voicing.shape[0] != est_cent.shape[0] or \
       ref_cent.shape[0] != est_cent.shape[0]:
        raise ValueError('All voicing and frequency arrays must have the '
                         'same length.')


def hz2cents(freq_hz, base_frequency=10.0):
    """Convert an array of frequency values in Hz to cents.
    0 values are left in place.

    Parameters
    ----------
    freq_hz : np.ndarray
        Array of frequencies in Hz.
    base_frequency : float
        Base frequency for conversion.
        (Default value = 10.0)

    Returns
    -------
    cent : np.ndarray
        Array of frequencies in cents, relative to base_frequency

    """
    freq_cent = np.zeros(freq_hz.shape[0])
    freq_nonz_ind = np.flatnonzero(freq_hz)
    normalized_frequency = np.abs(freq_hz[freq_nonz_ind])/base_frequency
    freq_cent[freq_nonz_ind] = 1200*np.log2(normalized_frequency)

    return freq_cent


def freq_to_voicing(frequencies):
    """Convert from an array of frequency values to frequency array +
    voice/unvoiced array

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies.  A frequency <= 0 indicates "unvoiced".

    Returns
    -------
    frequencies : np.ndarray
        Array of frequencies, all >= 0.
    voiced : np.ndarray
        Boolean array, same length as frequencies,
        which indicates voiced or unvoiced

    """
    return np.abs(frequencies), frequencies > 0


def constant_hop_timebase(hop, end_time):
    """Generates a time series from 0 to ``end_time`` with times spaced ``hop``
    apart

    Parameters
    ----------
    hop : float
        Spacing of samples in the time series
    end_time : float
        Time series will span ``[0, end_time]``

    Returns
    -------
    times : np.ndarray
        Generated timebase

    """
    # Compute new timebase.  Rounding/linspace is to avoid float problems.
    end_time = np.round(end_time, 10)
    times = np.linspace(0, hop*int(np.floor(end_time/hop)),
                        int(np.floor(end_time/hop)) + 1)
    times = np.round(times, 10)
    return times


def resample_melody_series(times, frequencies, voicing,
                           times_new, kind='linear'):
    """Resamples frequency and voicing time series to a new timescale. Maintains
    any zero ("unvoiced") values in frequencies.

    If ``times`` and ``times_new`` are equivalent, no resampling will be
    performed.

    Parameters
    ----------
    times : np.ndarray
        Times of each frequency value
    frequencies : np.ndarray
        Array of frequency values, >= 0
    voicing : np.ndarray
        Boolean array which indicates voiced or unvoiced
    times_new : np.ndarray
        Times to resample frequency and voicing sequences to
    kind : str
        kind parameter to pass to scipy.interpolate.interp1d.
        (Default value = 'linear')

    Returns
    -------
    frequencies_resampled : np.ndarray
        Frequency array resampled to new timebase
    voicing_resampled : np.ndarray, dtype=bool
        Boolean voicing array resampled to new timebase

    """
    # If the timebases are already the same, no need to interpolate
    if times.shape == times_new.shape and np.allclose(times, times_new):
        return frequencies, voicing.astype(np.bool)

    # Warn when the delta between the original times is not constant,
    # unless times[0] == 0. and frequencies[0] == frequencies[1] (see logic at
    # the beginning of to_cent_voicing)
    if not (np.allclose(np.diff(times), np.diff(times).mean()) or
            (np.allclose(np.diff(times[1:]), np.diff(times[1:]).mean()) and
             frequencies[0] == frequencies[1])):
        warnings.warn(
            "Non-uniform timescale passed to resample_melody_series.  Pitch "
            "will be linearly interpolated, which will result in undesirable "
            "behavior if silences are indicated by missing values.  Silences "
            "should be indicated by nonpositive frequency values.")
    # Round to avoid floating point problems
    times = np.round(times, 10)
    times_new = np.round(times_new, 10)
    # Add in an additional sample if we'll be asking for a time too large
    if times_new.max() > times.max():
        times = np.append(times, times_new.max())
        frequencies = np.append(frequencies, 0)
        voicing = np.append(voicing, 0)
    # We need to fix zero transitions if interpolation is not zero or nearest
    if kind != 'zero' and kind != 'nearest':
        # Fill in zero values with the last reported frequency
        # to avoid erroneous values when resampling
        frequencies_held = np.array(frequencies)
        for n, frequency in enumerate(frequencies[1:]):
            if frequency == 0:
                frequencies_held[n + 1] = frequencies_held[n]
        # Linearly interpolate frequencies
        frequencies_resampled = scipy.interpolate.interp1d(times,
                                                           frequencies_held,
                                                           kind)(times_new)
        # Retain zeros
        frequency_mask = scipy.interpolate.interp1d(times,
                                                    frequencies,
                                                    'zero')(times_new)
        frequencies_resampled *= (frequency_mask != 0)
    else:
        frequencies_resampled = scipy.interpolate.interp1d(times,
                                                           frequencies,
                                                           kind)(times_new)
    # Use nearest-neighbor for voicing if it was used for frequencies
    if kind == 'nearest':
        voicing_resampled = scipy.interpolate.interp1d(times,
                                                       voicing,
                                                       kind)(times_new)
    # otherwise, always use zeroth order
    else:
        voicing_resampled = scipy.interpolate.interp1d(times,
                                                       voicing,
                                                       'zero')(times_new)
    return frequencies_resampled, voicing_resampled.astype(np.bool)


def to_cent_voicing(ref_time, ref_freq, est_time, est_freq, base_frequency=10.,
                    hop=None, kind='linear'):
    """Converts reference and estimated time/frequency (Hz) annotations to sampled
    frequency (cent)/voicing arrays.

    A zero frequency indicates "unvoiced".

    A negative frequency indicates "Predicted as unvoiced, but if it's voiced,
    this is the frequency estimate".

    Parameters
    ----------
    ref_time : np.ndarray
        Time of each reference frequency value
    ref_freq : np.ndarray
        Array of reference frequency values
    est_time : np.ndarray
        Time of each estimated frequency value
    est_freq : np.ndarray
        Array of estimated frequency values
    base_frequency : float
        Base frequency in Hz for conversion to cents
        (Default value = 10.)
    hop : float
        Hop size, in seconds, to resample,
        default None which means use ref_time
    kind : str
        kind parameter to pass to scipy.interpolate.interp1d.
        (Default value = 'linear')

    Returns
    -------
    ref_voicing : np.ndarray, dtype=bool
        Resampled reference boolean voicing array
    ref_cent : np.ndarray
        Resampled reference frequency (cent) array
    est_voicing : np.ndarray, dtype=bool
        Resampled estimated boolean voicing array
    est_cent : np.ndarray
        Resampled estimated frequency (cent) array

    """
    # Check if missing sample at time 0 and if so add one
    if ref_time[0] > 0:
        ref_time = np.insert(ref_time, 0, 0)
        ref_freq = np.insert(ref_freq, 0, ref_freq[0])
    if est_time[0] > 0:
        est_time = np.insert(est_time, 0, 0)
        est_freq = np.insert(est_freq, 0, est_freq[0])
    # Get separated frequency array and voicing boolean array
    ref_freq, ref_voicing = freq_to_voicing(ref_freq)
    est_freq, est_voicing = freq_to_voicing(est_freq)
    # convert both sequences to cents
    ref_cent = hz2cents(ref_freq, base_frequency)
    est_cent = hz2cents(est_freq, base_frequency)
    # If we received a hop, use it to resample both
    if hop is not None:
        # Resample to common time base
        ref_cent, ref_voicing = resample_melody_series(
            ref_time, ref_cent, ref_voicing,
            constant_hop_timebase(hop, ref_time.max()), kind)
        est_cent, est_voicing = resample_melody_series(
            est_time, est_cent, est_voicing,
            constant_hop_timebase(hop, est_time.max()), kind)
    # Otherwise, only resample estimated to the reference time base
    else:
        est_cent, est_voicing = resample_melody_series(
            est_time, est_cent, est_voicing, ref_time, kind)
    # ensure the estimated sequence is the same length as the reference
    len_diff = ref_cent.shape[0] - est_cent.shape[0]
    if len_diff >= 0:
        est_cent = np.append(est_cent, np.zeros(len_diff))
        est_voicing = np.append(est_voicing, np.zeros(len_diff))
    else:
        est_cent = est_cent[:ref_cent.shape[0]]
        est_voicing = est_voicing[:ref_voicing.shape[0]]

    return (ref_voicing.astype(bool), ref_cent,
            est_voicing.astype(bool), est_cent)


def voicing_measures(ref_voicing, est_voicing):
    """Compute the voicing recall and false alarm rates given two voicing
    indicator sequences, one as reference (truth) and the other as the estimate
    (prediction).  The sequences must be of the same length.

    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
    >>> (ref_v, ref_c,
    ...  est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,
    ...                                                  ref_freq,
    ...                                                  est_time,
    ...                                                  est_freq)
    >>> recall, false_alarm = mir_eval.melody.voicing_measures(ref_v,
    ...                                                        est_v)

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference boolean voicing array
    est_voicing : np.ndarray
        Estimated boolean voicing array

    Returns
    -------
    vx_recall : float
        Voicing recall rate, the fraction of voiced frames in ref
        indicated as voiced in est
    vx_false_alarm : float
        Voicing false alarm rate, the fraction of unvoiced frames in ref
        indicated as voiced in est

    """
    validate_voicing(ref_voicing, est_voicing)
    ref_voicing = ref_voicing.astype(bool)
    est_voicing = est_voicing.astype(bool)
    # When input arrays are empty, return 0 by special case
    if ref_voicing.size == 0 or est_voicing.size == 0:
        return 0.

    # How voicing is computed
    #        | ref_v | !ref_v |
    # -------|-------|--------|
    # est_v  |  TP   |   FP   |
    # -------|-------|------- |
    # !est_v |  FN   |   TN   |
    # -------------------------

    TP = (ref_voicing*est_voicing).sum()
    FP = ((ref_voicing == 0)*est_voicing).sum()
    FN = (ref_voicing*(est_voicing == 0)).sum()
    TN = ((ref_voicing == 0)*(est_voicing == 0)).sum()

    # Voicing recall = fraction of voiced frames according the reference that
    # are declared as voiced by the estimate
    if TP + FN == 0:
        vx_recall = 0.
    else:
        vx_recall = TP/float(TP + FN)

    # Voicing false alarm = fraction of unvoiced frames according to the
    # reference that are declared as voiced by the estimate
    if FP + TN == 0:
        vx_false_alm = 0.
    else:
        vx_false_alm = FP/float(FP + TN)

    return vx_recall, vx_false_alm


def raw_pitch_accuracy(ref_voicing, ref_cent, est_voicing, est_cent,
                       cent_tolerance=50):
    """Compute the raw pitch accuracy given two pitch (frequency) sequences in
    cents and matching voicing indicator sequences. The first pitch and voicing
    arrays are treated as the reference (truth), and the second two as the
    estimate (prediction).  All 4 sequences must be of the same length.

    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
    >>> (ref_v, ref_c,
    ...  est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,
    ...                                                  ref_freq,
    ...                                                  est_time,
    ...                                                  est_freq)
    >>> raw_pitch = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c,
    ...                                                est_v, est_c)

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference boolean voicing array
    ref_cent : np.ndarray
        Reference pitch sequence in cents
    est_voicing : np.ndarray
        Estimated boolean voicing array
    est_cent : np.ndarray
        Estimate pitch sequence in cents
    cent_tolerance : float
        Maximum absolute deviation for a cent value to be considerd correct
        (Default value = 50)

    Returns
    -------
    raw_pitch : float
        Raw pitch accuracy, the fraction of voiced frames in ref_cent for
        which est_cent provides a correct frequency values
        (within cent_tolerance cents).

    """

    validate_voicing(ref_voicing, est_voicing)
    validate(ref_voicing, ref_cent, est_voicing, est_cent)
    ref_voicing = ref_voicing.astype(bool)
    est_voicing = est_voicing.astype(bool)
    # When input arrays are empty, return 0 by special case
    if ref_voicing.size == 0 or est_voicing.size == 0 \
       or ref_cent.size == 0 or est_cent.size == 0:
        return 0.
    # If there are no voiced frames in reference, metric is 0
    if ref_voicing.sum() == 0:
        return 0.

    # Raw pitch = the number of voiced frames in the reference for which the
    # estimate provides a correct frequency value (within cent_tolerance cents)
    # NB: voicing estimation is ignored in this measure
    frame_correct = (np.abs(ref_cent - est_cent)[ref_voicing] < cent_tolerance)
    raw_pitch = (frame_correct).sum()/float(ref_voicing.sum())

    return raw_pitch


def raw_chroma_accuracy(ref_voicing, ref_cent, est_voicing, est_cent,
                        cent_tolerance=50):
    """Compute the raw chroma accuracy given two pitch (frequency) sequences
    in cents and matching voicing indicator sequences. The first pitch and
    voicing arrays are treated as the reference (truth), and the second two as
    the estimate (prediction).  All 4 sequences must be of the same length.


    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
    >>> (ref_v, ref_c,
    ...  est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,
    ...                                                  ref_freq,
    ...                                                  est_time,
    ...                                                  est_freq)
    >>> raw_chroma = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c,
    ...                                                  est_v, est_c)


    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference boolean voicing array
    ref_cent : np.ndarray
        Reference pitch sequence in cents
    est_voicing : np.ndarray
        Estimated boolean voicing array
    est_cent : np.ndarray
        Estimate pitch sequence in cents
    cent_tolerance : float
        Maximum absolute deviation for a cent value to be considered correct
        (Default value = 50)


    Returns
    -------
    raw_chroma : float
        Raw chroma accuracy, the fraction of voiced frames in ref_cent for
        which est_cent provides a correct frequency values (within
        cent_tolerance cents), ignoring octave errors


    References
    ----------
    .. [#] J. Salamon, E. Gomez, D. P. W. Ellis and G. Richard, "Melody
        Extraction from Polyphonic Music Signals: Approaches, Applications
        and Challenges", IEEE Signal Processing Magazine, 31(2):118-134,
        Mar. 2014.


    .. [#] G. E. Poliner, D. P. W. Ellis, A. F. Ehmann, E. Gomez, S.
        Streich, and B. Ong. "Melody transcription from music audio:
        Approaches and evaluation", IEEE Transactions on Audio, Speech, and
        Language Processing, 15(4):1247-1256, 2007.

    """
    validate_voicing(ref_voicing, est_voicing)
    validate(ref_voicing, ref_cent, est_voicing, est_cent)
    ref_voicing = ref_voicing.astype(bool)
    est_voicing = est_voicing.astype(bool)
    # When input arrays are empty, return 0 by special case
    if ref_voicing.size == 0 or est_voicing.size == 0 \
       or ref_cent.size == 0 or est_cent.size == 0:
        return 0.

    # If there are no voiced frames in reference, metric is 0
    if ref_voicing.sum() == 0:
        return 0.

    # Raw chroma = same as raw pitch except that octave errors are ignored.
    cent_diff = np.abs(ref_cent - est_cent)
    octave = 1200*np.floor(cent_diff/1200.0 + 0.5)
    frame_correct = (np.abs(cent_diff - octave)[ref_voicing] < cent_tolerance)
    n_voiced = float(ref_voicing.sum())
    raw_chroma = (frame_correct).sum()/n_voiced
    return raw_chroma


def overall_accuracy(ref_voicing, ref_cent, est_voicing, est_cent,
                     cent_tolerance=50):
    """Compute the overall accuracy given two pitch (frequency) sequences in cents
    and matching voicing indicator sequences. The first pitch and voicing
    arrays are treated as the reference (truth), and the second two as the
    estimate (prediction).  All 4 sequences must be of the same length.

    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
    >>> (ref_v, ref_c,
    ...  est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,
    ...                                                  ref_freq,
    ...                                                  est_time,
    ...                                                  est_freq)
    >>> overall_accuracy = mir_eval.melody.overall_accuracy(ref_v, ref_c,
    ...                                                     est_v, est_c)

    Parameters
    ----------
    ref_voicing : np.ndarray
        Reference boolean voicing array
    ref_cent : np.ndarray
        Reference pitch sequence in cents
    est_voicing : np.ndarray
        Estimated boolean voicing array
    est_cent : np.ndarray
        Estimate pitch sequence in cents
    cent_tolerance : float
        Maximum absolute deviation for a cent value to be considered correct
        (Default value = 50)

    Returns
    -------
    overall_accuracy : float
        Overall accuracy, the total fraction of correctly estimates frames,
        where provides a correct frequency values (within cent_tolerance
        cents).

    """
    validate_voicing(ref_voicing, est_voicing)
    validate(ref_voicing, ref_cent, est_voicing, est_cent)
    ref_voicing = ref_voicing.astype(bool)
    est_voicing = est_voicing.astype(bool)
    # When input arrays are empty, return 0 by special case
    if ref_voicing.size == 0 or est_voicing.size == 0 \
       or ref_cent.size == 0 or est_cent.size == 0:
        return 0.

    # True negatives = frames correctly estimates as unvoiced
    TN = ((ref_voicing == 0)*(est_voicing == 0)).sum()

    cent_diff = np.abs(ref_cent - est_cent)
    frame_correct = (cent_diff[ref_voicing*est_voicing] < cent_tolerance)
    accuracy = (frame_correct.sum() + TN)/float(ref_cent.shape[0])

    return accuracy


def evaluate(ref_time, ref_freq, est_time, est_freq, **kwargs):
    """Evaluate two melody (predominant f0) transcriptions, where the first is
    treated as the reference (ground truth) and the second as the estimate to
    be evaluated (prediction).

    Examples
    --------
    >>> ref_time, ref_freq = mir_eval.io.load_time_series('ref.txt')
    >>> est_time, est_freq = mir_eval.io.load_time_series('est.txt')
    >>> scores = mir_eval.melody.evaluate(ref_time, ref_freq,
    ...                                   est_time, est_freq)

    Parameters
    ----------
    ref_time : np.ndarray
        Time of each reference frequency value
    ref_freq : np.ndarray
        Array of reference frequency values
    est_time : np.ndarray
        Time of each estimated frequency value
    est_freq : np.ndarray
        Array of estimated frequency values
    kwargs
        Additional keyword arguments which will be passed to the
        appropriate metric or preprocessing functions.

    Returns
    -------
    scores : dict
        Dictionary of scores, where the key is the metric name (str) and
        the value is the (float) score achieved.

    """
    # Convert to reference/estimated voicing/frequency (cent) arrays
    (ref_voicing, ref_cent,
     est_voicing, est_cent) = util.filter_kwargs(
         to_cent_voicing, ref_time, ref_freq, est_time, est_freq, **kwargs)

    # Compute metrics
    scores = collections.OrderedDict()

    (scores['Voicing Recall'],
     scores['Voicing False Alarm']) = util.filter_kwargs(voicing_measures,
                                                         ref_voicing,
                                                         est_voicing, **kwargs)

    scores['Raw Pitch Accuracy'] = util.filter_kwargs(raw_pitch_accuracy,
                                                      ref_voicing, ref_cent,
                                                      est_voicing, est_cent,
                                                      **kwargs)

    scores['Raw Chroma Accuracy'] = util.filter_kwargs(raw_chroma_accuracy,
                                                       ref_voicing, ref_cent,
                                                       est_voicing, est_cent,
                                                       **kwargs)

    scores['Overall Accuracy'] = util.filter_kwargs(overall_accuracy,
                                                    ref_voicing, ref_cent,
                                                    est_voicing, est_cent,
                                                    **kwargs)
    return scores

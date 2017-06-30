'''
Methods which sonify annotations for "evaluation by ear".
All functions return a raw signal at the specified sampling rate.
'''

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import interp1d

from . import util
from . import chord


def clicks(times, fs, click=None, length=None):
    """Returns a signal with the signal 'click' placed at each specified time

    Parameters
    ----------
    times : np.ndarray
        times to place clicks, in seconds
    fs : int
        desired sampling rate of the output signal
    click : np.ndarray
        click signal, defaults to a 1 kHz blip
    length : int
        desired number of samples in the output signal,
        defaults to ``times.max()*fs + click.shape[0] + 1``

    Returns
    -------
    click_signal : np.ndarray
        Synthesized click signal

    """
    # Create default click signal
    if click is None:
        # 1 kHz tone, 100ms
        click = np.sin(2*np.pi*np.arange(fs*.1)*1000/(1.*fs))
        # Exponential decay
        click *= np.exp(-np.arange(fs*.1)/(fs*.01))
    # Set default length
    if length is None:
        length = int(times.max()*fs + click.shape[0] + 1)

    # Pre-allocate click signal
    click_signal = np.zeros(length)
    # Place clicks
    for time in times:
        # Compute the boundaries of the click
        start = int(time*fs)
        end = start + click.shape[0]
        # Make sure we don't try to output past the end of the signal
        if start >= length:
            break
        if end >= length:
            click_signal[start:] = click[:length - start]
            break
        # Normally, just add a click here
        click_signal[start:end] = click
    return click_signal


def time_frequency(gram, frequencies, times, fs, function=np.sin, length=None):
    """Reverse synthesis of a time-frequency representation of a signal

    Parameters
    ----------
    gram : np.ndarray
        ``gram[n, m]`` is the magnitude of ``frequencies[n]``
        from ``times[m]`` to ``times[m + 1]``

        Non-positive magnitudes are interpreted as silence.

    frequencies : np.ndarray
        array of size ``gram.shape[0]`` denoting the frequency of
        each row of gram
    times : np.ndarray, shape= ``(gram.shape[1],)`` or ``(gram.shape[1], 2)``
        Either the start time of each column in the gram,
        or the time interval corresponding to each column.
    fs : int
        desired sampling rate of the output signal
    function : function
        function to use to synthesize notes, should be :math:`2\pi`-periodic
    length : int
        desired number of samples in the output signal,
        defaults to ``times[-1]*fs``

    Returns
    -------
    output : np.ndarray
        synthesized version of the piano roll

    """
    # Default value for length
    if times.ndim == 1:
        # Convert to intervals
        times = util.boundaries_to_intervals(times)

    if length is None:
        length = int(times[-1, 1] * fs)

    times, _ = util.adjust_intervals(times, t_max=length)

    # Truncate times so that the shape matches gram
    times = times[:gram.shape[1]]

    def _fast_synthesize(frequency):
        """A faster way to synthesize a signal.
            Generate one cycle, and simulate arbitrary repetitions
            using array indexing tricks.
        """
        frequency = float(frequency)

        # Generate ten periods at this frequency
        n_samples = int(10.0 * fs / frequency)

        short_signal = function(2.0 * np.pi * np.arange(n_samples) *
                                frequency / fs)

        # Calculate the number of loops we need to fill the duration
        n_repeats = int(np.ceil(length/float(short_signal.shape[0])))

        # Simulate tiling the short buffer by using stride tricks
        long_signal = as_strided(short_signal,
                                 shape=(n_repeats, len(short_signal)),
                                 strides=(0, short_signal.itemsize))

        # Use a flatiter to simulate a long 1D buffer
        return long_signal.flat

    # Threshold the tfgram to remove non-positive values
    gram = np.maximum(gram, 0)

    # Pre-allocate output signal
    output = np.zeros(length)
    for n, frequency in enumerate(frequencies):
        # Get a waveform of length samples at this frequency
        wave = _fast_synthesize(frequency)
        # Scale each time interval by the piano roll magnitude
        for m, (start, end) in enumerate((times * fs).astype(int)):
            # Clip the timings to make sure the indices are valid
            start, end = max(start, 0), min(end, length)

            # Sum into the aggregate output waveform
            output[start:end] += wave[start:end] * gram[n, m]

    # Normalize, but only if there's non-zero values
    norm = np.abs(output).max()
    if norm >= np.finfo(output.dtype).tiny:
        output /= norm

    return output


def pitch_contour(times, frequencies, fs, function=np.sin, length=None,
                  kind='linear'):
    '''Sonify a pitch contour.

    Parameters
    ----------
    times : np.ndarray
        time indices for each frequency measurement, in seconds

    frequencies : np.ndarray
        frequency measurements, in Hz.
        Non-positive measurements will be interpreted as un-voiced samples.

    fs : int
        desired sampling rate of the output signal

    function : function
        function to use to synthesize notes, should be :math:`2\pi`-periodic

    length : int
        desired number of samples in the output signal,
        defaults to ``max(times)*fs``

    kind : str
        Interpolation mode for the frequency estimator.
        See: ``scipy.interpolate.interp1d`` for valid settings.

    Returns
    -------
    output : np.ndarray
        synthesized version of the pitch contour
    '''

    fs = float(fs)

    if length is None:
        length = int(times.max() * fs)

    # Squash the negative frequencies.
    # wave(0) = 0, so clipping here will un-voice the corresponding instants
    frequencies = np.maximum(frequencies, 0.0)

    # Build a frequency interpolator
    f_interp = interp1d(times * fs, 2 * np.pi * frequencies / fs, kind=kind,
                        fill_value=0.0, bounds_error=False, copy=False)

    # Estimate frequency at sample points
    f_est = f_interp(np.arange(length))

    # Sonify the waveform
    return function(np.cumsum(f_est))


def chroma(chromagram, times, fs, **kwargs):
    """Reverse synthesis of a chromagram (semitone matrix)

    Parameters
    ----------
    chromagram : np.ndarray, shape=(12, times.shape[0])
        Chromagram matrix, where each row represents a semitone [C->Bb]
        i.e., ``chromagram[3, j]`` is the magnitude of D# from ``times[j]`` to
        ``times[j + 1]``
    times: np.ndarray, shape=(len(chord_labels),) or (len(chord_labels), 2)
        Either the start time of each column in the chromagram,
        or the time interval corresponding to each column.
    fs : int
        Sampling rate to synthesize audio data at
    kwargs
        Additional keyword arguments to pass to
        :func:`mir_eval.sonify.time_frequency`

    Returns
    -------
    output : np.ndarray
        Synthesized chromagram

    """
    # We'll just use time_frequency with a Shepard tone-gram
    # To create the Shepard tone-gram, we copy the chromagram across 7 octaves
    n_octaves = 7
    # starting from C2
    base_note = 24
    # and weight each octave by a normal distribution
    # The normal distribution has mean 72 (one octave above middle C)
    # and std 6 (one half octave)
    mean = 72
    std = 6
    notes = np.arange(12*n_octaves) + base_note
    shepard_weight = np.exp(-(notes - mean)**2./(2.*std**2.))
    # Copy the chromagram matrix vertically n_octaves times
    gram = np.tile(chromagram.T, n_octaves).T
    # This fixes issues if the supplied chromagram is int type
    gram = gram.astype(float)
    # Apply Sheppard weighting
    gram *= shepard_weight.reshape(-1, 1)
    # Compute frequencies
    frequencies = 440.0*(2.0**((notes - 69)/12.0))
    return time_frequency(gram, frequencies, times, fs, **kwargs)


def chords(chord_labels, intervals, fs, **kwargs):
    """Synthesizes chord labels

    Parameters
    ----------
    chord_labels : list of str
        List of chord label strings.
    intervals : np.ndarray, shape=(len(chord_labels), 2)
        Start and end times of each chord label
    fs : int
        Sampling rate to synthesize at
    kwargs
        Additional keyword arguments to pass to
        :func:`mir_eval.sonify.time_frequency`

    Returns
    -------
    output : np.ndarray
        Synthesized chord labels

    """
    util.validate_intervals(intervals)

    # Convert from labels to chroma
    roots, interval_bitmaps, _ = chord.encode_many(chord_labels)
    chromagram = np.array([np.roll(interval_bitmap, root)
                           for (interval_bitmap, root)
                           in zip(interval_bitmaps, roots)]).T

    return chroma(chromagram, intervals, fs, **kwargs)

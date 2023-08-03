from __future__ import absolute_import
import logging
from typing import Union, Optional

import numpy as np
from scipy.signal import stft, istft

import pya.asig
from .helper import basicplot


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


# TODO, check with multichannel
class Astft:
    """Audio spectrogram (STFT) class, attributes refers to scipy.signal.stft. With an addition
        attribute cn being the list of channel names, and label being the name of the Asig
    """

    def __init__(self, x: Union[pya.Asig, np.ndarray], sr: Optional[int] = None,
                 window: str = 'hann', nperseg: int = 256, noverlap: Optional[int] = None,
                 nfft: Optional[int] = None, detrend: bool = False,
                 return_onesided: bool = True, boundary: str = 'zeros',
                 padded: bool = True, cn: Optional[list] = None):
        """__init__() method

        Parameters
        ----------
        x : Asig or numpy.ndarray
            x can be two forms, the most commonly used is an Asig object.
            Such as directly acquired from an Asig object via Asig.to_stft().
        sr : int, optional
            sampling rate, this is only necessary if x is not Asig.
            (Default value = None)
        window : str
            type of the window function (Default value = 'hann')
        nperseg : int
            number of samples per stft segment (Default value = '256')
        noverlap : int, optional
            number of samples to overlap between segments (Default value = None)
        nfft : int, optional
            Length of the FFT used, if a zero padded FFT is desired. If
            `None`, the FFT length is `nperseg`. Defaults to `None`.
        detrend : str or function or bool
            Specifies how to detrend each segment. If detrend is a string,
            it is passed as the type argument to the detrend function. If it is a function,
            it takes a segment and returns a detrended segment. If detrend is False,
            no detrending is done. (Default value = False).
        return_onesided : bool
            If True, return a one-sided spectrum for real data. If False return a two-sided spectrum.
            Defaults to True, but for complex data, a two-sided spectrum is always returned. (Default value = True)
        boundary : str or None
            Specifies whether the input signal is extended at both ends, and how to generate the new values,
            in order to center the first windowed segment on the first input point.
            This has the benefit of enabling reconstruction of the first input point
            when the employed window function starts at zero.
            Valid options are ['even', 'odd', 'constant', 'zeros', None]. Defaults to ‘zeros’,
            for zero padding extension. I.e. [1, 2, 3, 4] is extended to [0, 1, 2, 3, 4, 0] for nperseg=3. (Default value = 'zeros')
        padded : bool
            Specifies whether the input signal is zero-padded at the end to make the signal fit exactly into
            an integer number of window segments, so that all of the signal is included in the output.
            Defaults to True. Padding occurs after boundary extension, if boundary is not None, and padded is True,
            as is the default. (Default value = True)
        cn : list, optional
            Channel names of the Asig, this will be used for the Astft for consistency. (Default value = None)
        """
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.detrend = detrend
        self.return_onesided = return_onesided
        self.boundary = boundary
        self.padded = padded
        # self.cn = cn
        self.im = None  # buffer for the image

        if isinstance(x, pya.asig.Asig):
            self.sr = x.sr
            self.channels = x.channels
            self.label = x.label + "_stft"
            self.cn = x.cn
            self.samples = x.samples
            if sr:
                self.sr = sr  # explicitly given sr overwrites Asig
            sig = x.sig

        elif isinstance(x, np.ndarray):
            # x is a numpy array instead of asig.
            self.channels = x.ndim
            self.samples = len(x)
            self.label = 'stft'
            self.cn = []
            if sr is None:
                raise AttributeError("sr (sampling rate) is required as an argument if input is a numpy array rather than Asig.")
            else:
                self.sr = sr
            sig = x

        else:
            raise TypeError("Unknown data type x, x should be either Asig or numpy.ndarray")

        self.freqs, self.times, self.stft = \
            stft(sig, fs=self.sr, window=window, nperseg=nperseg,
                 noverlap=noverlap, nfft=nfft, detrend=detrend,
                 return_onesided=return_onesided, boundary=boundary,
                 padded=padded, axis=0)

        if cn:
            if len(cn) == self.channels:
                self.cn = cn
            else:
                raise AttributeError("Length of cn should equal channels.")

    def to_sig(self, **kwargs):
        """Create signal from stft, i.e. perform istft,
        kwargs overwrite Astft values for istft

        Parameters
        ----------
        **kwargs : str
            optional keyboard arguments used in istft:
                'sr', 'window', 'nperseg', 'noverlap',
                'nfft', 'input_onesided', 'boundary'.
            also convert 'sr' to 'fs' since scipy uses 'fs' as sampling frequency.

        Returns
        -------
        _ : Asig
            Asig
        """
        for k in ['sr', 'window', 'nperseg', 'noverlap',
                  'nfft', 'input_onesided', 'boundary']:
            if k in kwargs.keys():
                kwargs[k] = self.__getattribute__(k)
        if 'sr' in kwargs.keys():
            kwargs['fs'] = kwargs['sr']
            del kwargs['sr']
        if self.channels == 1:
            # _ since 1st return value 'times' unused
            _, sig = istft(self.stft, **kwargs)
            return pya.asig.Asig(sig, sr=self.sr,
                                 label=self.label + '_2sig', cn=self.cn)
        else:
            _, sig = istft(self.stft, **kwargs)
            return pya.asig.Asig(np.transpose(sig),
                                 sr=self.sr, label=self.label + '_2sig', cn=self.cn)

    def plot(self, fn=lambda x: x, ax=None,
             offset=0, scale=1., xlim=None, ylim=None,
             show_bar=True, **kwargs):
        """Plot spectrogram

        Parameters
        ----------
        fn : func
            a function, by default is bypass
        ch : int or str or None
            By default it is None,
        ax : matplotlib.axes
            you can assign your plot to specific axes (Default value = None)
        xlim : tuple or list
            x_axis range (Default value = None)
        ylim : tuple or list
            y_axis range (Default value = None)
        **kwargs :
            keyward arguments of matplotlib's pcolormesh

        Returns
        -------
        _ : Asig
            self
        """
        self.im, ax = basicplot(fn(np.abs(self.stft)), (self.times, self.freqs),
                                channels=self.channels,
                                cn=self.cn, offset=offset, scale=scale,
                                ax=ax, typ='spectrogram', show_bar=show_bar,
                                xlabel='time', xlim=xlim, ylim=ylim, **kwargs)
        ax.set_ylabel('freq')
        return self

    def __repr__(self):
        return "Astft('{}'): {} x {} @ {} Hz = {:.3f} s cn={}".format(
            self.label, self.channels, self.samples,
            self.sr, self.samples / self.sr, self.cn)

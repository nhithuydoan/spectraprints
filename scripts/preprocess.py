"""This script preprocesses EDF files prior to estimation of the PSD. 

The Xue lab data was acquired at 5 kHz and contains neural and muscular data
channels....This module takes care of ...
"""

import pickle
import time
import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from openseize import producer
from openseize.file_io import edf
from openseize.filtering import iir
from openseize.resampling import resampling

from spectraprints.core import concurrency
from spectraprints.masking import masks


def preprocess(epath, savedir, channels=[0,1,2], fs=5000, M=20, fstop=60,
               stop_width=6, nominally=[48, 72], chunksize=30e5, axis=-1):
    """Preprocesses and EDF file at path by constructing, notch filtering and
    downsampling a producer.

    Args:
        epath:
            Path to an edf file.
        channels:
            The channels to include in the processed producer.
        fs:
            The sampling rate at which the data was collected.
        M:
            The downsample factor to reduce the number of samples produced by
            the returned processed producer.
        start:
            The start index at which production begins. If None start is 0.
        stop:
            The stop index at which production stops. If None stop is data
            length along axis.
        chunksize:
            The number of processed samples along axis to yield from the 
            producer at a time.
        axis:
            The axis of reading and production.

    Returns:
        A producer of preprocessed values.
    """

    # TODO remove
    t0 = time.perf_counter()

    epath = Path(epath)
    reader = edf.Reader(epath)
    reader.channels = channels

    # compute stop index to trim the readers data to one of nominally
    # FIXME what if 68 hours -> we should use 48 not 72 
    n = np.array(nominally) * 3600 * fs
    stop = n[np.argmin(np.abs(n - reader.shape[axis]))]
    if np.all(stop < nominally):
        name = epath.stem[0:20] + '...'
        msg = (f'The duration of file {name} is {reader.shape / (3600*fs)}' 
               f' hours which is < expected = {nominally}')
        warnings.warn(msg)

    # build producer
    pro = masks.between_pro(reader, 0, stop, chunksize, axis)

    # Notch filter the producer
    notch = iir.Notch(fstop, width=stop_width, fs=fs)
    result = notch(pro, chunksize, axis, dephase=False)

    # downsample the producer
    result = resampling.downsample(result, M, fs, chunksize, axis)
    
    # in-memory compute and write
    # TODO this will change once openseize accepts producers to write from
    x = result.to_array()

    print(x.shape)

    # create a header with new samples_per_record
    new_header = reader.header.filter(channels)
    new_header['samples_per_record'] = [fs // M for _ in channels]
    
    # FIXME we've trimmed samples to 48 or 72 hours so num_records is trimmed
    # but we still need to ensure whole number of records. This may require
    # padding x or trimming to the nearest number of records rather than exact
    # hours
    new_header['num_records'] = int(x.shape[axis] / (fs/M))

    print(new_header)
    
    fname = epath.stem + '_downsampled' + epath.suffix
    filepath = Path(savedir).joinpath(fname)
    with edf.Writer(filepath) as writer:
        writer.write(new_header, x, channels=channels)

    print(f'Preprocessing finished in {time.perf_counter() - t0} secs')
    reader.close()
    return x


def batch(dirpath, savedir, ncores=None, **kwargs):
    """ """
    
    t0 = time.perf_counter()

    epaths = list(Path(dirpath).glob('*edf'))
    
    func = partial(preprocess, **kwargs)
    workers = concurrency.set_cores(ncores, len(epaths))
    with Pool(workers) as pool:
        pool.map(f, epaths)

    elapsed = time.perf_counter() - t0
    msg = f'Saved {len(epaths)} files to {savedir} in {elapsed} s'
    print(msg)

if __name__ == '__main__':

    basepath  = '/media/matt/Zeus/jasmine/stxbp1/'
    name = 'CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11.edf'

    epath = Path(basepath).joinpath(name)

    x = preprocess(epath, savedir='/media/matt/Zeus/sandy/test/')

    fp = ('/media/matt/Zeus/sandy/test/'
         'CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11_downsampled.edf')
    reader = edf.Reader(fp)
    y = reader.read(start=0)

    print(np.allclose(x, y, atol=0.5))

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
               fwidth=6, nominally=[48, 72], chunksize=30e5, axis=-1,
               returning=False, verbose=False):
    """Preprocesses and EDF file at path by constructing, notch filtering and
    downsampling a producer.

    Args:
        epath:
            Path to an edf file.
        savedir:
            Directory location where preprocessed EDF will be saved to. The
            filename to save will be named epath.stem + '_preprocessed'. 
        channels:
            The channels to include in the processed producer.
        fs:
            The sampling rate at which the data was collected.
        M:
            The downsample factor to reduce the number of samples produced by
            the returned processed producer.
        fstop:
            The stop frequency of the preprocessing Notch IIR filter.
        fwidth:
            The width of the preprocessing Notch IIR filter.
        nominally:
            A sequence of expected lengths. The preprocessed data will be
            trimmed to match the nearest floor of this array. Example if epath
            data is 66 hours long it will be trimmed to 48 hours.
        chunksize:
            The number of processed samples along axis to yield from the 
            producer at a time. When called in a multiprocessing context this
            value should be set to the amount of samples any single process uses
            at one time.
        axis:
            The axis of reading and production.
        returning:
            Boolean indicating if preprocessed results should be returned.
        verbose:
            Boolean indicating if timing information should be printed to
            stdout.

    Returns:
        None if not returning else returns preprocessed data array.
    """

    t0 = time.perf_counter()

    epath = Path(epath)
    reader = edf.Reader(epath)
    reader.channels = channels

    # Trim data with floor chosen from nominals
    nsamples = np.array(nominally) * 3600 * fs
    idx = np.searchsorted(nsamples, reader.shape[axis], side='right') - 1
    if idx < 0:
        hrs = reader.shape[axis] / (3600 * fs)
        msg = (f'File {epath.stem} is {hrs} hrs, lower than all expected'
               f' durations {nominally} hrs.')
        raise ValueError(msg)
    stop = nsamples[idx]

    # ensure start to stop is evenly divisible by samples per record = fs/M
    # if stop ~ k * fs then stop/M ~ k * fs / M where k is integer
    stop = int(np.ceil(stop / fs) * fs)

    # build producer
    pro = masks.between_pro(reader, 0, stop, chunksize, axis)

    # Notch filter and downsample the producer
    notch = iir.Notch(fstop, width=fwidth, fs=fs)
    result = notch(pro, chunksize, axis, dephase=False)
    result = resampling.downsample(result, M, fs, chunksize, axis)
    
    # in-memory compute & write, FIXME allow writers to write pros
    x = result.to_array()

    # update headers samples_per_record & num_records fields
    header = reader.header.filter(channels)
    header['samples_per_record'] = [spr // M 
            for spr in reader.header['samples_per_record']]
    header['num_records'] = int(x.shape[axis] / (fs / M))

    # write the file
    fname = epath.stem + '_preprocessed' + epath.suffix
    filepath = Path(savedir).joinpath(fname)
    with edf.Writer(filepath) as writer:
        writer.write(header, x, channels=channels, verbose=False)

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f'Preprocessing & file writing finished in {elapsed} secs')
    
    # clean up reader resources & possibly return
    reader.close()
    if returning:
        return x


def batch(dirpath, savedir, ncores=None, verbose=True, **kwargs):
    """Preprocesses all EDF files in dir path saving the preprocessed data to
    savedir. 

    Args:
        dirpath:
            A directory containing EDF files to preprocess.
        savedir:
            A directory to where preprocessed data will be save in EDF format.
        ncores:
            The number of processing cores to concurrently preprocess EDF files
            in dirpath.
        **kwargs:
            Any valid kwarg for the preprocess function of this module.

    Returns:
        None
    """
    
    epaths = list(Path(dirpath).glob('*edf'))
    
    t0 = time.perf_counter()
    if verbose:
        msg = (f'Executing Batch Preprocessor for {len(epaths)} files'
               f' in {dirpath}')
        print(msg)
    
    # returns are not allowed in batch processing
    kwargs.pop('returning', False)
    
    func = partial(preprocess, savedir=savedir, **kwargs)
    workers = concurrency.set_cores(ncores, len(epaths))
    with Pool(workers) as pool:
        pool.map(func, epaths)

    elapsed = time.perf_counter() - t0
    msg = f'Saved {len(epaths)} files to {savedir} in {elapsed} s'
    print(msg)



if __name__ == '__main__':

    basepath = '/media/matt/Zeus/jasmine/test/'
    
    """
    name = 'CW0DI3_P097_KO_93_31_3dayEEG_2020-05-07_09_54_14.edf'
    epath = Path(basepath).joinpath(name)

    x = preprocess(epath, savedir='/media/matt/Zeus/sandy/test/', verbose=True)

    savepath = '/media/matt/Zeus/sandy/test/' + '_processed' + '.edf'
    reader = edf.Reader(savepath)
    y = reader.read(start=0)

    print(np.allclose(x, y, atol=0.5))
    """

    savedir = '/media/matt/Zeus/sandy/test/'
    batch(basepath, savedir)

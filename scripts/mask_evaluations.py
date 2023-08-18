import time

from openseize import producer
from openseize import file_io
from openseize.filtering import iir
from openseize.resampling import resampling

from spectraprints.masking import masks
from spectraprints.core.metastores import MetaMask


def preprocess(epath, channels, fs, M, start, stop, chunksize=30e5, axis=-1):
    """Preprocesses an EEG file at path by constructing, notch filtering and
    downsampling a producer.

    Args:
        epath:
            Path to an edf file.
        channels:
            The channels to include in the processed producer.
        fs:
            The sampling rate at which the data at path was collected.
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

    reader = file_io.edf.Reader(epath)
    reader.channels = channels

    start = 0 if not start else start
    stop = reader.shape[axis] if not stop else stop
    if stop - start < reader.shape[axis]:
        pro = masks.between_pro(reader, start, stop, chunksize, axis)

    # Notch filter the producer
    notch = iir.Notch(fstop=60, width=6, fs=fs)
    result = notch(pro, chunksize, axis, dephase=False)

    # downsample the producer
    result = resampling.downsample(result, M, fs, chunksize, axis)
    return result

def make_metamask(epath, apath, spath):
    """ """

    t0 = time.perf_counter()
    # FIXME need to fix parameters here and explain choices!
    
    with file_io.annotations.Pinnacle(apath, start=6) as reader:
        annotes = reader.read()

    # preprocess a producer for between 'Start' & 'Stop' annotes
    a, b = [int(a.time * 5000) for a in annotes if a.label in ['Start', 'Stop']]
    pro = preprocess(epath, channels=[0,1,2], fs=5000, M=20, start=a, stop=b,
                     chunksize=30e5, axis=-1)

    # build threshold masks
    nstds=[3,4,5,6]
    thresholded = masks.threshold(pro, nstds=nstds, winsize=1.5e4,
                                  radius=125)
    thresholded_names = [f'threshold={std}' for std in nstds]
    thresholded = dict(zip(thresholded_names, thresholded))
    
    # build annotation mask
    # TODO should work out consistency in labels
    # may need more flexibility in artifact mask like regex
    annotated = masks.artifact(apath, size=pro.shape[pro.axis], 
        labels=['Artifact', 'Artifact ','water_drinking','water_drinking '],
        fs=250, between=['Start', 'Stop'])
    
    # build state masks
    stated = [masks.state(spath, ls, fs=250, winsize=4) 
              for ls in [['w'], ['r', 'n']]]
    state_names = ['awake', 'sleep']
    stated = dict(zip(state_names, stated))

    print(f'MetaMask made in {time.perf_counter() - t0} s')

    return MetaMask(annotated=annotated, **thresholded, **stated)


if __name__ == '__main__':

    from pathlib import Path

    basepath = Path('/media/matt/Zeus/jasmine/stxbp1')

    efile = 'CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11.edf'
    afile = 'CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11.txt'
    sfile = 'CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11_sleep_states.csv'

    epath, apath, spath = [basepath.joinpath(x) for x in [efile, afile, sfile]]

    metamask = make_metamask(epath, apath, spath)




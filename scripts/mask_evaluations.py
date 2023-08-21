import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from openseize import producer
from openseize import file_io
from openseize.filtering import iir
from openseize.resampling import resampling

from spectraprints.masking import masks, metrics
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

def make_metamask(epath, apath, spath, verbose=False):
    """Returns a MetaMask instance containing thresholded masks, annotations
    mask, and Spindle sleep score masks.

    Fixed Args:
        - produced data
            channels:
                The channels to analyze = [0, 1, 2]
            start:
                The start index of production. This should match the start label
                of the annotations and align with the first detected state in
                the spindle file. This index should be obtained by reading the
                start annotation and converting from time to samples.
            stop:
                The stop index of production. This shuld match the stop label of
                the annotations file and align with the last detected state in
                the spindle file. This index should be obtained by reading the
                stop annotation and converting from time to samples.
            fs: 
                The sampling rate of the data = 5000
            M:
                The downsampling ratio = 20
            chunksize:
                The size of produced data from preprocessor = 30e5
            axis:
                The production axis = -1
        - threshold masks
            nstds:
                The standard deviations to use as the threshold. One mask per
                threshold will be stored to MetaMask instance = [3, 4, 5, 6]
            winsize:
                The window size in samples to evaluate the mean & standard
                deviation for thresholding = 1.
            radius:
                The max distance between two detected events below which all
                intervening samples will be marked as artifact.
        - annotated masks
            size:
                The size of the of masks - this must equal the producers shape
                along the production axis.
            labels:
                The labels that denote artifacts = ['Artifact', 'Artifact ',
                'water_drinking', 'water_drinking ']
            fs:
                The downsampled sampling rate, must match fs // M = 250
            between:
                The string names of the start and stop annotation. These should
                correspond to the start and stop indices provided to producer.
        - state masks:
            labels:
                A list of list of string labels to include in each mask, one 
                sublist per state = [['w'], ['r', 'n']] gives the awake and
                sleep states.
            fs:
                The downsampled sampling rate, must match fs // M = 250
            winsize:
                The interval in secs over which spindle evaluated state
                = 4 secs.
    """

    t0 = time.perf_counter()
   
    # production Args
    CHS = [0, 1, 2]
    FS = 5000
    M = 20
    CSIZE = 30E5 # LOW ENOUGH TO ALLOW MULTIPROCESSING TO COPY
    AXIS = -1

    # thresholding Args
    NSTDS = [3, 4, 5, 6]
    WINSIZE = 1.5E4 # @ FS = 250 THIS IS 60 SECS OF DATA
    RADIUS = 125 # THIS IS IN SAMPLES

    # annotation Args
    # TODO - need more flexibility in artifact mask like regex
    LABELS = ['Artifact', 'Artifact ','water_drinking','water_drinking ']
    DFS = FS // M
    BETWEEN = ['Start', 'Stop']

    # State Args
    STATE_LABELS = [['w'], ['r', 'n']]
    STATE_WINSIZE = 4

    with file_io.annotations.Pinnacle(apath, start=6) as reader:
        annotes = reader.read()

    # preprocess a producer for between 'Start' & 'Stop' annotes
    a, b = [int(a.time * FS) for a in annotes if a.label in BETWEEN]
    pro = preprocess(epath, channels=CHS, fs=FS, M=M, start=a, stop=b,
                     chunksize=CSIZE, axis=AXIS)

    # build threshold masks
    thresholded = masks.threshold(pro, nstds=NSTDS, winsize=WINSIZE, 
                                  radius=RADIUS)
    thresholded_names = [f'threshold={std}' for std in NSTDS]
    thresholded = dict(zip(thresholded_names, thresholded))
    
    # build annotation mask
    annotated = masks.artifact(apath, pro.shape[pro.axis], LABELS, DFS, BETWEEN)
    
    # build state masks
    stated = [masks.state(spath, ls, DFS, STATE_WINSIZE) for ls in STATE_LABELS]
    state_names = ['awake', 'sleep']
    stated = dict(zip(state_names, stated))

    if verbose:
        name = Path(epath).stem
        print(f' Completed building MetaMask for {name}'
               ' in {time.perf_counter() - t0} s')

    return MetaMask(annotated=annotated, **thresholded, **stated)


def evaluate_one(epath, apath, spath):
    """Evaluates threshold masks built from one EEG, annotation & spindle file.

    This function evaluates the masks defined by the fixed arguments in
    make_metamask. It computes the accuracies, sensitivities, specificities,
    precisions and percent_withins for each std in NSTDS.

    """

    accuracies = {}
    sensitivities = {}
    specificities = {}
    precisions = {}
    perc_withins = {}

    metamask = make_metamask(epath, apath, spath)
    
    # actual & detected are name, mask tuple and list of name, mask tuples
    _, ann_mask = metamask('awake', 'annotated')
    detected = [metamask('awake', attr) for attr, _ in metamask.__dict__.items()
            if 'threshold' in attr]

    for name, mask in detected:

        # compare metrics of Falses
        tp = metrics._tp(~mask, ~ann_mask)
        tn = metrics._tn(~mask, ~ann_mask)
        fp = metrics._fp(~mask, ~ann_mask)
        fn = metrics._fn(~mask, ~ann_mask) 

        accuracies[name] = metrics.accuracy(tp, tn, fp, fn)
        sensitivities[name] = metrics.sensitivity(tp, fn)
        specificities[name] = metrics.specificity(tn, fp)
        precisions[name] = metrics.precision(tp, fp)

        # percent withins
        perc_withins[name] = metrics.percent_within(mask, ann_mask)

    return accuracies, sensitivities, specificities, precisions, perc_withins


def evaluate(dirpaths, save_path, ncores=None):
    """ """

    t0 = time.perf_counter()

    # 1 and 2 should be in single_evaluate func
    # 1. in each process build a mask
    # 2. compute tp, tn, fp, fn and all metrics
    #
    # store metrics to dict and save


    pass


if __name__ == '__main__':

    from pathlib import Path

    basepath = Path('/media/matt/Zeus/jasmine/stxbp1')

    efile = 'CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11.edf'
    afile = 'CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11.txt'
    sfile = 'CW0DI2_P097_KO_92_30_3dayEEG_2020-05-07_09_54_11_sleep_states.csv'

    epath, apath, spath = [basepath.joinpath(x) for x in [efile, afile, sfile]]

    acc, sens, spec, prec, pwithin = evaluate_one(epath, apath, spath)




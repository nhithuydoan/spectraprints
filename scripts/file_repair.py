""""This module corrects files that are segregated over multiple files due to
recording problems such as power-outages. It joins the files together into
a single EDF."""

import copy
import re
from pathlib import Path
import numpy as np

from openseize.file_io import edf

#TODO
def validate_lengths(dirpath, minimum=24):
    """Validates that all files in dirpath exceed minimum number of hours."""

    pass

def locate(dirpath, fs, expected=72, minimum=24):
    """Locates EDF files in dirpath whose duration in hours is less than
    expected.

    Args:
        dirpath:
            A directory containing EDF files.
        fs:
            The sampling rate each EDF was recorded at assumed to be equivalent
            across all files in dirpath.
        expected:
            The number of hours expected in each file. Files below this are
            considered 'short'.

    Returns:
        A list of files whose number of recordings hours is below expected.
    """

    result = []
    paths = list(Path(dirpath).glob('*.edf'))
    for fp in paths:
        reader = edf.Reader(fp)
        if reader.shape[-1]/(3600*fs) < minimum:
            msg = f'This file -- {fp} -- is shorter than 24 hours!!'
            raise ValueError(msg)
        if reader.shape[-1] / (3600 * fs) < expected:
            result.append(reader.path)
        reader.close()
    return result

def pair_paths(paths, pattern=r'[^_]+'):
    """Returns tuples of paths from  list of paths both of which contain the
    same pattern.

    Args:
        paths:
            A sequence of path instances.
        pattern:
            A valid regex pattern to use for matching.
    """

    result = []
    while paths:
        # remove first path & get mouse_id from path
        path = paths.pop(0)
        mouse_id = re.search(pattern, path.stem).group()
        # find other path with matching mouse_id & remove it
        matching = [other for other in paths if mouse_id in other.stem][0]

        if not matching:
            msg = f'No match find for mouse_id {mouse_id}'
            raise ValueError(msg)

        paths.remove(matching)
        # add path and matching path to results
        result.append(sorted([path, matching]))

    return result

def combine(path, other, time, fs, save_dir=None):
    """Read n_hours of data from the start of each EDF file in path and other
    and writes the combined data to a new EDF file in save_dir.

    Args:
        path:
            A path to an edf file
        other:
            A path to another edf file
        time:
            The number of hours to take from the start of each EDF file to write
            to the new combined EDF file.
        fs:
            The sampling rate of the EDF data located at path and other.
        save_dir:
            An optional directory to write the combined EDF file to. If None,
            the combined file will be written to the same dir as path.
        fs:
            The sampling rate of the files, assuming all files are sampled at the 
            same rate.

    Returns:
        None
    """

    # FIXME Sort path and other
    readers = [edf.Reader(fp) for fp in (path, other)]
    arrs = [reader.read(0, time * 3600 * fs) for reader in readers]
    arrs = np.concatenate(arrs, axis=-1)

    header = copy.deepcopy(readers[0].header)
    header['num_records'] = int(data.shape[-1] / header.samples_per_record[0])

    new_name = path.stem + '_COMBINED'
    parent = path.parent
    if save_dir:
        parent = save_dir
    write_path = Path(parent).joinpath(new_name).with_suffix(path.suffix)

    with edf.Writer(write_path) as writer:
        writer.write(header, data, channels=[0,1,2,3])



if __name__ == '__main__':


    dirpath = '/media/sandy/Data_A/sandy/STXBP1_High_Dose_Exps_3/short_files/'

    shorts = locate(dirpath, fs=5000)
    paired = pair_paths(shorts)
    for tup in range(len(paired)):
        path, other = paired[tup]
        print([path, other])
       # header = combine(path, other, time=24, fs=5000, save_dir=None)


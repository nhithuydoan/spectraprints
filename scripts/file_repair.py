""""This module corrects files that are segregated over multiple files due to
recording problems such as power-outages. It joins the files together into
a single EDF."""

from pathlib import Path
import re

from openseize.file_io import edf

def locate(dirpath, fs, expected=72):
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
        result.append((path, matching))

    return result

def combine(ls, hour, fs):
    """ This function takes files in list and produce a new edf file with the
    desired length.    
    Args:
        ls: 
            A list contains all edf files.
        hour:
            The desired length for each edf.
        fs:
            The sampling rate of the files, assuming all files are sampled at the 
            same rate.
    """
    stop_sample = hour*fs*3600
    for animal in ls:
        all_arr = []
        for i in animal:
            file = edf.Reader(i)
            arr = file.read(0, stop_sample)
            all_arr.append(arr)
        new_file = np.concatenate(all_arr, axis = 0)

    # sandy TODO
    # 1. make a reader for path and other
    # 2. read 24 hours from each into arrays x and y
    # 3. concatenate x and y
    # 4. what should be changed in the header prior to writing
    # 5 make a filename
    # 5. explore edf.Writer





if __name__ == '__main__':

    dirpath = '/media/matt/Zeus/STXBP1_High_Dose_Exps_3/'

    shorts = locate(dirpath, fs=5000)
    paired = pair_paths(shorts)


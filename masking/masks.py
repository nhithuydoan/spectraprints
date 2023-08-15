from pathlib import Path
from openseize.file_io import annotations
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
from openseize import Producer


def threshold(pro: Producer,
              nstds: List[float],
              winsize: int,
              radius: Optional[int] = None,
) -> List[npt.NDArray[np.bool_]]:
    """ """

    pro.chunksize = winsize
    axis = pro.axis

    # intialize some mask
    masks = [np.ones(pro.shape[axis], dtype=bool) for std in nstds] 
    for idx, arr in enumerate(pro):

        mu = np.mean(arr, axis=axis, keepdims=True)
        std = np.std(arr, axis=axis, keepdims=True)

        if not np.any(std):
            std = np.ones_like(std)

        arr -= mu
        arr /= std

        for sigma, mask in zip(nstds, masks):
            _, cols = np.where(np.abs(arr) > sigma)
            cols =np.unique(cols + idx * winsize)
            mask[cols] = False

        # TODO add merging 8/16/2023


def artifact(path: Union[str, Path],
             size: int,
             labels: List[str],
             fs: float,
             between: Tuple[Union[str, None], Union[str, None]]=(None, None),
             **kwargs,
) -> npt.NDArray[np.bool_]:
    """Returns a boolean mask from a Pinnacle annotations file.

    Args:
        path:
            Pinnacle file path containing annotations.
        size:
            The length of the mask to return.
        labels:
            The artifact labels that are to be marked False in the returned
            boolean.
        fs:
            The sampling rate in Hertz of the data acquisition.
        between:
            The start and stop annotation labels between which labeled
            annotations will be included in the mask.
        kwargs:
            Keyword arguments are passed to openseize's Pinnacle initializer.

    Examples:
        >>> from openseize import demos
        >>> import numpy as np
        >>> path = demos.paths.locate('annotations_001.txt')
        >>> fs = 5000
        >>> size = 3775 * 5000
        >>> amask = artifact(path, size=size, labels=['rest'], fs=fs)
        >>> # validate 1st set of Falses in mask occurs at 1st annote
        >>> with annotations.Pinnacle(path, start=6) as reader:
        ...     annotes = reader.read('rest')
        >>> start, duration = annotes[0].time, annotes[0].duration
        >>> np.any(amask[int(start*5000): int((start+duration) * 5000)])
        False

    Returns:
        A 1-D boolean mask in which labeled indices are marked False.
    """

    start = kwargs.pop('start', 6)
    with annotations.Pinnacle(path, start=start, **kwargs) as reader:
        annotes = reader.read()

    # get first and last annote to return that are in between
    a, b = between
    first = next(ann for ann in annotes if ann.label == a) if a else annotes[0]
    last = next(ann for ann in annotes if ann.label == b) if a else annotes[-1]

    #filter the annotes for requested labels and filter between 
    annotes = [ann for ann in annotes if ann.label in labels]
    annotes = [ann for ann in annotes if first.time <= ann.time <= last.time]
    # adjust the annote times relative to first annote
    for annote in annotes:
        annote.time -= first.time

    return annotations.as_mask(annotes, size, fs, include=False)


if __name__ == '__main__':


    from openseize import demos

    path = demos.paths.locate('annotations_001.txt')

    size = 3775 * 5000
    fs = 5000
    amask = artifact(path, size=size, labels=['rest', 'grooming'], fs=fs)
                     


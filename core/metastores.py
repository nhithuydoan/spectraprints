import copy
import functools
import numbers
import warnings
from collections import abc
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from spectraprints.core import mixins


class MetaArray(mixins.ViewInstance):
    """A representation of a numpy NDArray containing both the array &
    coordinates, a dict of axes names & index labels.

    This simple object stores ndarrays & their coordinates. More sophisticated
    objects such as xarrays allow for arrays carrying metadata to be acted upon
    and propagate metadata appropriately. This does not occur with MetaArrays.

    Attrs:
        data:
            An N-dimensional numpy array to represent.
        **coords:
            keyword arguments specifying the axis names & index labels. The
            number of axes in coordinates must match the dims of data and the
            length of labels along axis must match length of data along axis.
                
            #### Coordinate labels may be list 1-D arrays or range instances
                arrays are converted to list

            EFFICIENCY NOTE ABOUT RANGE AND ARRAYS

    Examples:
        >>> data = np.random.random((3, 4, 6))
        >>> trials = [f'trial_{idx}' for idx in range(data.shape[0])]
        >>> counts = [f'count_{idx}' for idx in range(data.shape[1])]
        >>> times = np.arange(data.shape[-1])
        >>> # build a MetaArray
        >>> m = MetaArray(data, trial=trials, count=counts, time=times)
        >>> m.shape
        (3, 4, 6)
        >>> for axis, indices in m.coords.items():
        ...     print(f'{axis}: {indices}')
        trial: ['trial_0', 'trial_1', 'trial_2']
        count: ['count_0', 'count_1', 'count_2', 'count_3']
        time: [0 1 2 3 4 5]
        >>> z = m.select(trial=['trial_0', 'trial_2'], time=np.arange(6))
        >>> z.shape
        (2, 4, 6)
        >>> # compare slice of m's data to z's data
        >>> np.allclose(m.data[[0,2], :, 0:6], z.data)
        True
    """

    def __init__(self, data, **coords):
        """Initialize this MetaArray with an array & coordinates dictionary."""

        self.data = data
        self.coords = self._assign_coords(coords)
        
    def _assign_coords(self, coords):
        """Validates and assigns coordinates to this MetaArray.

        Args:
            coords:
                The axis names and axis index labels to use as coordinates for
                this MetaArray. The length of the labels along each axis
                must match data's shape along axis.

                #### Coordinate labels may be list 1-D arrays or range instances
                arrays are converted to list

        Returns:
            A dictionary of coordinates.

        Raises:
            A ValueError is issued if the dimensionality or shape of the
            coordinates does not match data's dims. or shape 
        """

        default = {f'axis{ix}': range(s) for ix, s in enumerate(self.shape)}
        coords = copy.deepcopy(coords) if coords else default

        coords = {name: list(labels) if not isinstance(labels, range) else labels
                  for name, labels in coords.items()}

        # validate dims & shape of coordinates
        if tuple(len(v) for v in coords.values()) != self.shape:
            msg = (f"The shape of the coordinates must match data's shape"
                    "{tuple(len(v) for v in coords.values())} != {self.shape}")
            raise ValueError(msg)

        return coords

    @property
    def shape(self):
        """Returns the shape of this MetaArray."""

        return self.data.shape

    def to_indices(self, 
                   name,
                   labels: Union[str, numbers.Number, Sequence, npt.NDArray],
    ) -> Tuple[int, Sequence]: 
        """Converts a coordinate axis name & labels to numeric data axis & 
        indices.

        Args:
            name:
                The name of a coordinate axis.
        """


        axis = tuple(self.coords).index(name)
        if not isinstance(labels, (abc.Sequence, np.ndarray)):
            labels = [labels]
        indices = [self.coords[name].index(label) for label in labels]

        return axis, indices

    def select(self, **selections):
        """Takes requested labeled elements along each axis in selections.
        
        Args:
            **selections:
              A list of named axes and labels to take from MetaArray.

        Returns:
            A MetaArray whose data and coordinates contain only selections.

        FIXME Note about slow selection for large label sequences
        """
        
        # coords will be mutated so copy for new instance
        coords = copy.deepcopy(self.coords)
        data = self.data

        for name, labels in selections.items():

            axis, indices = self.to_indices(name, labels)

            # filter the data and update axes indices
            data = np.take(data, indices, axis=axis)
            coords.update({name: labels})

        cls = type(self)
        instance = cls(data, **coords)

        metadata = {key: val for key, val in self.__dict__.items() if  
        return instance


class MetaMask(mixins.ViewInstance):
    """A callable that stores & returns element-wise combinations of 1-D
    boolean masks.
    
    Examples:
        >>> m = MetaMasks(x=[1, 0, 0, 1], y=[0, 1, 0, 1])
        >>> m('x', 'y', np.logical_and)
        [0, 0, 0, 1]
    """

    def __init__(self,
                 metadata: Optional[Dict] = None, 
                 **named_masks,
    ) -> Tuple[str, npt.NDArray[np.bool_]]:
        """Initialize this MetaMask with metadata and named mask to store.
        
        Args:
            metadata:
                Any desired metadata to associate with the named masks.
            named_masks:
                A mapping of named submasks to store.
        """

        self.__dict__.update(**named_masks)
        self.metadata = {} if not metadata else metadata

    def __call__(self, *names, logical=np.logical_and):
        """Returns the element-wise logical combination of all mask with name in
        names.

        Args:
            names:
                The string name(s) of mask to logically combine.
            logical:
                A callable that accepts and combines two 1-D boolean masks.

        Reutrns:
            A tuple containing a combined string name and a 1-D boolean array,
            the element-wise combination of each named mask.
        """

        submasks = [getattr(self, name) for name in names]
        
        lengths = np.array([len(m) for m in submasks])
        min_length = np.min(lengths)
        if any(lengths - min_length):
            
            msg = (f'Mask lengths are inconsistent lengths = {lengths}.'
                    'Truncating masks to minimum length = {min_length}')
            warnings.warn(msg)
            
            submasks = [mask[:minima] for mask in submasks]
        
        name = '_'.join(names)
        return name, functools.reduce(logical, submasks)

        


        
        



if __name__ == '__main__':

    import string
    s = (3, 2, 10)
    x = np.random.random(s)
    animals = [f'animal {idx}' for idx in range(s[0])]
    drugs = [letter for letter, _ in zip(string.ascii_letters, range(s[1]))]
    samples = np.arange(s[-1])
    samples = range(s[-1])

    m = MetaArray(x, animals=animals, drugs=drugs, samples=samples)
    g = m.select(drugs=['a'])

    #k = MetaMask(awake=[1,1,0,1,0,0], threshold_6=[0, 1, 1, 1, 1, 1])

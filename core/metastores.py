import copy
import functools
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import numpy.typing as npt
from spectraprints.core import mixins

class MetaArray(mixins.ViewInstance):
    """A representation of a numpy NDArray containing both the array &
    coordinates, a dict of axes names and axis index values.

    This is a simple object that stores ndarrays and their coordinates within
    a python object. Better and more sophisticated objects such as xarrays
    allow for arrays carrying metadata to be acted upon and propagate metadata
    appropriately.

    Attrs:
        data:
            An N-dimensional numpy array to represent.
        **coords:
            keyword arguments specifying the axis name and indices.

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
        """Initialize this MetaArray with a numpy array and coordinates."""

        self.data = data
        self.coords = self._assign_coords(coords)
        self.metadata = {}
        
    def _assign_coords(self, coords):
        """ """
       
        shape = self.shape
        result = {f'axis{idx}': np.arange(s) for idx, s in enumerate(shape)}
        
        if coords:

            if len(coords) != self.data.ndim:
                msg = (f'The coordinate dimensions must match the data'
                        'dimensions {len(coords)} != {self.data.ndim}')
                raise ValueError(msg)

            # copy since select method mutates coords
            result = copy.deepcopy(coords)
        
        return result

    def assign(self, **metadata):
        """ """

        self.metadata.update(metadata)

    @property
    def shape(self):
        """Returns the shape of this Archive."""

        return self.data.shape

    def select(self, **filters):
        """ """

        coords, data = copy.deepcopy(self.coords), self.data
        
        axes = tuple(self.coords)
        for name, elements in filters.items():

            # get data's axis number and indices to keep along axis
            axis = axes.index(name)

            if isinstance(elements, np.ndarray):
                locs = elements
            else:
                locs = [self.coords[name].index(el) for el in elements]
            
            # filter the data and update axes indices
            data = np.take(data, locs, axis=axis)
            coords.update({name: elements})

        cls = type(self)
        instance = cls(data, **coords)
        instance.metadata = self.metadata
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

    m = MetaArray(x, animals=animals, drugs=drugs, samples=np.arange(s[-1]))
    g = m.select(drugs=['a'])

    #k = MetaMask(awake=[1,1,0,1,0,0], threshold_6=[0, 1, 1, 1, 1, 1])

import copy
import numpy as np

from spectraprints.core import mixins

class MetaArray(mixins.ViewInstance):
    """ """

    def __init__(self, data, metadata=None, **indices):
        """ """

        self.data = data
        self.indices = self._indices(indices)
        self.axes = tuple(indices)
        self.metadata = {} if metadata is None else metadata
        
    def _indices(self, indices):
        """ """
        
        shape = self.shape
        result = {f'axis{idx}': np.arange(s) for idx, s in enumerate(shape)}
        
        if indices:
            result = copy.deepcopy(indices)
        
        return result

    @property
    def shape(self):
        """Returns the shape of this Archive."""

        return self.data.shape

    @classmethod
    def from_file(cls, path):
        """ """

        pass

    def save(self, path):
        """ """

        pass

    def filter(self, **filters):
        """ """

        indices, data = copy.deepcopy(self.indices), self.data
        for name, elements in filters.items():

            # get data's axis number and indices to keep along axis
            axis = self.axes.index(name)
            locs = [self.indices[name].index(el) for el in elements]
            
            # filter the data and update axes indices
            data = np.take(data, locs, axis=axis)
            indices.update(name=elements)

        cls = type(self)
        instance = cls(data, indices)
        instance.metadata = self.metadata
        return instance
      

class MetaItem(mixins.ViewInstance):
    """ """

    def __init__(self, datum, indices):
        """ """

        self.datum = datum
        self.__dict__.update(indices)
        

class MetaRaggedArray(mixins.ViewInstance):
    """ """

    def __init__(self, data, indices, metadata=None):
        """ """

        # data will be list
        # indices will be list of dicts
        # 0) ensure indices keys all match
        # 1) build data classes to store each data instance and indices
        #

        # ensure indices keys match for each datum in data
        # assert data are all same number of dims

        self._data = [MetaItem(dat, idxs) for dat, idxs in zip(data, indices)]    
        self.indices = indices
        self.axes = tuple(indices[0])
        self.metadata = {} if metadata is None else metadata

    @property
    def data(self):
        """ """

        return [metaitem.datum for metaitem in self._data]

    @property
    def shape(self):
        """ """

        return [len(ls) for ls in self.data]
    
    @classmethod
    def from_file(cls, path):
        """ """

        pass

    def save(self, path):
        """ """

        pass

    def filter(self, **filters):
        """ """

        data = self._data
        for name, elements in filters.items():

            metaitems = [item if el in item.name for item in data for el in item]
        
        indices = [item.__dict__ for item in metaitems]
        # remove the datum attr from indices
        [item.pop('datum') for item in indices]

        data = [metaitem.datum for metaitem in metaitems]

        cls = type(self)
        instance = cls(data, indices)
        instance.metadata = metadata
        return instance


if __name__ == '__main__':

    import string
    s = (3, 2, 10)
    x = np.random.random(s)
    animals = [f'animal {idx}' for idx in range(s[0])]
    drugs = [letter for letter, _ in zip(string.ascii_letters, range(s[1]))]

    #m = MetaArchive(x, animals=animals, drugs=drugs, samples=np.arange(s[-1]))

    y = [[[np.random.choice([True, False], size=l)]
          [
    indices = [{'geno': 'a', 'animals': [f'animal_{idx}' for idx in range(3)]},
            {'geno': 'b', 'animals': [f'animal_{idx}' for idx in range(4, 9)]}]
    f = MetaRaggedArray(y, indices)


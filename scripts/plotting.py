from typing import Dict, List, Optional
from matplotlib import pyplot as plt
import numpy as np
import pickle

from matplotlib import pyplot as plt
import numpy as np

def boxplot_dict(data: Dict, show_data: bool = False, jitter: float = 0,
                 **kwargs):
    """Boxplots the data in dictionary using keys as labels.

    Args:
        data:
            Dataset containing named arrays to boxplot.
        show_data:
            Boolean indicating if individual data points should be overlayed on
            the boxplots.
        jitter:
            The maximum x-displacement per sequence in data if show_data
            otherwise this parameter is ignored.
        kwargs:
            Any valid keyword argument for matplotlib's boxplot.
    
    Returns:
        A matplotlib axes instance.
    """
    
    fig, ax = plt.subplots()
    
    groups = ["Threshold = 3","Threshold = 4", "Threshold = 5", "Threshold = 6"]
    groups = data.keys()
    ax.boxplot(data.values(), labels=groups, **kwargs)
    if show_data:
        for idx, values in enumerate(data.values(), 1):
            locs = np.random.uniform(-jitter, jitter, len(values)) + idx
            plt.scatter(locs, values)
    return ax

if __name__ == '__main__':
    fp = '/media/sandy/Data_A/sandy/results/ube3a_mask_performances.pkl'
    with open(fp, 'rb') as infile:
        ube3a_perfs = pickle.load(infile)
    var_list = ['accuracy','sensitivity','specificity','precision','perc_within']
    for var in var_list:
        data = ube3a_perfs[var]
        ax = boxplot_dict(data = data, show_data=True, jitter= 0.05)
        plt.title(var.capitalize(), size =12)
        plt.xlabel("Masking at different thresholds")
        plt.ylabel("Similarity to human-annotated note (%)")
        name = ["Threshold = 3","Threshold = 4", "Threshold = 5", "Threshold = 6"]
        plt.show()

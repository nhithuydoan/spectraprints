from typing import Dict, List, Optional

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
    
    groups = data.keys()
    ax.boxplot(data.values(), labels=groups, **kwargs)
    if show_data:
        for idx, values in enumerate(data.values(), 1):
            locs = np.random.uniform(-jitter, jitter, len(values)) + idx
            plt.scatter(locs, values)
    return ax

if __name__ == '__main__':

    data = {'sleep': 3*np.random.random(10), 'awake': 5*np.random.random(22)}
    
    ax = boxplot_dict(data, show_data=True, jitter=0.05)

    ax.set_ylabel('counts')
    ax.set_xlabel('state')
    ax.set_title('some weird data')
    plt.show()


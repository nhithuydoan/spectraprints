def draw_boxplot(data = dict, xlabel = str, ylabel = str, title = str, labels = List[str],
                 show_pts = False, **kwargs):
    """
    This function takes dictionary data and draw a boxplot.
    
    Args: data: dataset in form of dictionary with keys and values. All
    variables should have the same length.  xlabel: label of x-axis ylabel:
    label of y-axis title: title of the plot show_pts: return a jitterplot
    overlay the boxplot if the argument is True labels: take a list of string to
    create names for x variables. If the list is empty, function will use name
    from the dataset as variable names kwargs: any valid key word arguments are
    passed into matplotlib's boxplot function.
    
    """
    
    data = data
    pro.chunksize = winsize
    fig, ax = plt.subplots()
    if not labels:
        labels = list(data.keys())
    else:
        labels = labels
    ax.boxplot(data.values(), medianprops = dict(color = "black", linewidth = 1.5),labels = labels)
    if show_pts == True:
        for i, d in enumerate(data):
            y = data[d]
            x = np.random.normal(i + 1, 0.04, len(y))
            plt.scatter(x, y)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, weight = 'bold', fontsize = 14)
    plt.show()

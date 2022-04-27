import matplotlib.pyplot as plt
import numpy as np
import collections

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def plot(upvotes, output_file, xlab, ylab, title_plot):
    ordered_upvotes = collections.OrderedDict(sorted(upvotes.items()))
    percentiles = [25,50,75,100]
    X = [str(round(np.percentile(list(ordered_upvotes.keys()), 25),2)), str(round(np.percentile(list(ordered_upvotes.keys()), 50),2)), str(round(np.percentile(list(ordered_upvotes.keys()), 75),2)), str(round(np.percentile(list(ordered_upvotes.keys()), 100),2))]
    JCA_avg = []
    JCH_avg = []
    SAT_avg = []
    SAH_avg = []
    for per in percentiles:
        JCA_avg.append(ordered_upvotes[find_nearest(list(ordered_upvotes.keys()), np.percentile(list(ordered_upvotes.keys()), per))][0])
        JCH_avg.append(ordered_upvotes[find_nearest(list(ordered_upvotes.keys()), np.percentile(list(ordered_upvotes.keys()), per))][1])
        SAT_avg.append(ordered_upvotes[find_nearest(list(ordered_upvotes.keys()), np.percentile(list(ordered_upvotes.keys()), per))][2])
        SAH_avg.append(ordered_upvotes[find_nearest(list(ordered_upvotes.keys()), np.percentile(list(ordered_upvotes.keys()), per))][3])
        
    X_axis = np.arange(len(X))

    plt.bar(X_axis - 0.1, JCA_avg, 0.1, label = 'JCA_avg')
    plt.bar(X_axis + 0.0, JCH_avg, 0.1, label = 'JCH_avg')
    plt.bar(X_axis + 0.1, SAT_avg, 0.1, label = 'SSA_avg')
    plt.bar(X_axis + 0.2, SAH_avg, 0.1, label = 'SSH_avg')
    
    plt.xticks(X_axis, X)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title_plot)
    plt.legend()
    plt.savefig(output_file)

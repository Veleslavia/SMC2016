import numpy as np
import matplotlib.pyplot as plt


"""
plot_confusion_matrix([[2, 0, 0, 0, 1],
                       [1, 2, 0, 1, 0],
                       [0, 1, 3, 1, 0],
                       [0, 1, 0, 3, 1],
                       [0, 1, 0, 0, 4]], labels=["short name", "One", "Two", "Three", "long name"],
                      title='Confusion matrix')
"""

def plot_confusion_matrix(cm, title='Confusion matrix', labels=None, cmap=plt.cm.Blues):

    if not labels or (len(labels) != len(cm)):
        labels = np.arange(len(cm))

    fig = plt.figure()
    axes = fig.add_axes([0, 0.2, 1.1, 0.7])
    axes.set_title(title)
    axes.set_xlabel("Predicted label")
    axes.set_ylabel("True label")

    axes.set_xticks(np.arange(len(labels)))
    axes.set_yticks(np.arange(len(labels)))

    axes.set_xticklabels(labels, rotation=45)
    axes.set_yticklabels(labels)
    axes.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.savefig("{filename}.pdf".format(filename=title), format='pdf', pad_inches=0.5)
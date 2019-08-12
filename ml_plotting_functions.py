# -*- coding: utf-8 -*-


from matplotlib import collections
from matplotlib import colors
from numpy.random import normal
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

plt.style.use('seaborn-darkgrid')

DEFAULT_MARBLE_COLOR = 'C0'


def draw_value_circles(ax, circle_colors, coords):
    """Draw a scatter plot on a list of coordinates"""

    ax.scatter(*list(zip(*coords)),
               facecolor=circle_colors,
               edgecolor='black',
               s=200)
    ax.axis('off')


def predict_values(true_values, quality):
    """Basically a classifier that draws from a gaussian.

    quality = 0: mean around 0.5, broad  -> basically random
    quality = 1: mean around 0.25 (for true_value == 1) or 0.75 (for true_value == 0), narrow -> true values should be separable
    """
    estimated_values = []

    for true_value in true_values:  # 0 or 1

        true_value = normal(0.5 + (true_value * 2 - 1) * 0.25 * quality, 0.17 - (0.075 * quality), 1)[0]
        true_value = max([min([true_value, 1]), 0])

        estimated_values.append(true_value)
    return estimated_values


def make_roc_curve_plot(ax, true_values, predicted_values, cutoff):
    """Calculate ROC and AUC from true and predicted values and draw."""

    fpr, tpr, thresholds = metrics.roc_curve(true_values, predicted_values)
    ax.plot(fpr, tpr, label='ROC')

    auc = metrics.roc_auc_score(true_values, predicted_values)
    ax.set_title(f"AUC: {auc:.3f}")

    for fp, tp, threshold in zip(fpr, tpr, thresholds):
        if threshold < cutoff:
            ax.plot(fp, tp, marker='o', markersize=10, color='grey', alpha=0.75)
            break

    ax.plot([0, 1], [0, 1], c='grey', alpha=0.5)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")


def make_precision_recall_bar_chart(ax, true_values, predicted_binary_values):
    """Calculate accuracy / recall / precision and draw as bar chart."""
    accuracy = metrics.accuracy_score(true_values, predicted_binary_values)
    recall = metrics.recall_score(true_values, predicted_binary_values)
    precision = metrics.precision_score(true_values, predicted_binary_values)

    pd.DataFrame.from_dict({
        'Accuracy': [accuracy],
        'Recall': [recall],
        'Precision': [precision]
    }, orient='index').plot(kind='bar', legend=False, rot=0, ax=ax, color=plt.cm.tab20c.colors[:3])

    ax.set_ylim(0, 1)


def swarmplot_with_custom_colors(ax, x_values, face_colors=None, edge_colors=None, **kwargs):
    """This is a wrapper around seaborn's swarmplot for setting colors post-plotting."""
    sns.swarmplot(x=x_values,
                  ax=ax,
                  **kwargs)
    ax.set_xlim(-0.01, 1.01)

    # modify path collection
    pc = [pc for pc in ax.get_children() if type(pc) == collections.PathCollection][0]
    if face_colors is not None:
        pc.set_facecolors(face_colors)
    if edge_colors is not None:
        pc.set_edgecolors(edge_colors)


def plot_predicted_values_as_swarmplot_with_color_gradient(ax, predicted_values, **kwargs):

    face_colors = [tuple(list(colors.to_rgb(DEFAULT_MARBLE_COLOR)) + [c]) for c in predicted_values]

    swarmplot_with_custom_colors(ax,
                                 x_values=predicted_values,
                                 face_colors=face_colors,
                                 **kwargs)

    ax.set_xlabel("Classifier Score")


def plot_predicted_values_as_swarmplot_with_green_red_outline(ax, true_values, predicted_values, predicted_binary_values, cutoff):
    eval_colors = []
    for true, pred in zip(true_values, predicted_binary_values):
        if true == pred:
            c = 'C2'  # 'lightgreen'
        else:
            c = 'C3'  # 'darkred'
        eval_colors.append(c)

    face_colors = [colors.to_rgba(DEFAULT_MARBLE_COLOR) if v == 1 else colors.to_rgba('white') for v in true_values]
    edge_colors = [colors.to_rgba(c) for c in eval_colors]
    swarmplot_with_custom_colors(ax,
                                 x_values=predicted_values,
                                 face_colors=face_colors,
                                 edge_colors=edge_colors,
                                 linewidth=1,
                                 s=10
                                 )

    ax.axvline(cutoff)
    ax.axvspan(cutoff, 1, color=DEFAULT_MARBLE_COLOR, alpha=0.1, zorder=-1)

    ax.axis('off')
    ax.set_title("Classifier result: true/false")



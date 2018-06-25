""" Plots the weight values obtained from the portfolio.

Copyright 2018 Cognitive Computation Lab
University of Freiburg
Nicolas Riesterer <riestern@tf.uni-freiburg.de>
Daniel Brand <daniel.brand@cognition.uni-freiburg.de>

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_LIST = [
    'PHM-Min',
    'PHM-Min-Att',
    'Matching',
    'Atmosphere',
    'Conversion',
    'FOL',
    'FOL-Strict',
    'PSYCOP',
    'WCS'
]

def main():
    # Load the data
    weight_df = pd.read_csv('output/weights.csv')

    # Prepare the heatmap matrix
    n_models = len(MODEL_LIST)
    n_tasks = len(weight_df['task'].unique())
    heatmap_data = np.zeros((n_models, n_tasks))

    for model_idx, model in enumerate(MODEL_LIST):
        model_df = weight_df.loc[weight_df['model'] == model]
        model_df = model_df.groupby('task', as_index=False).agg('mean')
        weights = model_df.sort_values('task')['weight']
        heatmap_data[model_idx] = weights

    # Prepare the syllogistic problem labels
    syllogs = []
    for p1 in ['A', 'I', 'E', 'O']:
        for p2 in ['A', 'I', 'E', 'O']:
            for f in ['1', '2', '3', '4']:
                syllogs.append(p1 + p2 + f)

    # Setup seaborn plotting
    sns.set(style='ticks')
    cmap = sns.cubehelix_palette(
        200, start=2.71, rot=0, dark=0.3, light=.85, reverse=False)

    # set figsize
    fac = 2
    figsize = (4.8 * fac, 2.4 * fac)

    # Draw the heatmaps
    fig, (ax0, ax1) = plt.subplots(
        2, 1, sharex=False, sharey=True, figsize=figsize)
    cbar_ax = fig.add_axes([.92, .19, .03, .7])

    # Upper heatmap
    h1 = sns.heatmap(
        heatmap_data[:, :32], ax=ax0, cmap=cmap, cbar=True,
        vmin=heatmap_data.min(), vmax=heatmap_data.max(), cbar_ax=cbar_ax,
        linewidths=0.6)
    h1.set_xticklabels(syllogs[:32], rotation=90)
    h1.set_yticklabels(MODEL_LIST, rotation=0)

    # Lower heatmap
    h2 = sns.heatmap(
        heatmap_data[:, 32:], ax=ax1, cmap=cmap, cbar=True,
        vmin=heatmap_data.min(), vmax=heatmap_data.max(), cbar_ax=cbar_ax,
        linewidths=0.6)
    h2.set_xticklabels(syllogs[32:], rotation=90)
    h2.set_yticklabels(MODEL_LIST, rotation=0)

    # Finalize and show the plots
    plt.subplots_adjust(top=0.99, left=0.12, right=0.9, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    main()

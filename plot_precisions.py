""" Plots the precision values obtained by the models.

Copyright 2018 Cognitive Computation Lab
University of Freiburg
Nicolas Riesterer <riestern@tf.uni-freiburg.de>
Daniel Brand <daniel.brand@cognition.uni-freiburg.de>

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Load the precision data
    precision_df = pd.read_csv('output/precisions.csv')
    n_models = len(precision_df['model'].unique())

    # Setup seaborn plotting
    sns.set(style='whitegrid')
    palette = sns.cubehelix_palette(
        n_models, start=2.71, rot=0, dark=0.3, light=.85, reverse=True)

    # Sort models (excluding MFA) by precision
    model_order = precision_df.groupby('model', as_index=False)['value'].agg(
        'mean').sort_values('value')['model'].tolist()
    model_order.remove('MFA')

    # Plot the precision barplot
    sns.barplot(
        x='model', y='value', data=precision_df, order=model_order,
        palette=palette)

    # Plot the MFA upper bound
    mfa_prec = precision_df.loc[precision_df['model'] == 'MFA']['value'].mean()
    plt.axhline(y=mfa_prec, color='#C3272B', ls='--')

    # Finalize and show the plot
    sns.despine()
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('Mean Precision')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

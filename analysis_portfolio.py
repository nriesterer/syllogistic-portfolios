""" Creates and evaluates the portfolio.

Copyright 2018 Cognitive Computation Lab
University of Freiburg
Daniel Brand <daniel.brand@cognition.uni-freiburg.de>
Nicolas Riesterer <riestern@tf.uni-freiburg.de>

"""

import collections

import pandas as pd
import numpy as np
import scipy.stats as ss

import syldata as sd
import sylmodel as sm

def simulate():
    # Simulation preferences
    n_train_examples = 100
    n_samples = 500

    # Collector variables for storing the results
    weight_data = []
    precision_data = []

    # Load models
    prediction_dict = {}
    for model in sm.MODEL_WHITELIST:
        prediction_dict[model] = sm.load_predictions(model)

    # Load the dataset
    data_df = sd.load_data()

    # Perform repeated random subsampling for n_sample iterations
    for sample in range(n_samples):
        print("Running sample " + str(sample + 1) + "/" + str(n_samples))

        # Split data into training and test set
        rnd_idxs = np.random.permutation(
            np.arange(len(data_df['subj'].unique())))
        train_df = data_df.loc[data_df['subj'].apply(
            lambda x: x in rnd_idxs[:n_train_examples])]
        test_df = data_df.loc[data_df['subj'].apply(
            lambda x: x in rnd_idxs[n_train_examples:])]

        train_data = np.array(list(sd.observation_dict(train_df).values()))
        test_data = np.array(list(sd.observation_dict(test_df).values()))

        # Obtain most frequent answer model
        most_freq = sm.most_frequent_model(test_df)

        # Aggregate the training data
        train_data_agg = np.zeros((9, 64))
        for idx in range(64):
            cur_train_data = train_data[:, idx]
            cnt = collections.Counter(cur_train_data)

            for key, value in dict(cnt).items():
                train_data_agg[key - 1, idx] = value

        # Create portfolio based on score metric
        rankings = []
        for model, prediction in prediction_dict.items():
            for task_idx, task_prediction in enumerate(prediction.T):
                rank = ss.rankdata(train_data_agg[:, task_idx] * -1)
                rrs = []
                nonzero = np.argwhere(task_prediction).flatten()
                for nonzero_idx in nonzero:
                    rrs.append(1 / rank[nonzero_idx])

                rankings.append({
                    'task': task_idx,
                    'model': model,
                    'mrr': np.mean(rrs)
                })

        ranking_df = pd.DataFrame(rankings)

        # Generate the portfolio prediction matrices of shape
        # (n_responses x n_syllogisms)
        portfolio = np.zeros((9, 64))
        portfolio_max = np.zeros((9, 64))

        for task_idx, task_df in ranking_df.groupby('task'):
            task_prediction = np.zeros((9,))
            for _, model_df in task_df.iterrows():
                weight = model_df['mrr']
                model_name = model_df['model']
                model_pred = prediction_dict[model_name][:, task_idx]

                portfolio[:, task_idx] += weight * model_pred

                weight_data.append({
                    'task': task_idx,
                    'model': model_name,
                    'weight': weight
                })

            # Create the Portfolio-Max composition by selecting the model(s)
            # with maximum MRR
            max_mrr = task_df['mrr'].max()
            for _, portfolio_df in task_df.iterrows():
                if portfolio_df['mrr'] != max_mrr:
                    continue

                model_name = portfolio_df['model']
                model_pred = prediction_dict[model_name][:, task_idx]

                portfolio_max[:, task_idx] += model_pred

        # Add most frequent answer model to evaluation
        prediction_dict['MFA'] = most_freq

        # Evaluate portfolio on data
        portfolio_pred_list = [
            ('Portfolio', portfolio), ('Portfolio-Max', portfolio_max)]
        eval_pred_list = list(prediction_dict.items()) + portfolio_pred_list

        for model, prediction in eval_pred_list:
            for responses in test_data:
                for task_id, resp in enumerate(responses):
                    task_prediction = prediction[:, task_id]
                    prediction_mask = task_prediction == task_prediction.max()
                    value = prediction_mask[resp - 1] / prediction_mask.sum()

                    precision_data.append({
                        'task': task_id,
                        'model': model,
                        'value': value
                    })

    # Store the resulting data
    weight_df = pd.DataFrame(weight_data)
    weight_df.to_csv('output/weights.csv', index=False)

    precision_df = pd.DataFrame(precision_data)
    precision_df.to_csv('output/precisions.csv', index=False)

if __name__ == '__main__':
    simulate()

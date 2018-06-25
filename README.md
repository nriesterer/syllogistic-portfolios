Syllogistic Portfolios
======================

This respository contains the code required to reproduce the results reported in "The Predictive Power of Heuristic Portfolios in Human Syllogistic Reasoning" (Riesterer, Brand & Ragni, 2018) in *Proceedings of the 41st German Conference on Artificial Intelligence*.

## Repository Contents

- `data/`: Folder containing the dataset.
- `models/`: Folder containing the model prediction tables partially obtained from [1].
- `output/`: Target folder for the analysis results
- `analysis_portfolio.py`: Script to run the portfolio analysis.
- `plot_precisions.py`: Script for creating the precision barplot.
- `plot_weights.py`: Script for creating the weight matrix heatmap.
- `syldata.py`: Auxiliary functions for data handling.
- `sylhelper.py`: Auxiliary functions for syllogisms.
- `sylmodel.py`: Auxiliary functions for model handling.

## Software Dependencies

- [Python 3](https://www.python.org)
- [Numpy](http://www.numpy.org)
- [Scipy](https://www.scipy.org)
- [Pandas](https://pandas.pydata.org)
- [Matplotlib](https://matplotlib.org)
- [Seaborn](https://seaborn.pydata.org)

## Running the Simulation

A reproduction of the results reported in the article can be achieved by running the following commands:

```shell
$> python analysis_portfolio.py
$> python plot_precisions.py
$> python plot_weights.py
```

`analysis_portfolio.py` performs the analysis and writes the intermediate output data to `output/`. Subsequently, the plots can be created by calling `plot_precisions.py` and `plot_weights.py`, respectively.

## References

[1]: Khemlani, S., & Johnson-Laird, P. N. (2012). Theories of the syllogism: A meta-analysis. *Psychological bulletin*, 138(3), 427.

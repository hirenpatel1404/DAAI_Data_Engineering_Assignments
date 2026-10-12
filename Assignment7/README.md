# Assignment 7: Data Distribution Shift Detection

In this assignment I simulate data distribution shift and test which detection methods can find it.

## What I did
- Generated a **reference** dataset (2,000 rows, 10 normal features) and a **current** dataset in which I shifted 7 features on purpose:
  - 5 strong shifts in mean and spread: f0 to f4
  - 2 subtle shifts: f5 has a small mean drift and f6 has a small variance change
  - 3 control features with no change: f7 to f9
- Visualised the shift with histograms and scatter plots.
- Ran univariate tests on every feature with a Bonferroni-corrected α (alpha) of 0.005:
  - KS (Kolmogorov-Smirnov)
  - AD (Anderson-Darling)
  - CvM (Cramér-von Mises)
  - Welch t-test
  - Levene
- Ran multivariate tests:
  - an MMD (Maximum Mean Discrepancy) permutation test
  - a Random Forest domain classifier
- Measured how the shift hurts a Random Forest model, and how retraining fixes it.

## Key results
- Every strong shift was caught by all five tests.
- f5 (mean drift) was caught by KS, AD, CvM and the t-test, but **missed by Levene**.
- f6 (variance only) was **caught only by Levene**.
- Combining tests caught 7 out of 7 shifts with 0 false alarms.
- Without Bonferroni, CvM would have raised a false alarm on f7 (p = 0.046).
- The domain classifier AUC (Area Under the ROC Curve) was 0.997 on all features and 0.50 on the controls. MMD was significant on all features (p = 0.005) and not significant on the controls.
- Model accuracy dropped from **0.935 to 0.768** on shifted data. Retraining brought it back to **0.945**.

## Files
- `Assignment_7_Data_Shift_Detection.ipynb`: full code, outputs, interpretation and recommendations
- `*.png`: saved charts (histograms, scatter pairs, p-values, domain classifier importance, model impact)

## How to run
Open the notebook in Jupyter with the Python 3.14 (Assignment6) kernel and choose **Run All**. It needs numpy, pandas, scipy, scikit-learn and matplotlib, and finishes in under a minute.

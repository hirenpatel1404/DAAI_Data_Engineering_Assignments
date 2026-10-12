# Assignment 10: Fairness Interventions on the German Credit Dataset

In this assignment I train a credit-risk model on the UCI German Credit data, measure its bias against two protected groups, and test three fairness interventions.

## Protected attributes
- **Sex:** derived from `personal_status`. Male is the privileged group.
- **Age:** 25 or younger is "young", over 25 is "older". Older is the privileged group.

## Models
| Model | Type |
|---|---|
| Baseline logistic regression | none |
| Reweighing (Kamiran & Calders) | pre-processing |
| Disparate Impact Remover (Feldman et al.) | pre-processing |
| Group-specific thresholds for equal opportunity (Hardt et al.) | post-processing |

I implemented all three interventions myself with NumPy and scikit-learn.

## Key results (mean over 20 random splits)
| Model | Balanced acc. | DI (Disparate Impact) sex | DI age | Splits failing four-fifths rule (age) |
|---|---|---|---|---|
| Baseline | 0.668 | 0.876 | 0.834 | 7 / 20 |
| Reweighing | 0.662 | 0.965 | 0.922 | 0 / 20 |
| DI Remover | 0.677 | 0.899 | 0.840 | 6 / 20 |
| Group thresholds (age) | 0.656 | 0.934 | 1.028 | 0 / 20 |

**Conclusion:** reweighing gives the best balance between fairness, performance and legal safety. The DI Remover has little effect, because the bias mainly sits in categorical proxy features that it does not repair.

## Files
- `Assignment_10_Fairness_German_Credit.ipynb`: code, outputs, interpretation of every step, and the conclusion
- `german.data`: the UCI German Credit dataset
- `*.png`: saved charts

## How to run
Open the notebook with the **Python 3.14 (Assignment6)** kernel and choose Run All. It needs pandas, numpy, scikit-learn and matplotlib, and finishes in under a minute.

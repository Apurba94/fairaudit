# FairAudit

A model can work well on average and fail one group of people badly. FairAudit measures that, with uncertainty, and writes the result up as a model card. It matters most where the people affected have little say, such as health, aid and education.

```python
from fairaudit import bootstrap, fairness_metrics, intersections, model_card

print(fairness_metrics(labels, predictions, groups))
print(bootstrap(labels, predictions, groups))

groups = intersections(sex, age_band)      # harm can hide where groups overlap
print(model_card("Referral model", "Rank cases for human review.", labels, predictions, groups, scores))
```

From a CSV with the columns `label`, `prediction`, `group` and optional `score`:

```
python -m fairaudit results.csv "Referral model" > card.md
```

## What it reports

| Question | Measure |
|---|---|
| Are groups selected at similar rates? | Demographic parity gap, disparate impact ratio |
| Are people who should be found, found equally? | Equal opportunity gap |
| Are error rates similar? | False positive gap, equalised odds gap |
| Does a positive prediction mean the same thing? | Predictive parity gap |
| Does a score mean the same risk? | Calibration by group |

Each comes with bootstrap intervals, and a permutation test asks whether a gap could be chance. `threshold_sweep` shows the trade-offs across operating points, and `model_card` writes everything in Markdown, following Mitchell et al. (2019).

## What this tool cannot do

- **It cannot tell you what is fair.** When base rates differ between groups, equal error rates and equal predictive values cannot both hold, so someone must decide which matters. The tool warns when this applies.
- **Measuring needs group labels,** and collecting them has privacy risks of its own. Take care, especially with vulnerable people.
- **It only sees your labels.** If the labels themselves are biased, a model can look fair while being unfair.
- `equalise_opportunity` sets a different threshold per group. In some settings, such as hiring, that may be unlawful or unwise, so take advice before using it.
- A clean report does not show that a model is fair. The model card says so in its text.

## How it is tested

Rates are checked by hand and, when scikit-learn is installed, on 100 random cases against its metrics. Simulations with a planted bias must be detected, while simulations without one must show small gaps. One test shows harm that is invisible by sex or age alone and clear in their intersection.

## Tests

```
pip install -r requirements.txt
python -m unittest discover -s tests
```

MIT licence.

"""Audit a binary or scored model across groups of people.

Fairness has no single number, and the common definitions cannot all hold at once when base
rates differ (Kleinberg et al. 2016; Chouldechova 2017). This code therefore reports them side
by side, with uncertainty, and leaves the choice of what matters to people who know the setting.
"""
import numpy as np

SMALL_GROUP = 30


def confusion(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    return int((y & p).sum()), int((~y & p).sum()), int((~y & ~p).sum()), int((y & ~p).sum())


def rates(y, p):
    """Standard rates for one group. A rate with no denominator is NaN, not zero."""
    tp, fp, tn, fn = confusion(y, p)
    n = tp + fp + tn + fn

    def ratio(a, b):
        return a / b if b else float("nan")
    return {"n": n, "base_rate": ratio(tp + fn, n), "selection_rate": ratio(tp + fp, n), "accuracy": ratio(tp + tn, n), "tpr": ratio(tp, tp + fn), "fpr": ratio(fp, fp + tn), "fnr": ratio(fn, tp + fn), "ppv": ratio(tp, tp + fp), "npv": ratio(tn, tn + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def by_group(y, p, groups):
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    return {str(g): rates(y[groups == g], p[groups == g]) for g in np.unique(groups)}


def intersections(*columns):
    """Combine attributes, because harm can hide where groups overlap."""
    return np.array([" & ".join(str(v) for v in row) for row in zip(*columns)])


def _gap(table, key):
    v = [r[key] for r in table.values() if not np.isnan(r[key])]
    return max(v) - min(v) if len(v) > 1 else float("nan")


def _ratio(table, key):
    v = [r[key] for r in table.values() if not np.isnan(r[key])]
    if len(v) < 2 or max(v) == 0:
        return float("nan")
    return min(v) / max(v)


def fairness_metrics(y, p, groups):
    t = by_group(y, p, groups)
    return {
        "demographic_parity_gap": _gap(t, "selection_rate"),
        "disparate_impact_ratio": _ratio(t, "selection_rate"),
        "equal_opportunity_gap": _gap(t, "tpr"),
        "false_positive_gap": _gap(t, "fpr"),
        "equalized_odds_gap": max(_gap(t, "tpr"), _gap(t, "fpr")) if not (np.isnan(_gap(t, "tpr")) or np.isnan(_gap(t, "fpr"))) else float("nan"),
        "predictive_parity_gap": _gap(t, "ppv"),
        "accuracy_gap": _gap(t, "accuracy"),
    }


def calibration_by_group(y, scores, groups, bins=10):
    """Do equal scores mean equal risk in every group? Returns expected calibration error by group."""
    y, scores, groups = np.asarray(y, float), np.asarray(scores, float), np.asarray(groups)
    edges = np.linspace(0, 1, bins + 1)
    out = {}
    for g in np.unique(groups):
        m = groups == g
        idx = np.clip(np.digitize(scores[m], edges) - 1, 0, bins - 1)
        e = 0.0
        for b in range(bins):
            k = idx == b
            if k.any():
                e += k.mean() * abs(scores[m][k].mean() - y[m][k].mean())
        out[str(g)] = float(e)
    return out


def bootstrap(y, p, groups, rounds=1000, level=0.95, seed=0):
    """Intervals on every fairness metric. Resamples within each group, so small groups stay present."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.default_rng(seed)
    members = [np.nonzero(groups == g)[0] for g in np.unique(groups)]
    keys = list(fairness_metrics(y, p, groups))
    samples = {k: [] for k in keys}
    for _ in range(rounds):
        idx = np.concatenate([rng.choice(m, len(m)) for m in members])
        f = fairness_metrics(y[idx], p[idx], groups[idx])
        for k in keys:
            samples[k].append(f[k])
    a = (1 - level) / 2
    out = {}
    for k in keys:
        v = np.array(samples[k], float)
        v = v[~np.isnan(v)]
        out[k] = (float(np.quantile(v, a)), float(np.quantile(v, 1 - a))) if v.size else (float("nan"), float("nan"))
    return out


def permutation_test(y, p, groups, metric="demographic_parity_gap", rounds=2000, seed=0):
    """Could this gap come from chance? Shuffles group labels and compares."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.default_rng(seed)
    observed = fairness_metrics(y, p, groups)[metric]
    hits = 0
    for _ in range(rounds):
        hits += fairness_metrics(y, p, rng.permutation(groups))[metric] >= observed - 1e-12
    return (hits + 1) / (rounds + 1)


def threshold_sweep(y, scores, groups, steps=101):
    """One row per threshold, for choosing an operating point with the trade-offs in view."""
    y, scores, groups = np.asarray(y), np.asarray(scores, float), np.asarray(groups)
    rows = []
    for t in np.linspace(0, 1, steps):
        p = scores >= t
        f = fairness_metrics(y, p, groups)
        rows.append({"threshold": float(t), "accuracy": rates(y, p)["accuracy"], **f})
    return rows


def equalise_opportunity(y, scores, groups, target=None):
    """Find one threshold per group that gives each about the same true positive rate.

    Group-specific thresholds may be unlawful or unwise in some settings, such as employment.
    The caller decides that. This only shows what the change would do.
    """
    y, scores, groups = np.asarray(y, bool), np.asarray(scores, float), np.asarray(groups)
    if target is None:
        target = float(((scores >= 0.5) & y).sum() / max(y.sum(), 1))
    out = {}
    for g in np.unique(groups):
        m = groups == g
        pos = np.sort(scores[m & y])[::-1]
        if len(pos) == 0:
            out[str(g)] = 1.0
            continue
        k = min(len(pos) - 1, max(0, int(np.ceil(target * len(pos))) - 1))
        out[str(g)] = float(pos[k])
    return out


def apply_thresholds(scores, groups, thresholds):
    scores, groups = np.asarray(scores, float), np.asarray(groups)
    return scores >= np.array([thresholds[str(g)] for g in groups])


def warnings(y, p, groups, four_fifths=0.8):
    """Plain-language flags for what a reader should look at."""
    table = by_group(y, p, groups)
    f = fairness_metrics(y, p, groups)
    out = []
    for g, r in table.items():
        if r["n"] < SMALL_GROUP:
            out.append(f"Group {g} has only {r['n']} people, so its numbers are unreliable.")
    d = f["disparate_impact_ratio"]
    if not np.isnan(d) and d < four_fifths:
        out.append(f"Selection rate ratio is {d:.2f}, below the four-fifths guide of {four_fifths}.")
    br = [r["base_rate"] for r in table.values() if not np.isnan(r["base_rate"])]
    if len(br) > 1 and max(br) - min(br) > 0.05:
        out.append("Base rates differ between groups, so equal error rates and equal predictive values cannot both hold.")
    return out


def model_card(name, purpose, y, p, groups, scores=None, not_for=None, data=None, contact=None, rounds=500):
    """A model card in Markdown (after Mitchell et al. 2019), filled with the measured numbers."""
    table = by_group(y, p, groups)
    metrics = fairness_metrics(y, p, groups)
    ci = bootstrap(y, p, groups, rounds)
    lines = [f"# Model card: {name}", "", "## Intended use", "", purpose, ""]
    if not_for:
        lines += ["Not intended for:", ""] + [f"- {x}" for x in not_for] + [""]
    if data:
        lines += ["## Evaluation data", "", data, ""]
    lines += ["## Performance by group", "", "| Group | n | Base rate | Accuracy | TPR | FPR | PPV |", "|---|---|---|---|---|---|---|"]

    def fmt(v):
        return "n/a" if np.isnan(v) else f"{v:.3f}"
    for g, r in table.items():
        lines.append(f"| {g} | {r['n']} | {fmt(r['base_rate'])} | {fmt(r['accuracy'])} | {fmt(r['tpr'])} | {fmt(r['fpr'])} | {fmt(r['ppv'])} |")
    lines += ["", "## Fairness measures", "", "| Measure | Value | 95% interval |", "|---|---|---|"]
    for k, v in metrics.items():
        lines.append(f"| {k.replace('_', ' ')} | {fmt(v)} | {fmt(ci[k][0])} to {fmt(ci[k][1])} |")
    if scores is not None:
        lines += ["", "## Calibration by group", "", "| Group | Expected calibration error |", "|---|---|"]
        lines += [f"| {g} | {e:.3f} |" for g, e in calibration_by_group(y, scores, groups).items()]
    flags = warnings(y, p, groups)
    lines += ["", "## Points to review", ""]
    lines += [f"- {w}" for w in flags] or ["- No automatic flags were raised. This does not show that the model is fair."]
    if contact:
        lines += ["", f"Contact: {contact}"]
    return "\n".join(lines) + "\n"

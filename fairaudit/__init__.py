"""FairAudit: measure how a model treats different groups."""
from .core import apply_thresholds, bootstrap, by_group, calibration_by_group, confusion, equalise_opportunity, fairness_metrics, intersections, model_card, permutation_test, rates, threshold_sweep, warnings

__all__ = ["apply_thresholds", "bootstrap", "by_group", "calibration_by_group", "confusion", "equalise_opportunity", "fairness_metrics", "intersections", "model_card", "permutation_test", "rates", "threshold_sweep", "warnings"]

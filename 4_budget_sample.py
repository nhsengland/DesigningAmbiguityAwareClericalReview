# ============================================================
# 4. Search for scale_star given a budget fraction
# ============================================================
from typing import Dict, List, Optional

import pandas as pd


def find_scale_for_budget(
    df_scored: pd.DataFrame,
    band_to_moe_base: Dict[str, float],
    budget_frac: float,
    strata_cols: List[str],
    band_col: str = "band_10",
    ambig_col: str = "ambig_bin",
    ambig_factor: Optional[Dict[str, float]] = None,
    prob_col: str = "match_probability",
    label_col: str = "true_match",
    max_iter: int = 20,
    tol: float = 0.002,
) -> float:
    """
    Find a global scaling factor 'scale_star' such that the
    total sample size is approximately budget_frac * N_total.

    Uses a simple bisection search on [scale_min, scale_max].

    Parameters
    ----------
    df_scored : DataFrame
        Pair-level table with scoring and strata columns defined.
    band_to_moe_base : dict
        Band -> baseline MOE mapping.
    budget_frac : float
        Target fraction of population to sample (e.g. 0.05).
    strata_cols : list of str
        Strata columns (includes band and ambiguity, etc.).
    max_iter : int
        Maximum bisection iterations.
    tol : float
        Tolerance on achieved fraction.

    Returns
    -------
    scale_star : float
        Scaling factor to use for the budget design.
    """
    N_total = len(df_scored)
    target_n = budget_frac * N_total

    # Initial lower and upper bounds for scale
    # Lower scale -> tighter MOE -> larger n; we search in log-space-ish
    scale_lo, scale_hi = 0.5, 5.0

    for it in range(max_iter):
        scale_mid = 0.5 * (scale_lo + scale_hi)
        plan_mid = build_sampling_plan(
            df=df_scored,
            strata_cols=strata_cols,
            band_to_moe_base=band_to_moe_base,
            scale=scale_mid,
            prob_col=prob_col,
            label_col=label_col,
            band_col=band_col,
            ambig_col=ambig_col,
            ambig_factor= ambig_factor
        )
        n_mid = total_sample_size(plan_mid)
        frac_mid = n_mid / N_total

        # Debug if needed:
        # print(f"Iter {it}: scale={scale_mid:.3f}, frac={frac_mid:.3%}")

        if abs(frac_mid - budget_frac) < tol:
            return scale_mid

        if frac_mid > budget_frac:
            # sampling too many: increase MOE -> increase scale
            scale_lo = scale_mid
        else:
            # sampling too few: decrease MOE -> decrease scale
            scale_hi = scale_mid

    # Return mid even if tolerance not hit
    return scale_mid

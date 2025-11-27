# ============================================================
# 3. Hypergeometric sample size, strata, and design helpers
# ============================================================

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def hypergeo_sample_size(
    N: int,
    p0: float,
    moe: float,
    z: float = 1.96,
    min_n: int = 0,
    max_n: Optional[int] = None,
) -> int:
    """
    Compute sample size n for a finite population (hypergeometric) such that
    the approximate 95% margin of error for p_hat is about `moe`.

    Uses the standard formula with Normal approximation:

      Var(p_hat) = p(1-p) (N - n) / [n (N - 1)]
      w ≈ z * sqrt(Var(p_hat))

    Solving for n gives:

      n = z^2 p(1-p) N / [ z^2 p(1-p) + w^2 (N-1) ]

    Parameters
    ----------
    N : int
        Population size in the stratum.
    p0 : float
        Design-stage guess for the true proportion.
    moe : float
        Target half-width (margin of error), in absolute probability units.
    z : float
        Normal quantile for desired coverage (1.96 ~ 95%).
    min_n : int
        Minimum sample size per stratum.
    max_n : int or None
        Maximum sample size per stratum. If None, <= N.

    Returns
    -------
    n : int
        Integer sample size within [min_n, max_n].
    """
    if N <= 0 or moe <= 0:
        return 0

    # Truncate p to avoid degeneracy
    p = max(1e-6, min(1 - 1e-6, p0))
    numerator = (z**2) * p * (1.0 - p) * N
    denominator = (z**2) * p * (1.0 - p) + (moe**2) * (N - 1)
    n = numerator / denominator
    n = int(math.ceil(n))

    if max_n is None or max_n > N:
        max_n = N
    n = max(min_n, min(n, max_n))
    return n


def compute_strata_summary(
    df: pd.DataFrame,
    strata_cols: List[str],
    prob_col: str = "match_probability",
    label_col: str = "true_match",
) -> pd.DataFrame:
    """
    Compute per-stratum summary:
      - N_s: population size
      - true_M_s: number of true matches
      - true_p_s: true stratum match rate
      - p_model_s: mean model match_probability in stratum
    """
    g = df.groupby(strata_cols)

    # population size per stratum
    N_s = g.size().rename("N_s")

    # true matches and true rate
    true_M_s = g[label_col].sum().rename("true_M_s")
    true_p_s = g[label_col].mean().rename("true_p_s")

    # mean model probability
    p_model_s = g[prob_col].mean().rename("p_model_s")

    grp = pd.concat(
        [N_s, true_M_s, true_p_s, p_model_s],
        axis=1
    ).reset_index()

    return grp


def build_sampling_plan(
    df: pd.DataFrame,
    strata_cols: List[str],
    band_to_moe_base: Dict[str, float],
    scale: float,
    prob_col: str = "match_probability",
    label_col: str = "true_match",
    band_col: str = "band_10",
    ambig_col: str = "ambig_bin",
    ambig_factor: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build a sampling plan for a given design scale, including:
      - stratum population counts
      - design-stage p0_s (from model)
      - ambiguity-aware MOE (band base * ambig_factor[bin] * scale)
      - hypergeometric sample size n_s

    We prioritise high-ambiguity strata by giving them smaller MOE
    via `ambig_factor`:

        moe_s = band_to_moe_base[band] * ambig_factor[ambig_bin] * scale

    Parameters
    ----------
    df : DataFrame
        Pair table with band_10, ambig_bin, prob_col, label_col, etc.
    strata_cols : list of str
        Columns defining strata (e.g. [band_10, ambig_bin, pattern, gender]).
    band_to_moe_base : dict
        Mapping band label -> baseline MOE (unscaled).
    scale : float
        Global scaling factor applied to all MOEs.
    prob_col : str
        Column with model match_probability.
    label_col : str
        Column with truth label.
    band_col : str
        Band column name.
    ambig_col : str
        Ambiguity bin column name (e.g. 'ambig_bin').
    ambig_factor : dict or None
        Mapping from ambiguity bin -> multiplicative factor.
        Higher ambiguity bins should typically have *smaller* factors
        (tighter MOE => larger n_s). If None, factor 1.0 is used
        for all ambiguity bins.

    Returns
    -------
    plan : DataFrame
        One row per stratum with columns:
          strata_cols, N_s, true_p_s, p_model_s, moe_s, n_s.
    """

    # Aggregate strata-level summaries
    strata = compute_strata_summary(
        df,
        strata_cols=strata_cols,
        prob_col=prob_col,
        label_col=label_col,
    )

    # If no ambiguity mapping provided, default to 1.0 for all bins
    if ambig_factor is None:
        ambig_factor = {}

    # Design-stage p0_s from model, truncated to [0.01, 0.99]
    p0 = np.clip(strata["p_model_s"].to_numpy(), 0.01, 0.99)

    moe_list: List[float] = []
    n_list: List[int] = []

    for idx, row in strata.iterrows():
        band = row[band_col]
        ambig = row[ambig_col]
        N_s = int(row["N_s"])

        base_moe = band_to_moe_base.get(band, 0.05)  # fallback if band missing
        # Learned ambiguity factor; default 1.0 if bin not in mapping
        factor_ambig = ambig_factor.get(ambig, 1.0)

        moe_s = base_moe * factor_ambig * scale

        n_s = hypergeo_sample_size(
            N=N_s,
            p0=p0[idx],
            moe=moe_s,
            z=1.96,
            min_n=0,
            max_n=N_s,
        )

        moe_list.append(moe_s)
        n_list.append(n_s)

    strata["moe_s"] = moe_list
    strata["n_s"] = n_list

    return strata


def total_sample_size(plan: pd.DataFrame) -> int:
    """Compute total planned sample size from a sampling plan."""
    return int(plan["n_s"].sum())

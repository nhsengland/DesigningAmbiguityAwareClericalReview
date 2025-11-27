# ============================================================
# 2. Matchability, conditional perplexity, bands, patterns, truth
# ============================================================
"""

Step 2 of the linkage QA pipeline:
  - Compute record-level matchability and conditional perplexity
    from Splink outputs (weights or probabilities) using an
    explicit "no-match" state.
  - Enrich the pair table with:
        * matchability
        * cond_perplexity (per-record conditional perplexity)
        * 10-band discretisation of match_probability (band_10)
        * comparison pattern code built from gamma levels
        * true_match label from cluster_l == cluster_r
        * gender_l as subgroup column

This file assumes that:

  - df_pred_raw is the Splink prediction table with columns:
        unique_id_l, unique_id_r,
        match_weight, match_probability,
        cluster_l, cluster_r,
        gender_l (from additional_columns_to_retain),
        and gamma-level columns (ending with "_gamma").

Usage from a notebook:

    from step2_uncertainty_bands_patterns import (
        prepare_pairs_with_uncertainty,
    )

    df_pred, record_metrics, gender_col = prepare_pairs_with_uncertainty(df_pred_raw)

"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# 2.1  Core entropy/perplexity utilities
# --------------------------------------------------------------------

def _entropy_bits(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    Shannon entropy in bits for a discrete distribution.

    Parameters
    ----------
    p : array-like
        Non-negative weights or probabilities.
    eps : float
        Numerical floor to avoid log(0).

    Returns
    -------
    H : float
        Entropy in bits (>= 0).
    """
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return 0.0

    p = np.clip(p / s, eps, 1.0 - eps)
    p /= p.sum()

    # log base 2
    H = -(p * (np.log(p) / np.log(2.0))).sum()
    return float(max(H, 0.0))


def compute_perplexity_metrics_from_splink(
    df_edges: pd.DataFrame,
    *,
    source_col: str = "unique_id_l",
    candidate_col: str = "unique_id_r",
    weight_col: str = "match_weight",
    prob_col: Optional[str] = None,
    prior_default_odds: float = 1.0,
    topk: Optional[int] = None,
    min_cum_mass: Optional[float] = None,   # e.g. 0.99
    return_edge_probs: bool = True,
    all_source_ids: Optional[pd.Series] = None,
    eps: float = 1e-12,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Compute record-level matchability and perplexity metrics from Splink outputs.

    For each left record (source_col = sid) with candidates j:

      - Let O_ij be odds vs null for each candidate.
      - Let O_0 be odds for the "no-match" state (prior_default_odds,
        possibly rescaled for numerical stability).
      - Define:

            Z      = O_0 + sum_j O_ij
            p_null = O_0 / Z
            p_i    = O_ij / Z              (joint over {candidates + null})
            matchability = 1 - p_null

      - Conditional over candidates given "some match exists":

            p_cond(j) = p_i(j) / matchability
            H_cond    = entropy_bits(p_cond)
            perp_cond = 2 ** H_cond

      - Also compute entropy/perplexity over {candidates + null}:

            H_all, perp_all.

    Optional trimming:
      - Use `topk` and/or `min_cum_mass` to keep only the strongest
        candidates per record before normalisation, based on either
        match_probability or match_weight.

    Parameters
    ----------
    df_edges : DataFrame
        Pair-level table with at least [source_col, candidate_col,
        weight_col] and optionally prob_col.
    source_col : str
        Column name for left/anchor record id.
    candidate_col : str
        Column name for right/candidate record id.
    weight_col : str
        Column name for FS match weight (base 2).
    prob_col : str or None
        Column with pairwise match_probability. If provided, odds are
        derived from this; otherwise we exponentiate weights.
    prior_default_odds : float
        Unnormalised odds for the no-match state. Controls how much prior
        mass is available for "no match" vs candidates.
    topk : int or None
        If set, keep at most this many top candidates per record.
    min_cum_mass : float or None
        If set (e.g. 0.99), keep the smallest number of strongest
        candidates whose odds sum accounts for at least this proportion
        of candidate odds.
    return_edge_probs : bool
        If True, return an enriched edge table with p_i and p_cond.
    all_source_ids : Series or None
        If provided, ensures records with zero candidates are included
        in the output record_metrics with matchability=0.
    eps : float
        Numerical floor to avoid division by zero and log(0).

    Returns
    -------
    record_metrics : DataFrame
        One row per source record with columns:

          [source_col, p_null, matchability,
           H_cond, perp_cond, H_all, perp_all,
           top1, margin, n_candidates, n_kept, trimmed, zero_candidate]

    edges_enriched : DataFrame or None
        If return_edge_probs is True, pair table with p_i and p_cond
        for the kept candidates; else None.
    """
    df = df_edges.copy()
    rec_rows: List[Dict] = []
    edge_rows: List[pd.DataFrame] = []

    # raw candidate counts per record
    if len(df) > 0:
        n_cand_map = df.groupby(source_col, sort=False)[candidate_col].size()
    else:
        n_cand_map = pd.Series(dtype=int)

    for sid, g in df.groupby(source_col, sort=False):
        g = g.copy()

        # odds vs null
        if prob_col is not None:
            q = g[prob_col].astype(float).clip(eps, 1.0 - eps)
            Oi = (q / (1.0 - q)).to_numpy(float)
            O0_scaled = prior_default_odds
            order_strength = Oi  # for trimming
        else:
            M = g[weight_col].astype(float).to_numpy()
            mmax = np.max(M) if M.size else 0.0
            Oi = np.power(2.0, M - mmax)
            O0_scaled = prior_default_odds * np.power(2.0, -mmax)
            order_strength = M

        # optional trimming
        keep_idx = np.arange(len(Oi))
        if len(Oi) and (topk is not None or min_cum_mass is not None):
            order = np.argsort(-order_strength)  # descending
            Oi_sorted = Oi[order]
            k = len(Oi_sorted)
            if min_cum_mass is not None:
                cs = np.cumsum(Oi_sorted)
                k = min(k, np.searchsorted(cs / cs[-1], min_cum_mass, side="left") + 1)
            if topk is not None:
                k = min(k, int(topk))
            keep_ord = order[:k]
            keep_idx = np.sort(keep_ord)
            Oi = Oi[keep_idx]

        # normalisation with no-match state
        Z = O0_scaled + Oi.sum()
        if Z > 0:
            p_null = O0_scaled / Z
            p_i = Oi / Z
        else:
            p_null = 1.0
            p_i = np.zeros_like(Oi)

        matchability = 1.0 - p_null

        # conditional over candidates, given "some match"
        if matchability > eps and p_i.size:
            p_cond = p_i / matchability
            H_cond = _entropy_bits(p_cond, eps)
            perp_cond = float(2.0 ** H_cond)

            ord_pc = np.sort(p_cond)[::-1]
            top1 = float(ord_pc[0])
            margin = float(ord_pc[0] - ord_pc[1]) if ord_pc.size >= 2 else float("nan")
        else:
            p_cond = np.zeros_like(p_i)
            H_cond, perp_cond = 0.0, 1.0
            top1, margin = 0.0, float("nan")

        # entropy over {candidates + null}
        if p_i.size:
            H_all = _entropy_bits(np.r_[p_i, p_null], eps)
        else:
            H_all = _entropy_bits([p_null], eps)
        perp_all = float(2.0 ** H_all)

        rec_rows.append(
            {
                source_col: sid,
                "p_null": float(p_null),
                "matchability": float(matchability),
                "H_cond": H_cond,
                "perp_cond": perp_cond,
                "H_all": H_all,
                "perp_all": perp_all,
                "top1": top1,
                "margin": margin,
                "n_candidates": int(n_cand_map.get(sid, 0)),
                "n_kept": int(len(Oi)),
                "trimmed": bool(len(Oi) != int(n_cand_map.get(sid, 0))),
            }
        )

        if return_edge_probs:
            g_keep = g.iloc[keep_idx].copy()
            g_keep["p_i"] = p_i
            g_keep["p_cond"] = p_cond
            edge_rows.append(g_keep)

    record_metrics = pd.DataFrame(rec_rows)

    # Include zero-candidate left records if requested
    if all_source_ids is not None:
        all_ids = pd.Series(all_source_ids, name=source_col).drop_duplicates()
        record_metrics = all_ids.to_frame().merge(
            record_metrics, on=source_col, how="left"
        )
        fills = {
            "p_null": 1.0,
            "matchability": 0.0,
            "H_cond": 0.0,
            "perp_cond": 1.0,
            "H_all": 0.0,
            "perp_all": 1.0,
            "top1": 0.0,
            "margin": np.nan,
            "n_candidates": 0,
            "n_kept": 0,
            "trimmed": False,
        }
        for c, v in fills.items():
            record_metrics[c] = record_metrics[c].fillna(v)
        record_metrics["zero_candidate"] = record_metrics["n_candidates"].eq(0)
    else:
        record_metrics["zero_candidate"] = record_metrics["n_candidates"].eq(0)

    edges_enriched = (
        pd.concat(edge_rows, ignore_index=True)
        if (return_edge_probs and edge_rows)
        else None
    )

    return record_metrics, edges_enriched


# --------------------------------------------------------------------
# 2.2  Bands, patterns, truth
# --------------------------------------------------------------------

def build_match_pattern(
    df_pairs: pd.DataFrame,
    gamma_concat_col: str = "gamma_concat",
    gamma_prefix: str = "gamma_",
) -> pd.Series:
    """
    Build a compact pattern code from Splink gamma-level columns.

    Priority:
      1) If `gamma_concat_col` exists (e.g. 'gamma_concat'), use it directly.
      2) Otherwise, find all columns whose names start with `gamma_prefix`
         (default 'gamma_'), sort them lexicographically, cast to int,
         and concatenate as a comma-separated string.

    Examples of resulting codes:
      "2,1,1,0,3" etc.

    Parameters
    ----------
    df_pairs : DataFrame
        Pair-level table with gamma columns (from Splink).
    gamma_concat_col : str
        Name of pre-combined gamma pattern column if present.
    gamma_prefix : str
        Prefix used to identify individual gamma columns, default 'gamma_'.

    Returns
    -------
    pattern : Series of str
        Pattern code per row, suitable for grouping / stratification.
    """

    # Case 1: Splink gave us a ready-made gamma_concat
    if gamma_concat_col in df_pairs.columns:
        return df_pairs[gamma_concat_col].astype(str)

    # Case 2: construct from individual gamma_* columns
    gamma_cols = sorted(
        [c for c in df_pairs.columns if c.startswith(gamma_prefix)]
    )
    if not gamma_cols:
        raise ValueError(
            f"No {gamma_prefix}* or {gamma_concat_col} columns found in pair table."
        )

    pattern = (
        df_pairs[gamma_cols]
        .astype("int64")     # safe for typical Splink gamma outputs
        .astype(str)
        .agg(",".join, axis=1)
    )
    return pattern



def add_bands_patterns_truth(
    df_pred_raw: pd.DataFrame,
    prob_col: str = "match_probability",
    band_col: str = "band_10",
) -> Tuple[pd.DataFrame, str]:
    """
    Enrich the raw Splink prediction table with:

      - band_10: decile-based bands on match_probability (b1..b10)
      - match_pattern: concatenated gamma pattern
      - true_match: 1_{cluster_l == cluster_r}
      - gender_l: demographic column name returned for downstream use

    Parameters
    ----------
    df_pred_raw : DataFrame
        Raw Splink prediction output.
    prob_col : str
        Column with match_probability.
    band_col : str
        Name of the band column to create.

    Returns
    -------
    df_pred : DataFrame
        Enriched pair table.
    gender_col : str
        Name of the gender column to use (gender_l).
    """
    df_pred = df_pred_raw.copy()

    # 1) Banding by probability deciles
    probs = df_pred[prob_col].astype(float)
    # Use qcut with duplicates='drop' to be robust to flat distributions
    df_pred[band_col] = pd.qcut(
        probs,
        q=10,
        labels=[f"b{i}" for i in range(1, 11)],
        duplicates="drop",
    )

    # 2) Comparison pattern
    df_pred["match_pattern"] = build_match_pattern(df_pred)

    # 3) Gender column
    gender_col = "gender_l"
    if gender_col not in df_pred.columns:
        raise ValueError(
            f"{gender_col!r} not found in prediction output. "
            "Make sure 'gender' was included in additional_columns_to_retain."
        )

    # 4) Underlying truth from cluster_l / cluster_r
    if "cluster_l" not in df_pred.columns or "cluster_r" not in df_pred.columns:
        raise ValueError(
            "cluster_l / cluster_r not found. "
            "Make sure 'cluster' was retained as an additional column."
        )
    df_pred["true_match"] = (df_pred["cluster_l"] == df_pred["cluster_r"]).astype(int)

    return df_pred, gender_col


# --------------------------------------------------------------------
# 2.3  High-level wrapper for step 2
# --------------------------------------------------------------------

def prepare_pairs_with_uncertainty(
    df_pred_raw: pd.DataFrame,
    *,
    prob_col: str = "match_probability",
    weight_col: str = "match_weight",
    band_col: str = "band_10",
    prior_default_odds: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    High-level wrapper for step 2:

      - Take raw Splink predictions.
      - Compute record-level matchability and conditional perplexity
        using compute_perplexity_metrics_from_splink.
      - Add bands, comparison patterns, and true_match.

    Parameters
    ----------
    df_pred_raw : DataFrame
        Raw Splink prediction table from linker.inference.predict().
    prob_col : str
        Column name for match_probability.
    weight_col : str
        Column name for match_weight (FS weight).
    band_col : str
        Name of the band column to create (default 'band_10').
    prior_default_odds : float
        Prior odds for the no-match state in the perplexity calculation.

    Returns
    -------
    df_pred : DataFrame
        Pair-level table enriched with:

          band_10, match_pattern, true_match,
          matchability, cond_perplexity (perp_cond).

    record_metrics : DataFrame
        Record-level table with:

          [unique_id_l, p_null, matchability, H_cond, perp_cond,
           H_all, perp_all, top1, margin, n_candidates, ...]

    gender_col : str
        Name of the gender column (gender_l).
    """
    # 1) Record-level uncertainty metrics
    record_metrics, _ = compute_perplexity_metrics_from_splink(
        df_edges=df_pred_raw,
        source_col="unique_id_l",
        candidate_col="unique_id_r",
        weight_col=weight_col,
        prob_col=prob_col,
        prior_default_odds=prior_default_odds,
        topk=None,
        min_cum_mass=None,
        return_edge_probs=False,
        all_source_ids=df_pred_raw["unique_id_l"],
    )

    # 2) Merge matchability + cond_perplexity back to pairs
    rm = record_metrics[["unique_id_l", "matchability", "perp_cond"]].rename(
        columns={"perp_cond": "cond_perplexity"}
    )

    df_pred = df_pred_raw.merge(rm, on="unique_id_l", how="left")

    # 3) Add bands, patterns, truth, gender
    df_pred, gender_col = add_bands_patterns_truth(
        df_pred, prob_col=prob_col, band_col=band_col
    )

    return df_pred, record_metrics, gender_col

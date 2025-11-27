# ============================================================
# 5. Draw samples, evaluate error, and representation
# ============================================================

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def draw_clerical_sample(
    df: pd.DataFrame,
    plan: pd.DataFrame,
    strata_cols: List[str],
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Draw a clerical sample according to the sampling plan.

    Parameters
    ----------
    df : DataFrame
        Full pair table.
    plan : DataFrame
        Sampling plan with strata_cols + 'n_s'.
    strata_cols : list of str
        Strata columns to join on (e.g. band_10, ambig_bin, pattern, gender).
    random_state : int or None
        Seed for reproducible sampling.

    Returns
    -------
    df_sample : DataFrame
        Subset of df representing the clerical sample. Carries 'n_s'
        from the plan for reference, but sampling is performed with
        simple random sampling without replacement within each stratum.
    """
    if random_state is not None:
        rs = np.random.RandomState(random_state)
    else:
        rs = np.random

    df = df.copy()
    plan_key = strata_cols + ["n_s"]
    df = df.merge(plan[plan_key], on=strata_cols, how="left")
    df["n_s"] = df["n_s"].fillna(0).astype(int)

    samples = []
    for _, g in df.groupby(strata_cols):
        n_s = int(g["n_s"].iloc[0])
        if n_s <= 0:
            continue
        if n_s >= len(g):
            samples.append(g)
        else:
            idx = rs.choice(g.index.to_numpy(), size=n_s, replace=False)
            samples.append(g.loc[idx])

    if not samples:
        return df.head(0).copy()

    df_sample = pd.concat(samples, axis=0).reset_index(drop=True)
    return df_sample


def estimate_global_and_band_rates(
    df: pd.DataFrame,
    df_sample: pd.DataFrame,
    band_col: str = "band_10",
    label_col: str = "true_match",
) -> Tuple[float, Dict[str, float]]:
    """
    Estimate global and band-level match rates from a clerical sample.

    For band-level estimates we use the simple band-wise sample
    proportion:

        p_hat_b = (sum_{sample in band b} Y) / n_b

    and then aggregate to global via:

        p_hat_global = sum_b N_b * p_hat_b / N_total

    Under SRSWOR within bands, this is an unbiased estimator of the
    band and global match rates.

    Parameters
    ----------
    df : DataFrame
        Full pair table (population).
    df_sample : DataFrame
        Clerical sample drawn by draw_clerical_sample.
    band_col : str
        Band column name (e.g. 'band_10').
    label_col : str
        Ground-truth match indicator.

    Returns
    -------
    p_hat_global : float
        Estimated global match rate.
    p_hat_band : dict
        Mapping band -> estimated match rate p_hat_b.
    """
    N_total = len(df)

    # Population band sizes
    N_b = df.groupby(band_col, as_index=False)[label_col].size()
    N_b = N_b.rename(columns={"size": "N_b"})

    # Sample counts and matches by band
    sample_counts = (
        df_sample
        .groupby(band_col, as_index=False)
        .agg(
            n_b=(label_col, "size"),
            X_b=(label_col, "sum"),
        )
    )

    band_stats = N_b.merge(sample_counts, on=band_col, how="left")
    band_stats[["n_b", "X_b"]] = band_stats[["n_b", "X_b"]].fillna(0)

    band_stats["p_hat_b"] = np.where(
        band_stats["n_b"] > 0,
        band_stats["X_b"] / band_stats["n_b"],
        0.0,
    )

    p_hat_band = dict(zip(band_stats[band_col], band_stats["p_hat_b"]))
    p_hat_global = (band_stats["N_b"] * band_stats["p_hat_b"]).sum() / N_total

    return p_hat_global, p_hat_band


def run_replicates_design(
    df_pred: pd.DataFrame,
    band_to_moe_base: Dict[str, float],
    scale: float,
    strata_cols: List[str],
    band_col: str = "band_10",
    ambig_col: str = "ambig_bin",
    prob_col: str = "match_probability",
    label_col: str = "true_match",
    ambig_factor: Optional[Dict[str, float]] = None,
    n_reps: int = 5,
    seed_base: int = 100,
) -> Dict[str, pd.DataFrame]:
    """
    Run replicate evaluation for a given design scale:
      - build an ambiguity-aware sampling plan
      - for each replicate, draw a clerical sample
      - compute global and band-level errors relative to truth.

    Parameters
    ----------
    df_pred : DataFrame
        Pair table with scores, truth, and strata columns.
    band_to_moe_base : dict
        Mapping band -> baseline MOE.
    scale : float
        Global MOE scaling factor (e.g. 1.0 baseline, scale_star for budget).
    strata_cols : list of str
        Strata columns (e.g. [band_10, ambig_bin, pattern, gender]).
    band_col : str
        Name of band column (default 'band_10').
    ambig_col : str
        Name of ambiguity-bin column (default 'ambig_bin').
    prob_col : str
        Column with model match_probability.
    label_col : str
        Column with true_match indicator.
    ambig_factor : dict or None
        Mapping from ambiguity bin -> MOE factor. If None, defaults to
        1.0 for all bins inside build_sampling_plan.
    n_reps : int
        Number of replicate samples to draw.
    seed_base : int
        Base seed; replicate r uses seed = seed_base + r.

    Returns
    -------
    result : dict
        {
          "global_errors": DataFrame,
          "band_reliability": DataFrame,
          "plan": DataFrame
        }
    """
    plan = build_sampling_plan(
        df=df_pred,
        strata_cols=strata_cols,
        band_to_moe_base=band_to_moe_base,
        scale=scale,
        prob_col=prob_col,
        label_col=label_col,
        band_col=band_col,
        ambig_col=ambig_col,
        ambig_factor=ambig_factor,
    )

    N_total = len(df_pred)
    global_errors = []
    band_error_rows = []
    ambig_comp_rows = []

    # True rates
    true_global = df_pred[label_col].mean()
    true_band = (
        df_pred.groupby(band_col, as_index=False)[label_col]
        .mean()
        .rename(columns={label_col: "true_match_rate"})
    )
    true_band_dict = dict(zip(true_band[band_col], true_band["true_match_rate"]))

    # Population band × ambiguity counts (constant across replicates)
    pop_band_ambig = (
        df_pred
        .groupby([band_col, ambig_col], as_index=False)[label_col]
        .size()
        .rename(columns={"size": "N_pop"})
    )

    for r in range(n_reps):
        seed = seed_base + r
        df_sample = draw_clerical_sample(
            df_pred,
            plan=plan,
            strata_cols=strata_cols,
            random_state=seed,
        )

        # ----- error metrics -----
        p_hat_global, p_hat_band = estimate_global_and_band_rates(
            df=df_pred,
            df_sample=df_sample,
            band_col=band_col,
            label_col=label_col,
        )

        global_abs_error_pp = abs(p_hat_global - true_global) * 100.0

        global_errors.append(
            {
                "rep": r,
                "n_sample": len(df_sample),
                "p_hat_global": p_hat_global,
                "true_global": true_global,
                "global_abs_error_pp": global_abs_error_pp,
            }
        )

        # Band-level errors for this replicate
        for b, true_p in true_band_dict.items():
            est_p = p_hat_band.get(b, 0.0)
            band_error_rows.append(
                {
                    "rep": r,
                    "band_10": b,
                    "true_match_rate": true_p,
                    "abs_error_pp": abs(est_p - true_p) * 100.0,
                }
            )

        # ----- band × ambiguity composition for this replicate -----
        samp_band_ambig = (
            df_sample
            .groupby([band_col, ambig_col], as_index=False)[label_col]
            .size()
            .rename(columns={"size": "n_sample"})
        )

        # merge with population counts so we know N_pop even when n_sample=0
        ba_merge = pop_band_ambig.merge(
            samp_band_ambig,
            on=[band_col, ambig_col],
            how="left",
        )
        ba_merge["n_sample"] = ba_merge["n_sample"].fillna(0).astype(int)
        ba_merge["rep"] = r

        ambig_comp_rows.extend(ba_merge.to_dict(orient="records"))

    global_errors_df = pd.DataFrame(global_errors)

    # Summarise band-level reliability across reps
    band_err = pd.DataFrame(band_error_rows)
    g_band = band_err.groupby("band_10")

    true_match_rate = g_band["true_match_rate"].mean().rename("true_match_rate")
    mean_abs_error_pp = g_band["abs_error_pp"].mean().rename("mean_abs_error_pp")
    max_abs_error_pp = g_band["abs_error_pp"].max().rename("max_abs_error_pp")
    q95_abs_error_pp = g_band["abs_error_pp"].quantile(0.95).rename("q95_abs_error_pp")

    band_reliability = pd.concat(
        [true_match_rate, mean_abs_error_pp, max_abs_error_pp, q95_abs_error_pp],
        axis=1
    ).reset_index()

    band_reliability["true_match_rate_pp"] = (
        band_reliability["true_match_rate"] * 100.0
    )

    ambig_comp = pd.DataFrame(ambig_comp_rows)

    return {
        "global_errors": global_errors_df,
        "band_reliability": band_reliability,
        "band_errors_long": band_err,
        "plan": plan,
        "ambig_comp": ambig_comp,
    }



def l1_distance_pct(
    pop_counts: pd.Series,
    sample_counts: pd.Series
) -> float:
    """
    Compute L1 distance between two empirical distributions, in percentage
    of mass to move (0–100).

    Parameters
    ----------
    pop_counts : Series
        Counts per category in the population.
    sample_counts : Series
        Counts per category in the sample.

    Returns
    -------
    L1_pct : float
        50 * sum_k |p_pop(k) - p_samp(k)|.
    """
    # Align categories
    all_cats = sorted(set(pop_counts.index) | set(sample_counts.index))
    pop = pop_counts.reindex(all_cats, fill_value=0).astype(float)
    samp = sample_counts.reindex(all_cats, fill_value=0).astype(float)

    if pop.sum() == 0 or samp.sum() == 0:
        return 0.0

    p_pop = pop / pop.sum()
    p_samp = samp / samp.sum()

    L1 = np.abs(p_pop - p_samp).sum()
    return 50.0 * L1  # convert to [0, 100]


def collect_representation_over_reps(
    df_pred: pd.DataFrame,
    band_to_moe_base: Dict[str, float],
    scale: float,
    strata_cols: List[str],
    band_col: str = "band_10",
    pattern_col: str = "match_pattern",
    gender_col: str = "gender_l",
    ambig_col: str = "ambig_bin",
    ambig_factor: Optional[Dict[str, float]] = None,
    n_reps: int = 5,
    seed_base: int = 2000,
) -> Tuple[pd.DataFrame, float]:
    """
    Run summarise_representation_for_design n_reps times, and summarise
    mean/sd of L1 distances per band. Returns:

      - summary: per-band mean/sd for pattern, gender, ambiguity
      - mean_n_sample: mean total sample size over reps
    """
    all_band_stats = []
    total_sample_by_rep = []

    for r in range(n_reps):
        bc, plan, n_sample, frac_sample = summarise_representation_for_design(
            df_pred=df_pred,
            band_to_moe_base=band_to_moe_base,
            scale=scale,
            strata_cols=strata_cols,
            band_col=band_col,
            pattern_col=pattern_col,
            gender_col=gender_col,
            ambig_col=ambig_col,
            ambig_factor=ambig_factor,
            random_state=seed_base + r,
        )
        bc = bc.copy()
        bc["rep"] = r
        all_band_stats.append(bc)
        total_sample_by_rep.append(n_sample)

    all_band_stats = pd.concat(all_band_stats, ignore_index=True)

    g = all_band_stats.groupby("band_10")
    summary = (
        g.agg(
            L1_pattern_mean=("L1_pattern_pct", "mean"),
            L1_pattern_sd=("L1_pattern_pct", "std"),
            L1_gender_mean=("L1_gender_pct", "mean"),
            L1_gender_sd=("L1_gender_pct", "std"),
            L1_ambig_mean=("L1_ambig_pct", "mean"),
            L1_ambig_sd=("L1_ambig_pct", "std"),
        )
        .reset_index()
        .sort_values("band_10")
    )

    mean_n_sample = float(np.mean(total_sample_by_rep))
    return summary, mean_n_sample


def summarise_representation_for_design(
    df_pred: pd.DataFrame,
    band_to_moe_base: Dict[str, float],
    scale: float,
    strata_cols: List[str],
    band_col: str = "band_10",
    pattern_col: str = "match_pattern",
    gender_col: str = "gender_l",
    ambig_col: str = "ambig_bin",
    prob_col: str = "match_probability",
    label_col: str = "true_match",
    ambig_factor: Optional[Dict[str, float]] = None,
    random_state: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, float]:
    """
    For a single draw under a given design, summarise band-level representation
    in terms of:
      - sample rate
      - L1 distance (pattern, gender, ambiguity)

    Parameters
    ----------
    df_pred : DataFrame
        Full pair table.
    band_to_moe_base : dict
        Band -> baseline MOE.
    scale : float
        Design scale.
    strata_cols : list of str
        Strata columns (e.g. [band_10, ambig_bin, pattern, gender]).
    band_col, pattern_col, gender_col, ambig_col : str
        Column names for band, pattern, gender, ambiguity bins.
    prob_col : str
        Column with model match_probability (for planning).
    label_col : str
        Column with true_match indicator (for planning summaries).
    ambig_factor : dict or None
        Mapping from ambiguity bin -> MOE factor. If None, defaults to 1.0
        for all bins inside build_sampling_plan.
    random_state : int
        Seed for sampling.

    Returns
    -------
    band_comp : DataFrame
        Band-level representation metrics.
    plan : DataFrame
        Sampling plan used.
    n_total_sample : int
        Total sample size in this draw.
    frac_sample : float
        Sample fraction.
    """
    # 1) Build the ambiguity-aware sampling plan for this design
    plan = build_sampling_plan(
        df=df_pred,
        strata_cols=strata_cols,
        band_to_moe_base=band_to_moe_base,
        scale=scale,
        prob_col=prob_col,
        label_col=label_col,
        band_col=band_col,
        ambig_col=ambig_col,
        ambig_factor=ambig_factor,
    )

    # 2) Draw a single clerical sample according to the plan
    df_sample = draw_clerical_sample(
        df=df_pred,
        plan=plan,
        strata_cols=strata_cols,
        random_state=random_state,
    )

    N_total = len(df_pred)
    n_sample = len(df_sample)
    frac_sample = n_sample / N_total if N_total > 0 else 0.0

    # 3) Band-level representation metrics
    band_stats = []

    for band, g_pop in df_pred.groupby(band_col):
        g_samp = df_sample[df_sample[band_col] == band]

        N_pop = len(g_pop)
        N_samp = len(g_samp)
        sample_rate = N_samp / N_pop if N_pop > 0 else 0.0

        # Pattern L1
        L1_pattern_pct = l1_distance_pct(
            g_pop[pattern_col].value_counts(),
            g_samp[pattern_col].value_counts(),
        )

        # Gender L1
        L1_gender_pct = l1_distance_pct(
            g_pop[gender_col].value_counts(),
            g_samp[gender_col].value_counts(),
        )

        # Ambiguity-bin L1
        L1_ambig_pct = l1_distance_pct(
            g_pop[ambig_col].value_counts(),
            g_samp[ambig_col].value_counts(),
        )

        band_stats.append(
            {
                band_col: band,
                "N_pop": N_pop,
                "N_sample": N_samp,
                "sample_rate": sample_rate,
                "L1_pattern_pct": L1_pattern_pct,
                "L1_gender_pct": L1_gender_pct,
                "L1_ambig_pct": L1_ambig_pct,
            }
        )

    band_comp = pd.DataFrame(band_stats).sort_values(band_col)

    return band_comp, plan, n_sample, frac_sample


def summarise_ambiguity_over_reps(
    ambig_comp: pd.DataFrame,
    band_col: str = "band_10",
    ambig_col: str = "ambig_bin",
) -> pd.DataFrame:
    """
    Summarise band × ambiguity-bin composition across replicates.

    For each (band, ambiguity_bin) we compute:
      - pop_prop: proportion of pairs in that bin in the population band
      - sample_prop_mean: mean proportion in clerical samples
      - sample_prop_sd:   SD across replicates
      - sample_prop_se:   SE across replicates

    Parameters
    ----------
    ambig_comp : DataFrame
        Output 'ambig_comp' from run_replicates_design:
        columns: rep, band_col, ambig_col, N_pop, n_sample.
    band_col : str
        Band column name.
    ambig_col : str
        Ambiguity bin column name.

    Returns
    -------
    summary : DataFrame
        One row per (band, ambiguity_bin) with population and
        sample proportions + error-bar statistics.
    """

    df = ambig_comp.copy()

    # ---- population proportions (constant across reps) ----
    pop = (
        df.groupby([band_col, ambig_col])["N_pop"]
        .first()
        .reset_index()
    )

    pop_band_tot = (
        pop.groupby(band_col)["N_pop"]
        .sum()
        .rename("N_b")
        .reset_index()
    )

    pop = pop.merge(pop_band_tot, on=band_col, how="left")
    pop["pop_prop"] = np.where(
        pop["N_b"] > 0,
        pop["N_pop"] / pop["N_b"],
        np.nan,
    )

    # ---- sample proportions per replicate ----
    samp = (
        df.groupby(["rep", band_col, ambig_col])["n_sample"]
        .sum()
        .reset_index()
    )

    band_rep_tot = (
        samp.groupby(["rep", band_col])["n_sample"]
        .sum()
        .rename("n_b_rep")
        .reset_index()
    )

    samp = samp.merge(band_rep_tot, on=["rep", band_col], how="left")
    samp["sample_prop"] = np.where(
        samp["n_b_rep"] > 0,
        samp["n_sample"] / samp["n_b_rep"],
        np.nan,
    )

    # ---- average & variability across replicates ----
    g = samp.groupby([band_col, ambig_col])["sample_prop"]

    summary = g.mean().rename("sample_prop_mean").to_frame()
    summary["sample_prop_sd"] = g.std(ddof=1)
    n_reps = samp["rep"].nunique()
    summary["sample_prop_se"] = summary["sample_prop_sd"] / np.sqrt(max(n_reps, 1))

    summary = summary.reset_index()

    # merge in population proportions
    summary = summary.merge(
        pop[[band_col, ambig_col, "pop_prop"]],
        on=[band_col, ambig_col],
        how="left",
    )

    return summary

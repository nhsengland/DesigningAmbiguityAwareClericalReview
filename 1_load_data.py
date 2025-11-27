import pandas as pd
import splink.comparison_library as cl

# Splink imports (DuckDB backend)
from splink import DuckDBAPI, Linker, SettingsCreator, block_on, splink_datasets

# ============================================================
# 1. Data loading, Splink model fitting, scoring
# ============================================================

def load_data() -> pd.DataFrame:
    """
    Load the synthetic historical_50k dataset from Splink.
    

    Returns
    -------
    df : DataFrame
        One row per record, with quasi-identifiers and 'cluster' truth.
    """
    df = splink_datasets.historical_50k
    # Ensure we have an explicit unique_id if not present
    df['gender'] = df['gender'].where(df['gender'].isin(['male', 'female']) | df['gender'].isna(), 
                                  'unknown')
    return df


def fit_splink_model(df: pd.DataFrame) -> Linker:
    """
    Fit a Splink Fellegi–Sunter model on the historical_50k dataset
    using DuckDB as the backend.

    Returns
    -------
    linker : splink.Linker
        Trained linkage model for dedupe_only.
    """
    db_api = DuckDBAPI()

    # Blocking rules for predictions
    blocking_rules = [
        block_on("substr(first_name,1,3)", "substr(surname,1,4)"),
        block_on("surname", "dob"),
        block_on("first_name", "dob"),
        block_on("postcode_fake", "first_name"),
        block_on("postcode_fake", "surname"),
        block_on("dob", "birth_place"),
        block_on("substr(postcode_fake,1,3)", "dob"),
        block_on("substr(postcode_fake,1,3)", "first_name"),
        block_on("substr(postcode_fake,1,3)", "surname"),
        block_on("substr(first_name,1,2)", "substr(surname,1,2)", "substr(dob,1,4)"),
    ]

    # Needed to apply term frequencies to first+surname comparison
    df = df.copy()
    df["first_name_surname_concat"] = df["first_name"] + " " + df["surname"]

    # Settings with comparisons
    settings = SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=blocking_rules,
        comparisons=[
            cl.ForenameSurnameComparison(
                "first_name",
                "surname",
                forename_surname_concat_col_name="first_name_surname_concat",
            ),
            cl.DateOfBirthComparison("dob", input_is_string=True),
            cl.PostcodeComparison("postcode_fake"),
            cl.ExactMatch("birth_place").configure(term_frequency_adjustments=True),
            cl.ExactMatch("occupation").configure(term_frequency_adjustments=True),
        ],
        retain_intermediate_calculation_columns=True,
        additional_columns_to_retain=["gender", "cluster"],  # keep for truth & subgroup
    )

    linker = Linker(df, settings, db_api=db_api)

    # Estimate P(two random records match) using strong-block training
    linker.training.estimate_probability_two_random_records_match(
        [
            block_on("first_name", "surname", "dob"),
            block_on("substr(first_name,1,2)", "surname", "substr(postcode_fake,1,2)"),
            block_on("dob", "postcode_fake"),
        ],
        recall=0.6,
    )

    # Estimate parameters via EM on a DOB block
    training_blocking_rule = block_on("dob")
    _ = linker.training.estimate_parameters_using_expectation_maximisation(
        training_blocking_rule, estimate_without_term_frequencies=True
    )

    return linker


def get_predictions(linker: Linker) -> pd.DataFrame:
    """
    Use the fitted linker to score all candidate pairs and return as pandas.

    Returns
    -------
    df_pred : DataFrame
        Pair-level table with match probabilities, comparison patterns,
        and retained additional columns (gender, cluster, etc.).
    """
    df_predict = linker.inference.predict()
    df_pred = df_predict.as_pandas_dataframe()  # all rows
    return df_pred


from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def learn_ambiguity_bins_from_distribution(
    rec_unc: pd.DataFrame,
    match_col: str = "matchability",
    cpp_col: str = "cond_perplexity",
    min_k: int = 3,
    max_k: int = 6,
    random_state: int = 42,
    max_cpp: float = 50.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Learn ambiguity bins via clustering in (matchability, log1p(cpp)) space.

    Steps:
      1. Work on record-level rec_unc (one row per unique_id_l).
      2. Transform cpp -> log1p(cpp) for stability.
      3. For k in [min_k, ..., max_k]:
           - run k-means
           - compute silhouette score
         pick k with best silhouette.
      4. Order clusters by median cpp (low -> high ambiguity).
      5. Create a categorical ambig_bin = 'a1', 'a2', ..., 'aK'
         where 'a1' is lowest ambiguity and 'aK' the highest.
      6. Learn an ambiguity factor per bin, decreasing smoothly from
         max_factor to min_factor as ambiguity increases.

    Returns
    -------
    rec_unc_out : DataFrame
        rec_unc with new column 'ambig_bin'.
    ambig_factor : dict
        Mapping ambig_bin -> factor to be used in MOE scaling.
    """

    # 1. Restrict to finite values and optionally cap cpp for stability
    rec = rec_unc.copy()
    mask = np.isfinite(rec[match_col]) & np.isfinite(rec[cpp_col])
    rec = rec[mask].copy()

    # Cap extreme cpp for the clustering step (they are rare and distort scale)
    rec["cpp_clipped"] = np.minimum(rec[cpp_col], max_cpp)

    # 2. Feature matrix: matchability + log1p(cpp)
    rec["log_cpp"] = np.log1p(rec["cpp_clipped"])
    X = np.vstack([rec[match_col].to_numpy(),
                   rec["log_cpp"].to_numpy()]).T

    # 3. Choose k by silhouette score
    best_k = None
    best_score = -np.inf
    best_labels = None

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        # Silhouette requires at least 2 clusters and fewer than samples
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(X):
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k = k
            best_score = score
            best_labels = labels

    if best_labels is None:
        # Fallback: single bin if clustering fails.
        rec_unc["ambig_bin"] = "a1"
        return rec_unc, {"a1": 1.0}

    rec["ambig_cluster"] = best_labels

    # 4. Order clusters by median cpp (low -> high ambiguity)
    cluster_median_cpp = (
        rec.groupby("ambig_cluster")[cpp_col]
        .median()
        .sort_values()
    )
    ordered_clusters = cluster_median_cpp.index.tolist()
    cluster_to_rank = {c: r for r, c in enumerate(ordered_clusters)}

    rec["ambig_rank"] = rec["ambig_cluster"].map(cluster_to_rank)

    # 5. Create ambig_bin labels a1, a2, ..., aK
    n_bins = len(ordered_clusters)
    rank_to_bin = {r: f"a{r+1}" for r in range(n_bins)}
    rec["ambig_bin"] = rec["ambig_rank"].map(rank_to_bin)

    # 6. Learn ambiguity factors: low ambiguity -> larger factor
    #    (looser MOE), high ambiguity -> smaller factor (tighter MOE).
    max_factor = 1.6
    min_factor = 0.7
    ranks = np.array(range(n_bins), dtype=float)
    factors = np.interp(
        ranks,
        [ranks.min(), ranks.max()],
        [max_factor, min_factor],
    )
    ambig_factor = {
        rank_to_bin[r]: float(factors[r])
        for r in range(n_bins)
    }

    # Map back to full rec_unc
    rec_unc_out = rec_unc.copy()
    rec_unc_out = rec_unc_out.merge(
        rec[["matchability", "cond_perplexity", "ambig_bin"]],
        on=[match_col, cpp_col],
        how="left",
    )

    # Any records not clustered (very rare) go to lowest-ambiguity bin
    rec_unc_out["ambig_bin"] = rec_unc_out["ambig_bin"].fillna("a1")

    return rec_unc_out, ambig_factor

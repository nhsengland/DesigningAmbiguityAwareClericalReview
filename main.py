# ============================================================
# 7. Main pipeline: compare designs with ambiguity-aware strata
# ============================================================
import pandas as pd

from "1_load_data.py" import *
from  "2_matchability.py" import *
from  "3_sample_size.py" import *
from  "4_budget_sample.py" import *
from  "5_draw_sample.py" import *
from  "6_visualisation_helper.py" import *

def main_compare_designs():
    """
    Main pipeline to:

      1) Load data, fit Splink, score pairs.
         - Define matchability and conditional perplexity (odds-based).
         - Define underlying truth and comparison patterns.

      2) Define base band -> MOE schedule (priority structure).
         - Print/table match_probability by band.
         - Plot perplexity vs matchability (global + high-score bands).

      3) Learn ambiguity bins from (matchability, cond_perplexity) and
         obtain ambiguity factors for MOE scaling.

      4) Find scale_star for a resource-constrained design (~5%),
         prioritising higher-ambiguity strata via ambig_factor.

      5) Run replicate evaluation for baseline and budget designs.

      6) Evaluate representation for baseline and budget designs
         in terms of pattern, gender, and ambiguity (single draw).

      7) Compare deviation from truth:
         - global
         - band-level error comparison.

      8) Compare representation per band and visualise:
         - band-level error + pattern L1 (baseline vs budget)
         - band × ambiguity-bin composition with error bars.
    """

    # ------------------------------------------------------------------
    # 1) Load data + fit Splink + score pairs
    # ------------------------------------------------------------------
    print("Loading data and fitting model...")
    df = load_data()
    linker = fit_splink_model(df)
    df_pred_raw = get_predictions(linker)

    # 1a) Matchability, conditional perplexity, bands, patterns, truth
    print("Computing matchability, conditional perplexity, bands, patterns, truth...")
    df_pred, record_metrics, gender_col = prepare_pairs_with_uncertainty(
        df_pred_raw,
        prob_col="match_probability",
        weight_col="match_weight",
        band_col="band_10",
        prior_default_odds=1.0,   # neutral prior; can tune via overlap model in production
    )
    total_N = len(df_pred)
    print(f"Total scored pairs: {total_N:,}")

    # 1b) Learn ambiguity bins from the (matchability, cond_perplexity) distribution
    print("Learning ambiguity bins from (matchability, cond_perplexity)...")
    rec_unc = record_metrics[["unique_id_l", "matchability", "perp_cond"]].rename(
        columns={"perp_cond": "cond_perplexity"}
    )

    rec_unc_learned, ambig_factor = learn_ambiguity_bins_from_distribution(
        rec_unc,
        match_col="matchability",
        cpp_col="cond_perplexity",
        min_k=3,
        max_k=6,
        random_state=42,
        max_cpp=50.0,   # cap for clustering stability
    )

    # Attach ambig_bin to pair table
    df_pred = df_pred.merge(
        rec_unc_learned[["unique_id_l", "ambig_bin"]],
        on="unique_id_l",
        how="left",
    )

    # ------------------------------------------------------------------
    # 2) Base band -> MOE schedule (priority structure) + diagnostics
    # ------------------------------------------------------------------
    band_to_moe_base = {
        "b10": 0.03,
        "b9":  0.03,
        "b8":  0.035,
        "b7":  0.04,
        "b6":  0.05,
        "b5":  0.05,
        "b4":  0.06,
        "b3":  0.06,
        "b2":  0.07,
        "b1":  0.07,
    }

    print("Summary of match_probability by band:")
    plot_match_probability_by_band(df_pred, band_col="band_10")

    print("Plotting conditional perplexity vs matchability (all bands, log count)...")
    plot_perplexity_vs_matchability(df_pred, log_count=True)

    print("Plotting conditional perplexity vs matchability (bands 6–10, log count)...")
    high_bands = [f"b{i}" for i in range(6, 11)]
    plot_perplexity_vs_matchability(
        df_pred,
        bands=high_bands,
        log_count=True,
    )

    print("Plotting conditional perplexity vs matchability (bands 6–10, cpp<=50)...")
    plot_perplexity_vs_matchability(
        df_pred,
        bands=high_bands,
        log_count=True,
        max_cpp=50.0,
    )

    # Strata: score band × ambiguity bin × pattern × gender
    strata_cols = ["band_10", "ambig_bin", "match_pattern", gender_col]

    # ------------------------------------------------------------------
    # 3) Find scale_star for 5% resource-constrained design
    # ------------------------------------------------------------------
    budget_frac = 0.05
    print("Searching for scale factor to achieve ~5% budget...")

    scale_star = find_scale_for_budget(
        df_scored=df_pred,
        band_to_moe_base=band_to_moe_base,
        budget_frac=budget_frac,
        strata_cols=strata_cols,
        band_col="band_10",
        ambig_col="ambig_bin",
        prob_col="match_probability",
        label_col="true_match",
        ambig_factor=ambig_factor,
        max_iter=20,
        tol=0.001,
    )
    print(f"scale_star (budget design): {scale_star:.3f}")

    # ------------------------------------------------------------------
    # 4) Run replicate evaluation for BOTH designs
    # ------------------------------------------------------------------
    print("\nRunning baseline design (scale=1.0)...")
    res_base = run_replicates_design(
        df_pred=df_pred,
        band_to_moe_base=band_to_moe_base,
        scale=1.0,
        strata_cols=strata_cols,
        band_col="band_10",
        ambig_col="ambig_bin",
        prob_col="match_probability",
        label_col="true_match",
        ambig_factor=ambig_factor,   # still ambiguity-aware, just unscaled
        n_reps=5,
        seed_base=100,
    )

    print("Running budget design (scale=scale_star)...")
    res_budget = run_replicates_design(
        df_pred=df_pred,
        band_to_moe_base=band_to_moe_base,
        scale=scale_star,
        strata_cols=strata_cols,
        band_col="band_10",
        ambig_col="ambig_bin",
        prob_col="match_probability",
        label_col="true_match",
        ambig_factor=ambig_factor,
        n_reps=5,
        seed_base=1000,
    )

    # ------------------------------------------------------------------
    # 5) Representation metrics for ONE replicate per design
    # ------------------------------------------------------------------
    print("\nComputing representation metrics for baseline design...")
    band_comp_base, plan_base, n_base_one, frac_base_one = (
        summarise_representation_for_design(
            df_pred=df_pred,
            band_to_moe_base=band_to_moe_base,
            scale=1.0,
            strata_cols=strata_cols,
            band_col="band_10",
            pattern_col="match_pattern",
            gender_col=gender_col,
            ambig_col="ambig_bin",
            ambig_factor=ambig_factor,
            random_state=123,
        )
    )

    print("Computing representation metrics for budget design...")
    band_comp_budget, plan_budget, n_budget_one, frac_budget_one = (
        summarise_representation_for_design(
            df_pred=df_pred,
            band_to_moe_base=band_to_moe_base,
            scale=scale_star,
            strata_cols=strata_cols,
            band_col="band_10",
            pattern_col="match_pattern",
            gender_col=gender_col,
            ambig_col="ambig_bin",
            ambig_factor=ambig_factor,
            random_state=123,
        )
    )

    print(f"\nSingle-draw baseline sample fraction: {frac_base_one:.2%}")
    print(f"Single-draw budget   sample fraction: {frac_budget_one:.2%}")

    # ------------------------------------------------------------------
    # 6) Compare deviation from truth (global + band-level)
    # ------------------------------------------------------------------
    ge_base   = res_base["global_errors"]
    ge_budget = res_budget["global_errors"]

    global_summary = pd.DataFrame({
        "design": ["baseline", "budget"],
        "mean_global_abs_error_pp": [
            ge_base["global_abs_error_pp"].mean(),
            ge_budget["global_abs_error_pp"].mean(),
        ],
        "max_global_abs_error_pp": [
            ge_base["global_abs_error_pp"].max(),
            ge_budget["global_abs_error_pp"].max(),
        ],
        "mean_n_sample": [
            ge_base["n_sample"].mean(),
            ge_budget["n_sample"].mean(),
        ],
    })

    print("\n=== Global deviation from truth (percentage points) ===")
    print(global_summary)

    # Band-level error comparison
    br_base = res_base["band_reliability"].rename(
        columns={
            "mean_abs_error_pp": "mean_abs_error_pp_base",
            "max_abs_error_pp": "max_abs_error_pp_base",
            "q95_abs_error_pp": "q95_abs_error_pp_base",
        }
    )
    br_budget = res_budget["band_reliability"].rename(
        columns={
            "mean_abs_error_pp": "mean_abs_error_pp_budget",
            "max_abs_error_pp": "max_abs_error_pp_budget",
            "q95_abs_error_pp": "q95_abs_error_pp_budget",
        }
    )

    band_error_compare = (
        br_base[
            [
                "band_10",
                "true_match_rate_pp",
                "mean_abs_error_pp_base",
                "max_abs_error_pp_base",
                "q95_abs_error_pp_base",
            ]
        ]
        .merge(
            br_budget[
                [
                    "band_10",
                    "mean_abs_error_pp_budget",
                    "max_abs_error_pp_budget",
                    "q95_abs_error_pp_budget",
                ]
            ],
            on="band_10",
            how="inner",
        )
    )

    band_error_compare["delta_mean_abs_error_pp"] = (
        band_error_compare["mean_abs_error_pp_budget"]
        - band_error_compare["mean_abs_error_pp_base"]
    )
    band_error_compare["delta_max_abs_error_pp"] = (
        band_error_compare["max_abs_error_pp_budget"]
        - band_error_compare["max_abs_error_pp_base"]
    )

    print("\n=== Band-level deviation from truth: budget vs baseline (pp) ===")
    print(band_error_compare)

    # ------------------------------------------------------------------
    # 7) Compare representation (pattern/gender/ambiguity L1) per band
    # ------------------------------------------------------------------
    bc_base = band_comp_base.rename(
        columns={
            "N_sample": "N_sample_base",
            "sample_rate": "sample_rate_base",
            "L1_pattern_pct": "L1_pattern_pct_base",
            "L1_gender_pct": "L1_gender_pct_base",
            "L1_ambig_pct": "L1_ambig_pct_base",
        }
    )
    bc_budget = band_comp_budget.rename(
        columns={
            "N_sample": "N_sample_budget",
            "sample_rate": "sample_rate_budget",
            "L1_pattern_pct": "L1_pattern_pct_budget",
            "L1_gender_pct": "L1_gender_pct_budget",
            "L1_ambig_pct": "L1_ambig_pct_budget",
        }
    )

    band_repr_compare = (
        bc_base[
            [
                "band_10",
                "N_pop",
                "N_sample_base",
                "sample_rate_base",
                "L1_pattern_pct_base",
                "L1_gender_pct_base",
                "L1_ambig_pct_base",
            ]
        ]
        .merge(
            bc_budget[
                [
                    "band_10",
                    "N_sample_budget",
                    "sample_rate_budget",
                    "L1_pattern_pct_budget",
                    "L1_gender_pct_budget",
                    "L1_ambig_pct_budget",
                ]
            ],
            on="band_10",
            how="inner",
        )
    )

    band_repr_compare["delta_L1_pattern_pct"] = (
        band_repr_compare["L1_pattern_pct_budget"]
        - band_repr_compare["L1_pattern_pct_base"]
    )
    band_repr_compare["delta_L1_gender_pct"] = (
        band_repr_compare["L1_gender_pct_budget"]
        - band_repr_compare["L1_gender_pct_base"]
    )
    band_repr_compare["delta_L1_ambig_pct"] = (
        band_repr_compare["L1_ambig_pct_budget"]
        - band_repr_compare["L1_ambig_pct_base"]
    )

    print(
        "\n=== Band-level representation "
        "(pattern/gender/ambiguity L1, % mass moved) ==="
    )
    print(band_repr_compare)

    # ------------------------------------------------------------------
    # 8) Visualisations
    # ------------------------------------------------------------------

    # 8a) Band-level mean abs. error & pattern L1
    plot_band_error_and_pattern(
        band_error_compare=band_error_compare,
        band_repr_compare=band_repr_compare,
    )

    plot_band_error_with_errorbars(
        band_errors_base=res_base["band_errors_long"],
        band_errors_budget=res_budget["band_errors_long"],
    )

    # 8b) Ambiguity-bin composition with error bars (across 5 reps)
    print("\nSummarising ambiguity-bin composition across replicates...")
    ambig_summary_base = summarise_ambiguity_over_reps(
        res_base["ambig_comp"],
        band_col="band_10",
        ambig_col="ambig_bin",
    )
    ambig_summary_budget = summarise_ambiguity_over_reps(
        res_budget["ambig_comp"],
        band_col="band_10",
        ambig_col="ambig_bin",
    )

    print("\nPlotting ambiguity-bin composition (population vs baseline vs budget)...")
    plot_ambiguity_composition_with_errorbars(
        ambig_summary_base,
        ambig_summary_budget,
        band_col="band_10",
        ambig_col="ambig_bin",
    )

     # 8b. Representation: run 5 replicate samples per design to get
    #     mean/sd of L1 distances by band.
    repr_base_summary, mean_n_base_repr = collect_representation_over_reps(
        df_pred=df_pred,
        band_to_moe_base=band_to_moe_base,
        scale=1.0,
        strata_cols=strata_cols,
        band_col="band_10",
        pattern_col="match_pattern",
        gender_col=gender_col,
        ambig_col="ambig_bin",
        ambig_factor=ambig_factor,
        n_reps=5,
        seed_base=3000,
    )

    repr_budget_summary, mean_n_budget_repr = collect_representation_over_reps(
        df_pred=df_pred,
        band_to_moe_base=band_to_moe_base,
        scale=scale_star,
        strata_cols=strata_cols,
        band_col="band_10",
        pattern_col="match_pattern",
        gender_col=gender_col,
        ambig_col="ambig_bin",
        ambig_factor=ambig_factor,
        n_reps=5,
        seed_base=4000,
    )
    
    # 8c. Pattern representation chart
    plot_pattern_representation_with_errorbars(
        repr_base=repr_base_summary,
        repr_budget=repr_budget_summary,
    )

    # 8d. Ambiguity-bin representation chart
    plot_ambiguity_representation_with_errorbars(
        repr_base=repr_base_summary,
        repr_budget=repr_budget_summary,
    )

    show_examples_by_ambiguity(df_pred, n_per_bin=5)

    ambig_joint_summary = (
    rec_unc_learned
        .groupby("ambig_bin")
        .agg(
            n                   = ("cond_perplexity", "count"),
            mean_cpp            = ("cond_perplexity", "mean"),
            sd_cpp              = ("cond_perplexity", "std"),
            mean_matchability   = ("matchability", "mean"),
            sd_matchability     = ("matchability", "std"),
        )
        .reset_index()
        .sort_values("ambig_bin")
    )

    print(ambig_joint_summary)

    plot_perplexity_vs_matchability_facet(
        df_pred,
        band_col="band_10",
        bands=["b6", "b7", "b8", "b9", "b10"],
        log_count=True,
        max_cpp=None,   # or e.g. 50.0 if you *do* want to cap
    )
    # Draw a single clerical sample under each design for plotting
    sample_base_gender = draw_clerical_sample(
        df=df_pred,
        plan=plan_base,
        strata_cols=strata_cols,
        random_state=123,   # keep fixed for reproducibility
    )

    sample_budget_gender = draw_clerical_sample(
        df=df_pred,
        plan=plan_budget,
        strata_cols=strata_cols,
        random_state=123,   # same seed so differences reflect design, not RNG
    )

    plot_gender_representation_by_band(
    df_pop=df_pred,
    df_sample_base=sample_base_gender,
    df_sample_budget=sample_budget_gender,
    band_col="band_10",
    gender_col=gender_col,
    )


# If running as a script, call main():
if __name__ == "__main__":
    main_compare_designs()

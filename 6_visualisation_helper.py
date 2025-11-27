# ============================================================
# 6. Visualisation helpers
# ============================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
import re

def plot_band_error_with_errorbars(
    band_errors_base: pd.DataFrame,
    band_errors_budget: pd.DataFrame,
):
    """
    Plot band-level mean absolute error with error bars (SD across reps),
    baseline vs budget, in a single chart.
    """
    def _summarise(band_err: pd.DataFrame) -> pd.DataFrame:
        g = band_err.groupby("band_10")["abs_error_pp"]
        return (
            g.agg(mean="mean", sd="std")
             .reset_index()
             .sort_values("band_10")
        )

    sum_base = _summarise(band_errors_base)
    sum_budget = _summarise(band_errors_budget)

    df = sum_base.merge(
        sum_budget,
        on="band_10",
        suffixes=("_base", "_budget"),
        how="inner",
    )

    bands = df["band_10"].tolist()
    x = np.arange(len(bands))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        x - width / 2,
        df["mean_base"],
        width,
        yerr=df["sd_base"],
        capsize=3,
        label="baseline",
        alpha=0.9,
    )
    plt.bar(
        x + width / 2,
        df["mean_budget"],
        width,
        yerr=df["sd_budget"],
        capsize=3,
        label="budget",
        alpha=0.9,
    )
    plt.xticks(x, bands)
    plt.xlabel("band_10")
    plt.ylabel("Mean abs. error (pp)")
    plt.title("Band-level mean absolute error (±1 SD over 5 reps)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_match_probability_by_band(
    df_pred: pd.DataFrame,
    band_col: str = "band_10",
    prob_col: str = "match_probability",
) -> None:
    """
    Print summary stats of match_probability by band and
    plot a simple boxplot.

    Parameters
    ----------
    df_pred : DataFrame
        Pair-level table with bands and match_probability.
    band_col : str
        Column with band labels (e.g. 'band_10').
    prob_col : str
        Column with model match probabilities.
    """
    summary = (
        df_pred
        .groupby(band_col)[prob_col]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
        .sort_values(band_col)
    )

    print("\n=== match_probability by band ===")
    print(summary)

    plt.figure(figsize=(10, 5))

    # Ensure bands appear in b1..b10 order where present
    unique_bands = list(df_pred[band_col].dropna().unique())
    band_order = [f"b{i}" for i in range(1, 11) if f"b{i}" in unique_bands]

    df_pred.boxplot(
        column=prob_col,
        by=band_col,
        grid=False,
        showfliers=False,
    )
    plt.suptitle("")
    plt.title("Match probability by band")
    plt.xlabel(band_col)
    plt.ylabel(prob_col)
    plt.show()


def plot_perplexity_vs_matchability(
    df_pred: pd.DataFrame,
    band_col: str = "band_10",
    bands: Optional[List[str]] = None,
    max_cpp: Optional[float] = None,
    log_count: bool = True,
    matchability_col: str = "matchability",
    cpp_col: str = "cond_perplexity",
) -> None:
    """
    Hexbin plot of conditional perplexity vs matchability.

    Parameters
    ----------
    df_pred : DataFrame
        Pair-level table (already has matchability + cond_perplexity
        merged from record_metrics).
    band_col : str
        Column with band labels (e.g. 'band_10').
    bands : list of str or None
        If provided, restrict to these bands (e.g. ['b6','b7','b8','b9','b10']).
    max_cpp : float or None
        Optional cap on conditional perplexity for plotting (e.g. 50).
    log_count : bool
        If True, use log-normalisation for hexbin counts.
    matchability_col : str
        Column with record-level matchability.
    cpp_col : str
        Column with record-level conditional perplexity.
    """
    df_plot = df_pred.copy()

    if bands is not None:
        df_plot = df_plot[df_plot[band_col].isin(bands)]

    if max_cpp is not None:
        df_plot = df_plot[df_plot[cpp_col] <= max_cpp]

    if df_plot.empty:
        print("No data to plot after band / cpp filters.")
        return

    norm = LogNorm() if log_count else None

    plt.figure(figsize=(7, 6))
    hb = plt.hexbin(
        df_plot[matchability_col],
        df_plot[cpp_col],
        gridsize=40,
        mincnt=1,
        norm=norm,
    )
    cbar = plt.colorbar(hb)
    cbar.set_label("count (log scale)" if log_count else "count")

    plt.xlabel(matchability_col)
    plt.ylabel(cpp_col)
    title_suffix = ""
    if bands is not None:
        title_suffix = f" (bands: {', '.join(sorted(set(bands)))} )"
    plt.title(f"Conditional perplexity vs matchability{title_suffix}")
    plt.tight_layout()
    plt.show()


def plot_band_error_and_pattern(
    band_error_compare: pd.DataFrame,
    band_repr_compare: pd.DataFrame,
    band_col: str = "band_10",
) -> None:
    """
    Visualise:
      - band-level mean_abs_error_pp (baseline vs budget)
      - band-level L1_pattern_pct (baseline vs budget)
      - optionally, L1_ambig_pct (baseline vs budget) if present.

    Parameters
    ----------
    band_error_compare : DataFrame
        Output from main_compare_designs for band-level errors
        (contains mean_abs_error_pp_base, mean_abs_error_pp_budget, etc.).
    band_repr_compare : DataFrame
        Output from main_compare_designs for band-level representation
        (contains L1_pattern_pct_base, L1_pattern_pct_budget, and possibly
         L1_ambig_pct_base / L1_ambig_pct_budget).
    band_col : str
        Band column name (default 'band_10').
    """
    # Merge error and representation on band
    df = band_error_compare.merge(
        band_repr_compare[
            [
                band_col,
                "L1_pattern_pct_base",
                "L1_pattern_pct_budget",
                "L1_ambig_pct_base",
                "L1_ambig_pct_budget",
            ]
        ],
        on=band_col,
        how="left",
    )

    # Ensure standard band order
    bands = [f"b{i}" for i in range(1, 11)]
    df = df.set_index(band_col).reindex(bands).dropna(how="all").reset_index()

    x = np.arange(len(df))
    width = 0.35

    # Decide whether we have ambiguity metrics
    has_ambig = (
        "L1_ambig_pct_base" in df.columns
        and "L1_ambig_pct_budget" in df.columns
    )

    n_panels = 3 if has_ambig else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), sharex=True)

    if n_panels == 2:
        ax_err, ax_pattern = axes
        ax_ambig = None
    else:
        ax_err, ax_pattern, ax_ambig = axes

    # Panel 1: band-level mean absolute error
    ax_err.bar(
        x - width / 2,
        df["mean_abs_error_pp_base"],
        width,
        label="baseline",
    )
    ax_err.bar(
        x + width / 2,
        df["mean_abs_error_pp_budget"],
        width,
        label="budget",
    )
    ax_err.set_xticks(x)
    ax_err.set_xticklabels(df[band_col])
    ax_err.set_xlabel(band_col)
    ax_err.set_ylabel("Mean abs. error (pp)")
    ax_err.set_title("Band-level mean absolute error")
    ax_err.legend()

    # Panel 2: pattern L1
    ax_pattern.bar(
        x - width / 2,
        df["L1_pattern_pct_base"],
        width,
        label="baseline",
    )
    ax_pattern.bar(
        x + width / 2,
        df["L1_pattern_pct_budget"],
        width,
        label="budget",
    )
    ax_pattern.set_xticks(x)
    ax_pattern.set_xticklabels(df[band_col])
    ax_pattern.set_xlabel(band_col)
    ax_pattern.set_ylabel("L1 pattern distance (% mass moved)")
    ax_pattern.set_title("Pattern representation (baseline vs budget)")
    ax_pattern.legend()

    # Panel 3: ambiguity L1 (if available)
    if has_ambig and ax_ambig is not None:
        ax_ambig.bar(
            x - width / 2,
            df["L1_ambig_pct_base"],
            width,
            label="baseline",
        )
        ax_ambig.bar(
            x + width / 2,
            df["L1_ambig_pct_budget"],
            width,
            label="budget",
        )
        ax_ambig.set_xticks(x)
        ax_ambig.set_xticklabels(df[band_col])
        ax_ambig.set_xlabel(band_col)
        ax_ambig.set_ylabel("L1 ambiguity distance (% mass moved)")
        ax_ambig.set_title("Ambiguity-bin representation")
        ax_ambig.legend()

    plt.tight_layout()
    plt.show()

def plot_pattern_representation_with_errorbars(
    repr_base: pd.DataFrame,
    repr_budget: pd.DataFrame,
):
    """
    Plot band-level pattern L1 distance (% mass moved),
    with error bars (SD across reps), baseline vs budget.
    """
    df = repr_base[["band_10", "L1_pattern_mean", "L1_pattern_sd"]].merge(
        repr_budget[["band_10", "L1_pattern_mean", "L1_pattern_sd"]],
        on="band_10",
        suffixes=("_base", "_budget"),
        how="inner",
    ).sort_values("band_10")

    bands = df["band_10"].tolist()
    x = np.arange(len(bands))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        x - width / 2,
        df["L1_pattern_mean_base"],
        width,
        yerr=df["L1_pattern_sd_base"],
        capsize=3,
        label="baseline",
        alpha=0.9,
    )
    plt.bar(
        x + width / 2,
        df["L1_pattern_mean_budget"],
        width,
        yerr=df["L1_pattern_sd_budget"],
        capsize=3,
        label="budget",
        alpha=0.9,
    )
    plt.xticks(x, bands)
    plt.xlabel("band_10")
    plt.ylabel("L1 pattern distance (% mass moved)")
    plt.title("Pattern representation (±1 SD over 5 reps)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ambiguity_representation_with_errorbars(
    repr_base: pd.DataFrame,
    repr_budget: pd.DataFrame,
):
    """
    Plot band-level ambiguity-bin L1 distance (% mass moved),
    with error bars (SD across reps), baseline vs budget.
    """
    df = repr_base[["band_10", "L1_ambig_mean", "L1_ambig_sd"]].merge(
        repr_budget[["band_10", "L1_ambig_mean", "L1_ambig_sd"]],
        on="band_10",
        suffixes=("_base", "_budget"),
        how="inner",
    ).sort_values("band_10")

    bands = df["band_10"].tolist()
    x = np.arange(len(bands))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        x - width / 2,
        df["L1_ambig_mean_base"],
        width,
        yerr=df["L1_ambig_sd_base"],
        capsize=3,
        label="baseline",
        alpha=0.9,
    )
    plt.bar(
        x + width / 2,
        df["L1_ambig_mean_budget"],
        width,
        yerr=df["L1_ambig_sd_budget"],
        capsize=3,
        label="budget",
        alpha=0.9,
    )
    plt.xticks(x, bands)
    plt.xlabel("band_10")
    plt.ylabel("L1 ambiguity distance (% mass moved)")
    plt.title("Ambiguity-bin representation (±1 SD over 5 reps)")
    plt.legend()
    plt.tight_layout()
    plt.show()



import pandas as pd


def show_examples_by_ambiguity(
    df_pairs: pd.DataFrame,
    ambig_col: str = "ambig_bin",
    n_per_bin: int = 5,
    id_base_cols = ("first_name", "surname", "dob", "postcode_fake",
                    "birth_place", "occupation"),
    extra_cols = (
        "band_10",
        "match_probability",
        "matchability",
        "cond_perplexity",
        "match_pattern",
        "true_match",
    ),
    random_state: int = 42,
):    
    """
    Print a few example pairs from each ambiguity bin, showing both _l and _r
    identifiers, the gamma columns, and a legend for the gamma order.

    Parameters
    ----------
    df_pairs : DataFrame
        Pair-level Splink predictions with:
          - *_l / *_r columns for identifiers,
          - gamma_* columns,
          - band_10, ambig_bin, matchability, cond_perplexity, match_pattern, etc.
    ambig_col : str
        Column containing the learned ambiguity bin (e.g. 'a1'...'aK').
    n_per_bin : int
        Number of pairs to show per bin (max, truncated if bin smaller).
    id_base_cols : tuple of str
        Base names of identifier fields; function will look for <base>_l
        and <base>_r, and include them if present.
    extra_cols : tuple of str
        Additional columns to display for context.
    random_state : int
        Seed for reproducible sampling.
    """

    rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------
    # 1. Figure out gamma order and print a legend once
    # ------------------------------------------------------------
    gamma_cols = sorted([c for c in df_pairs.columns if c.startswith("gamma_")])

    if not gamma_cols:
        print("No gamma_* columns found; cannot map match_pattern order.")
    else:
        print("Gamma pattern order and associated identifiers:")
        for pos, gcol in enumerate(gamma_cols, start=1):
            base = gcol.replace("gamma_", "")
            cand_l = f"{base}_l"
            cand_r = f"{base}_r"
            id_cols = []
            if cand_l in df_pairs.columns:
                id_cols.append(cand_l)
            if cand_r in df_pairs.columns:
                id_cols.append(cand_r)
            id_cols_str = ", ".join(id_cols) if id_cols else "[no *_l/_r columns found]"
            print(f"  ({pos}) {gcol}  |  base='{base}'  ->  {id_cols_str}")
        print()

    # ------------------------------------------------------------
    # 2. Show examples by ambiguity bin
    # ------------------------------------------------------------
    bins = (
        df_pairs[ambig_col]
        .dropna()
        .unique()
        .tolist()
    )
    bins = sorted(bins)  # e.g. ['a1','a2',...]

    for b in bins:
        sub = df_pairs[df_pairs[ambig_col] == b]
        n_bin = len(sub)

        # Important: N here is number of **pairs**, not unique records.
        print(f"\n=== Ambiguity bin {b} (N_pairs = {n_bin}) ===")

        if n_bin == 0:
            print("  [no pairs in this bin]")
            continue

        n_show = min(n_per_bin, n_bin)
        ex = sub.sample(n=n_show, random_state=rng)

        # Build list of columns to show: left/right identifiers, then extras, then gamma_*
        display_cols = []

        # 2a. add identifier columns in _l/_r form
        for base in id_base_cols:
            l_col = f"{base}_l"
            r_col = f"{base}_r"
            if l_col in ex.columns and l_col not in display_cols:
                display_cols.append(l_col)
            if r_col in ex.columns and r_col not in display_cols:
                display_cols.append(r_col)

        # 2b. add extra context columns
        for col in extra_cols:
            if col in ex.columns and col not in display_cols:
                display_cols.append(col)

        # 2c. add gamma_* columns in the same sorted order used in match_pattern
        for gcol in gamma_cols:
            if gcol in ex.columns and gcol not in display_cols:
                display_cols.append(gcol)

        # If for some reason nothing selected, fall back to a few generic columns
        if not display_cols:
            display_cols = [c for c in ex.columns if c.endswith(("_l", "_r"))][:8]

        print(ex[display_cols].to_string(index=False))


def plot_perplexity_vs_matchability_facet(
    df_pred: pd.DataFrame,
    band_col: str = "band_10",
    bands: list[str] | None = None,
    log_count: bool = True,
    gridsize: int = 35,
    max_cpp: float | None = None,
):
    """
    Faceted hexbin plot of conditional perplexity vs matchability,
    one panel per band, with a single colourbar on the far right.

    Parameters
    ----------
    df_pred : DataFrame
        Needs columns: 'matchability', 'cond_perplexity', band_col.
    band_col : str
        Band label column, e.g. 'band_10'.
    bands : list[str] or None
        Bands to include, e.g. ['b6','b7','b8','b9','b10'].
        If None, uses all unique band values.
    log_count : bool
        If True, hexbin colour is log(count).
    gridsize : int
        Hexbin grid size.
    max_cpp : float or None
        Optional cap on cond_perplexity (same for all bands).
        If None, use full range.
    """
    df = df_pred.copy()

    # Which bands to show
    if bands is None:
        bands = sorted(df[band_col].unique())
    else:
        bands = [b for b in bands if b in df[band_col].unique()]

    if max_cpp is not None:
        df = df[df["cond_perplexity"] <= max_cpp]

    n_bands = len(bands)
    if n_bands == 0:
        print("No bands to plot.")
        return

    norm = LogNorm() if log_count else None

    fig, axes = plt.subplots(
        1, n_bands,
        figsize=(4 * n_bands, 5),
        sharex=True,
        sharey=True,            # common y-scale across bands
        squeeze=False,
    )
    axes = axes[0]

    hb_last = None

    for ax, b in zip(axes, bands):
        d = df[df[band_col] == b]
        if d.empty:
            ax.set_title(f"Band {b} (no data)")
            continue

        hb = ax.hexbin(
            d["matchability"],
            d["cond_perplexity"],
            gridsize=gridsize,
            mincnt=1,
            norm=norm,
        )
        hb_last = hb

        ax.set_title(f"Band {b}")
        ax.set_xlabel("matchability")
        ax.set_xlim(0, 1.0)

    axes[0].set_ylabel("cond_perplexity")

    # Leave room on right for colourbar
    plt.tight_layout(rect=[0.0, 0.0, 0.9, 1.0])

    # Colourbar at far right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(hb_last, cax=cbar_ax)
    cbar.set_label("count (log scale)" if log_count else "count")

    fig.suptitle("Conditional perplexity vs matchability, by band", y=1.02)
    plt.show()


def plot_gender_representation_by_band(
    df_pop: pd.DataFrame,
    df_sample_base: pd.DataFrame,
    df_sample_budget: pd.DataFrame,
    band_col: str = "band_10",
    gender_col: str = "gender_l",
):
    """
    Plot gender representation by band as deviations from the population,
    and print the corresponding L1_gender_pct per band for baseline/budget.

    For each band b and gender g we plot:
        Δ_{b,g} = p_sample(g | b) - p_pop(g | b)

    The L1 distance per band is:
        L1_gender_pct_b = 50 * Σ_g |Δ_{b,g}|

    Parameters
    ----------
    df_pop : DataFrame
        Full pair population.
    df_sample_base : DataFrame
        Sample under baseline design.
    df_sample_budget : DataFrame
        Sample under budget design.
    band_col : str
        Band column name.
    gender_col : str
        Gender column name (e.g. 'gender_l').
    """
    # --- 1. Clean band ordering: b1, b2, ..., b10 ---
    def _band_key(b):
            # extract numeric part, default 0 if not found
            m = re.search(r"(\d+)", str(b))
            return int(m.group(1)) if m else 0

        bands = sorted(df_pop[band_col].dropna().unique(), key=_band_key)

        # Genders to plot
        genders = list(df_pop[gender_col].dropna().unique())
        genders.sort()  # alphabetical: e.g. ['female', 'male', 'unknown']
        n_g = len(genders)

        # --- 2. Compute Δ (sample - population) for each design ---
        # precompute population band sizes (for normalisation)
        pop_band_sizes = df_pop.groupby(band_col)[gender_col].size()

        # helper: get conditional proportion p(g | band)
        def _conditional_prop(df, band, gender):
            sub = df[df[band_col] == band]
            if len(sub) == 0:
                return 0.0
            counts = sub[gender_col].value_counts(normalize=True)
            return float(counts.get(gender, 0.0))

        # also compute L1_gender_pct per band
        l1_rows = []

        # Prepare plot
        fig, axes = plt.subplots(
            1, n_g, figsize=(4 * n_g + 2, 4), sharey=True
        )
        if n_g == 1:
            axes = [axes]

        for j, g in enumerate(genders):
            ax = axes[j]
            diffs_base = []
            diffs_budget = []

            for b in bands:
                # population prop
                p_pop = _conditional_prop(df_pop, b, g)
                # baseline
                p_base = _conditional_prop(df_sample_base, b, g)
                # budget
                p_budget = _conditional_prop(df_sample_budget, b, g)

                diffs_base.append(p_base - p_pop)
                diffs_budget.append(p_budget - p_pop)

            # bar positions
            x = np.arange(len(bands))
            width = 0.35

            ax.axhline(0.0, color="grey", linewidth=0.5)
            ax.bar(
                x - width / 2,
                diffs_base,
                width,
                label="baseline",
            )
            ax.bar(
                x + width / 2,
                diffs_budget,
                width,
                label="budget",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(bands)
            ax.set_xlabel(band_col)
            if j == 0:
                ax.set_ylabel("Δ proportion (sample – population)")
            ax.set_title(f"Gender = {g}")

            if j == 0:
                ax.legend()

        fig.suptitle("Gender representation by band: deviation from population", y=1.02)
        plt.tight_layout()
        plt.show()

        # --- 3. L1_gender_pct per band (for both designs) ---
        l1_table = []
        for b in bands:
            pop_counts = df_pop[df_pop[band_col] == b][gender_col].value_counts()
            base_counts = df_sample_base[df_sample_base[band_col] == b][
                gender_col
            ].value_counts()
            budget_counts = df_sample_budget[df_sample_budget[band_col] == b][
                gender_col
            ].value_counts()

            L1_base = l1_distance_pct(pop_counts, base_counts)
            L1_budget = l1_distance_pct(pop_counts, budget_counts)

            l1_table.append(
                {
                    "band_10": b,
                    "L1_gender_pct_base": L1_base,
                    "L1_gender_pct_budget": L1_budget,
                }
            )

        l1_df = pd.DataFrame(l1_table)
        print("\n=== L1 gender distance per band (%% mass moved) ===")
        print(l1_df.to_string(index=False))

    return l1_df
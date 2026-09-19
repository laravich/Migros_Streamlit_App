"""One render function per page of the app.

Each function receives the already-loaded data and does three things: explain the step
in one or two lines, show the chart, and show the small table behind it. The analytical
work itself lives in ``analysis.py``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import analysis, charts, config
from .data_loader import brand_summary, canton_summary, national_kpis


# --------------------------------------------------------------------------------------
# Shared table helpers
# --------------------------------------------------------------------------------------

# Reusable column formats, so a franc looks the same on every page
COLUMN_FORMATS = {
    "population": st.column_config.NumberColumn("Population", format="localized"),
    "catchment_population_5km": st.column_config.NumberColumn("Population 5 km", format="localized"),
    "avg_income_per_taxpayer": st.column_config.NumberColumn("Avg income", format="CHF %.0f"),
    "external_competitor_count": st.column_config.NumberColumn("Competitors", format="%d"),
    "nearest_migros_distance_km": st.column_config.NumberColumn("Nearest Migros", format="%.2f km"),
    "opportunity_score": st.column_config.NumberColumn("Score", format="%.1f"),
    "predicted_probability": st.column_config.NumberColumn("Model probability", format="percent"),
    "postal_code": st.column_config.TextColumn("Postcode"),
    "locality_name": st.column_config.TextColumn("Locality"),
    "municipality_name": st.column_config.TextColumn("Municipality"),
    "primary_canton": st.column_config.TextColumn("Canton"),
    "rank": st.column_config.NumberColumn("Rank", format="%d"),
}


def show_table(frame: pd.DataFrame, caption: str | None = None, height: int | str = "content") -> None:
    """Render a compact, consistently formatted table under a chart."""
    if caption:
        st.caption(caption)

    # Only pass formats for columns that are actually present in this table
    column_config = {name: fmt for name, fmt in COLUMN_FORMATS.items() if name in frame.columns}

    # height="content" lets short tables size themselves; a pixel value scrolls long ones
    st.dataframe(frame, column_config=column_config, hide_index=True,
                 width="stretch", height=height)


def page_header(title: str, subtitle: str) -> None:
    """Consistent title block on every page."""
    st.subheader(title)
    st.caption(subtitle)


# --------------------------------------------------------------------------------------
# 1. Overview
# --------------------------------------------------------------------------------------

def render_overview(master: pd.DataFrame) -> None:
    page_header("National picture",
                "Where Migros stands today across all Swiss postcodes with complete data")

    kpis = national_kpis(master)

    # Five headline numbers, the same ones used on the title slide of the deck
    columns = st.columns(5)
    columns[0].metric("Postcodes analysed", f"{kpis['postcodes']:,}")
    columns[1].metric("Population covered", f"{kpis['total_population'] / 1e6:.2f}M")
    columns[2].metric("Migros stores", f"{kpis['migros_stores']:,}")
    columns[3].metric("Postcodes with a Migros", f"{kpis['postcodes_with_migros']:,}")
    columns[4].metric("Population without one", f"{kpis['share_without_migros']:.1f}%")

    left, right = st.columns([3, 2])
    summary = brand_summary(master)
    left.plotly_chart(charts.brand_bar(summary), width="stretch")
    right.plotly_chart(charts.coverage_donut(kpis), width="stretch")

    show_table(
        summary.rename(columns={"brand": "Brand", "stores": "Stores", "share_pct": "Share of all locations"}),
        "Store counts per brand. Denner belongs to Migros Group, so Migros Group runs "
        f"{summary.loc[summary['Brand'].isin(['Migros', 'Denner']), 'Stores'].sum():,} locations in total."
        if "Brand" in summary.columns else None,
    )


# --------------------------------------------------------------------------------------
# 2. Canton screening
# --------------------------------------------------------------------------------------

def render_cantons(master: pd.DataFrame) -> None:
    page_header("Step 1 - Canton screening",
                "Coverage against purchasing power, to decide which cantons deserve a closer look")

    cantons = canton_summary(master)

    st.plotly_chart(charts.canton_bubble(cantons), width="stretch")

    # The table lets a viewer check the exact numbers behind any bubble
    table = cantons[["canton_name", "primary_canton", "population", "avg_income_per_taxpayer",
                     "migros_count", "migros_per_100k", "competitor_count", "competitors_per_100k"]]

    show_table(
        table.round({"avg_income_per_taxpayer": 0, "migros_per_100k": 2, "competitors_per_100k": 2}),
        "All 26 cantons, sorted by population. Schwyz, Zurich and Aargau combine meaningful "
        "population or purchasing power with below-median Migros coverage.",
        height=340,
    )


# --------------------------------------------------------------------------------------
# 3. Postcode gaps
# --------------------------------------------------------------------------------------

def render_gaps(candidates: pd.DataFrame) -> None:
    page_header("Step 2 - Postcode gaps in the screened cantons",
                "Postcodes in Zurich, Schwyz and Aargau that contain no Migros at all")

    # A slider instead of a fixed number, so the depth of the shortlist is the viewer's choice
    top_n = st.slider("Postcodes shown per canton", min_value=5, max_value=15, value=10)

    st.plotly_chart(charts.postcode_facets(candidates, config.SCREENED_CANTONS, top_n),
                    width="stretch")

    table = (candidates.nlargest(top_n * len(config.SCREENED_CANTONS), "population")
             [["postal_code", "locality_name", "municipality_name", "primary_canton", "population",
               "avg_income_per_taxpayer", "external_competitor_count", "nearest_migros_distance_km"]])

    show_table(table, "Largest uncovered postcodes across the three cantons.", height=320)


# --------------------------------------------------------------------------------------
# 4. Weighted ranking
# --------------------------------------------------------------------------------------

def render_ranking(candidates: pd.DataFrame) -> pd.DataFrame:
    page_header("Step 3 - Weighted opportunity score",
                "Four factors combined into one score - move the sliders to see how much the weights matter")

    # Exposing the weights makes the weakness of this method visible: the ranking is only
    # as objective as the weights somebody chose.
    with st.expander("Adjust the weights", expanded=False):
        columns = st.columns(4)
        weights = {
            "population": columns[0].slider("Population", 0.0, 1.0, config.DEFAULT_WEIGHTS["population"], 0.05),
            "income": columns[1].slider("Income", 0.0, 1.0, config.DEFAULT_WEIGHTS["income"], 0.05),
            "distance": columns[2].slider("Distance to Migros", 0.0, 1.0, config.DEFAULT_WEIGHTS["distance"], 0.05),
            "low_competition": columns[3].slider("Low competition", 0.0, 1.0,
                                                 config.DEFAULT_WEIGHTS["low_competition"], 0.05),
        }

    total = sum(weights.values())
    if total == 0:
        st.warning("At least one weight has to be above zero.")
        st.stop()

    # Normalising keeps the score on a 0-100 scale whatever the sliders add up to
    weights = {key: value / total for key, value in weights.items()}

    scored = analysis.score_opportunities(candidates, weights)

    st.plotly_chart(charts.opportunity_ranking(scored, top_n=15), width="stretch")

    table = scored.head(15)[["rank", "postal_code", "locality_name", "primary_canton", "population",
                             "avg_income_per_taxpayer", "external_competitor_count",
                             "nearest_migros_distance_km", "opportunity_score"]]

    show_table(table, "Top 15 candidates under the current weights.", height=320)

    return scored


# --------------------------------------------------------------------------------------
# 5. Logistic regression
# --------------------------------------------------------------------------------------

def render_model(master: pd.DataFrame, shortlist: list[str]) -> None:
    page_header("Step 4 - Statistical coverage model",
                "A logistic regression learns what a Migros postcode looks like, then flags the postcodes "
                "that fit the pattern but have no store")

    model = analysis.fit_logit(master)

    columns = st.columns(4)
    columns[0].metric("Observations", f"{model['observations']:,}")
    columns[1].metric("Pseudo R²", f"{model['pseudo_r2']:.3f}")
    columns[2].metric("Hold-out accuracy", f"{model['metrics']['accuracy']:.3f}")
    columns[3].metric("Hold-out ROC AUC", f"{model['metrics']['roc_auc']:.3f}")

    left, right = st.columns([3, 2])
    left.plotly_chart(charts.odds_ratio_bar(model["effects"]), width="stretch")

    with right:
        st.caption("Coefficients (log-odds) and significance")
        show_table(model["effects"][["label", "coefficient", "odds_ratio_per_1_sd", "p_value"]]
                   .round({"coefficient": 3, "odds_ratio_per_1_sd": 2, "p_value": 4}))
        st.caption("Hold-out confusion matrix")
        st.dataframe(model["metrics"]["confusion"], width="stretch")

    st.markdown("---")

    # The four shortlisted postcodes scored by the model rather than by chosen weights
    candidates = model["scored"][model["scored"]["postal_code"].isin(shortlist)].copy()

    st.plotly_chart(charts.probability_bar(candidates), width="stretch")

    show_table(
        candidates.sort_values("predicted_probability", ascending=False)
        [["postal_code", "locality_name", "primary_canton", "population", "avg_income_per_taxpayer",
          "external_competitor_count", "catchment_population_5km", "predicted_probability"]],
        "Read it as: postcodes with this profile have a Migros X% of the time - but these have none. "
        "The highest value is the clearest coverage gap.",
    )


# --------------------------------------------------------------------------------------
# 6. Catchment comparison
# --------------------------------------------------------------------------------------

def render_catchment(master: pd.DataFrame, shortlist: list[str]) -> None:
    page_header("Step 5 - Catchment check",
                "Postcode counts can mislead: a postcode can show zero Migros while being ringed by them")

    labels = dict(zip(master["postal_code"], master["location_label"]))

    columns = st.columns(2)
    first = columns[0].selectbox("First location", shortlist, index=0,
                                 format_func=lambda code: labels.get(code, code))
    second = columns[1].selectbox("Second location", shortlist,
                                  index=min(1, len(shortlist) - 1),
                                  format_func=lambda code: labels.get(code, code))

    profiles = [analysis.catchment_profile(master, first), analysis.catchment_profile(master, second)]

    st.plotly_chart(charts.catchment_comparison(profiles), width="stretch")

    # Side-by-side ring tables: the numbers behind the three panels above
    display_columns = ["radius_km", "postcodes", "population", "migros_stores", "denner_stores",
                       "external_competitors", "migros_group_per_100k", "competitors_per_100k"]

    for profile, column in zip(profiles, st.columns(2)):
        with column:
            st.caption(profile["location"].iloc[0])
            show_table(profile[display_columns].round(2))

    st.markdown("---")
    left, right = st.columns(2)
    left.plotly_chart(charts.catchment_bars(profiles[0]), width="stretch")
    right.plotly_chart(charts.catchment_bars(profiles[1]), width="stretch")


# --------------------------------------------------------------------------------------
# 7. Site selection
# --------------------------------------------------------------------------------------

def render_sites(master: pd.DataFrame, stores: pd.DataFrame, localities: pd.DataFrame) -> None:
    page_header("Step 6 - Where inside the village?",
                "A 100 m grid scored with a Huff gravity model: expected customers captured, "
                "discounted by distance from the main retail road")

    result = analysis.search_sites(master, stores, localities, config.SIEBNEN_POSTCODE)
    village = master.loc[master["postal_code"] == config.SIEBNEN_POSTCODE, "locality_name"].iloc[0]

    best = result["sites"].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Recommended coordinates", f"{best['latitude']:.5f}, {best['longitude']:.5f}")
    columns[1].metric("Expected customers", f"{best['captured_population']:,.0f}")
    columns[2].metric("Nearest competitor", f"{best['distance_to_nearest_store_km'] * 1000:.0f} m")
    columns[3].metric("Nearest Migros", f"{best['distance_to_nearest_migros_km']:.2f} km")

    st.plotly_chart(charts.site_map(result, village), width="stretch")

    sites_table = result["sites"][["site_rank", "latitude", "longitude", "captured_population",
                                   "site_score", "distance_to_axis_km", "nearest_existing_store",
                                   "distance_to_nearest_migros_km"]].round(
        {"latitude": 5, "longitude": 5, "captured_population": 0, "site_score": 0,
         "distance_to_axis_km": 2, "distance_to_nearest_migros_km": 2})

    show_table(sites_table, "The three best separated sites. Coordinates can be pasted straight into a map tool.")

    # A download button turns the result into something usable outside the app
    st.download_button(
        "Download site coordinates (CSV)",
        data=result["sites"].to_csv(index=False).encode("utf-8"),
        file_name="suggested_sites.csv",
        mime="text/csv",
    )

    with st.expander("Regional context map"):
        st.plotly_chart(charts.regional_map(result), width="stretch")

    with st.expander("Demand points used by the model"):
        show_table(result["demand"].round({"latitude": 5, "longitude": 5, "population": 0}), height=300)


# --------------------------------------------------------------------------------------
# 8. Conclusion
# --------------------------------------------------------------------------------------

def render_conclusion(master: pd.DataFrame, shortlist: list[str]) -> None:
    page_header("Conclusion", "What the analysis recommends, and what it cannot tell you")

    model = analysis.fit_logit(master)
    probability = model["scored"].loc[
        model["scored"]["postal_code"] == config.SIEBNEN_POSTCODE, "predicted_probability"].iloc[0]

    st.success(
        f"**Open the new store in Siebnen (8854), canton Schwyz.** "
        f"The coverage model gives it a {probability:.1%} probability of containing a Migros - "
        f"the highest of the shortlist - while it has none."
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Why Siebnen and not Kilchberg**")
        st.markdown(
            "- Largest uncovered population of the shortlist\n"
            "- No Migros within 2 km, only one within 5 km\n"
            "- Coop, Aldi and Lidl nearby prove supermarket demand exists\n"
            "- Lowest cannibalisation risk of all candidates\n"
            "- Kilchberg ranked first on income alone, but Migros Group already runs five stores within 2 km"
        )

    with right:
        st.markdown("**Limitations**")
        st.markdown(
            "- The population split inside postcode 8854 is an assumption, not measured data\n"
            "- No land-use, zoning or parcel-availability data: the model finds demand, not plots for sale\n"
            "- The main road is inferred from existing store positions, not a road network\n"
            "- Huff parameters are literature defaults, not calibrated on Swiss trip data\n"
            "- The model describes where stores are today, which reflects past decisions, not profitability"
        )

    st.markdown("---")
    st.caption("Shortlisted candidates at a glance")

    table = model["scored"][model["scored"]["postal_code"].isin(shortlist)][
        ["postal_code", "locality_name", "primary_canton", "population", "avg_income_per_taxpayer",
         "external_competitor_count", "predicted_probability"]
    ].sort_values("predicted_probability", ascending=False)

    show_table(table)

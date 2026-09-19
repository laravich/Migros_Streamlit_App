"""Every figure of the app, built with the shared theme from ``config``.

Keeping the charts in one module means a styling decision is made once and the page
files stay readable: they only decide *what* to show, never *how* it looks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config

# Plotly renamed its map traces in 5.24 (MapLibre replaced Mapbox). Detecting the version
# once keeps the app working on both old and new installations.
_PLOTLY_VERSION = tuple(int(part) for part in plotly.__version__.split(".")[:2])
_USE_NEW_MAPS = _PLOTLY_VERSION >= (5, 24)

ScatterMapTrace = go.Scattermap if _USE_NEW_MAPS else go.Scattermapbox
DensityMapTrace = go.Densitymap if _USE_NEW_MAPS else go.Densitymapbox


def _apply_map_layout(figure: go.Figure, latitude: float, longitude: float, zoom: float) -> go.Figure:
    """Centre a map figure, using whichever map property this Plotly version supports."""
    # OpenStreetMap needs no API token, so the app works offline of any map account
    settings = dict(style="open-street-map", center=dict(lat=latitude, lon=longitude), zoom=zoom)
    figure.update_layout(map=settings) if _USE_NEW_MAPS else figure.update_layout(mapbox=settings)
    return figure


# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------

def brand_bar(summary: pd.DataFrame) -> go.Figure:
    """Supermarket locations per brand."""
    ordered = summary.sort_values("stores")  # ascending: Plotly draws horizontal bars bottom-up

    figure = px.bar(
        ordered, x="stores", y="brand", orientation="h",
        color="brand", color_discrete_map=config.BRAND_COLORS, text="stores",
        title="Supermarket locations by brand",
    )

    figure.update_traces(
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>Stores: %{x:,}<extra></extra>",
    )
    figure.update_layout(showlegend=False, xaxis_title="Number of locations", yaxis_title="", height=380)

    return figure


def coverage_donut(kpis: dict) -> go.Figure:
    """Share of the population living in a postcode with and without a Migros."""
    covered = kpis["total_population"] - kpis["population_without_migros"]

    figure = go.Figure(go.Pie(
        labels=["Lives in a postcode with a Migros", "No Migros in the postcode"],
        values=[covered, kpis["population_without_migros"]],
        hole=0.62,
        # Orange for the covered share, muted grey for the gap: the accent marks Migros
        marker=dict(colors=[config.COLORS["primary"], config.COLORS["grid"]]),
        hovertemplate="<b>%{label}</b><br>%{value:,.0f} residents (%{percent})<extra></extra>",
    ))

    figure.update_layout(
        title="Population coverage",
        annotations=[dict(text=f"{100 - kpis['share_without_migros']:.0f}%<br>covered",
                          showarrow=False, font=dict(size=20, color=config.COLORS["text"]))],
        height=380, showlegend=False,
    )

    return figure


# --------------------------------------------------------------------------------------
# Canton screening
# --------------------------------------------------------------------------------------

def canton_bubble(cantons: pd.DataFrame) -> go.Figure:
    """Coverage against purchasing power, split into quadrants by the medians."""
    median_income = cantons["avg_income_per_taxpayer"].median()
    median_coverage = cantons["migros_per_100k"].median()

    figure = px.scatter(
        cantons,
        x="avg_income_per_taxpayer", y="migros_per_100k",
        size="population", color="competitors_per_100k",
        text="primary_canton", hover_name="canton_name",
        size_max=70, color_continuous_scale=config.SEQUENTIAL_SCALE,
        custom_data=["canton_name", "population", "avg_income_per_taxpayer", "migros_count",
                     "competitor_count", "migros_per_100k", "competitors_per_100k"],
        title="Migros coverage and purchasing power by canton",
    )

    figure.update_traces(
        textposition="middle center",
        textfont=dict(size=10, color=config.COLORS["text"]),
        marker=dict(opacity=0.85, line=dict(color="white", width=1.5)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Population: %{customdata[1]:,.0f}<br>"
            "Average income: CHF %{customdata[2]:,.0f}<br>"
            "Migros stores: %{customdata[3]:,.0f} (%{customdata[5]:.2f} per 100k)<br>"
            "Competitors: %{customdata[4]:,.0f} (%{customdata[6]:.2f} per 100k)"
            "<extra></extra>"
        ),
    )

    # Median lines turn the scatter into four readable strategy quadrants
    figure.add_vline(x=median_income, line_dash="dash", line_width=1, line_color=config.COLORS["muted"])
    figure.add_hline(y=median_coverage, line_dash="dash", line_width=1, line_color=config.COLORS["muted"])

    # The quadrant that matters: high income, weak coverage
    figure.add_annotation(
        x=cantons["avg_income_per_taxpayer"].max(), y=cantons["migros_per_100k"].min(),
        text="<b>POTENTIAL OPPORTUNITY</b>", showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(size=13, color=config.COLORS["primary"]),
    )

    figure.update_layout(
        xaxis_title="Average income per taxpayer (CHF)",
        yaxis_title="Migros stores per 100,000 residents",
        coloraxis_colorbar_title="Competitors<br>per 100k",
        height=560,
    )
    figure.update_xaxes(tickprefix="CHF ", tickformat=",")

    return figure


# --------------------------------------------------------------------------------------
# Postcode gaps
# --------------------------------------------------------------------------------------

def postcode_facets(candidates: pd.DataFrame, cantons: list[str], top_n: int = 10) -> go.Figure:
    """One panel per screened canton with its largest uncovered postcodes."""
    figure = make_subplots(rows=1, cols=len(cantons), shared_yaxes=True, horizontal_spacing=0.05,
                           subplot_titles=[f"{canton}" for canton in cantons])

    # One shared colour range, otherwise the same income would look different per panel
    income_min = candidates["avg_income_per_taxpayer"].min()
    income_max = candidates["avg_income_per_taxpayer"].max()

    for column, canton in enumerate(cantons, start=1):
        panel = candidates[candidates["primary_canton"] == canton].nlargest(top_n, "population")

        figure.add_trace(
            go.Bar(
                x=panel["postal_code"], y=panel["population"],
                marker=dict(color=panel["avg_income_per_taxpayer"], coloraxis="coloraxis",
                            line=dict(color="white", width=1)),
                text=panel["population"], texttemplate="%{text:,.0f}",
                textposition="outside", cliponaxis=False,
                customdata=panel[["locality_name", "avg_income_per_taxpayer", "external_competitor_count"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]} (%{x})</b><br>"
                    "Population: %{y:,.0f}<br>"
                    "Average income: CHF %{customdata[1]:,.0f}<br>"
                    "External competitors: %{customdata[2]:,.0f}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=column,
        )

    figure.update_layout(
        title=f"Largest postcodes without a Migros (top {top_n} per canton)",
        coloraxis=dict(colorscale=config.SEQUENTIAL_SCALE, cmin=income_min, cmax=income_max,
                       colorbar=dict(title="Average income<br>per taxpayer", tickprefix="CHF ", tickformat=",.0f")),
        height=520, bargap=0.2,
    )
    figure.update_xaxes(type="category", tickangle=-45, title_text="Postcode")
    figure.update_yaxes(title_text="Population", tickformat=",", row=1, col=1)

    return figure


def opportunity_ranking(scored: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """The weighted-score ranking of candidate postcodes."""
    top = scored.head(top_n).sort_values("opportunity_score")  # ascending for bottom-up bars

    figure = px.bar(
        top, x="opportunity_score", y="rank_label", orientation="h",
        color="primary_canton",
        color_discrete_sequence=[config.COLORS["primary"], config.COLORS["secondary"], config.COLORS["positive"]],
        text="opportunity_score",
        custom_data=["rank", "population", "avg_income_per_taxpayer",
                     "nearest_migros_distance_km", "external_competitor_count"],
        title=f"Top {top_n} postcode opportunities (weighted score)",
    )

    figure.update_traces(
        texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Score: %{x:.1f}<br>"
            "Population: %{customdata[1]:,.0f}<br>"
            "Average income: CHF %{customdata[2]:,.0f}<br>"
            "Nearest Migros: %{customdata[3]:.2f} km<br>"
            "External competitors: %{customdata[4]:,.0f}"
            "<extra></extra>"
        ),
    )

    figure.update_layout(
        xaxis_title="Opportunity score", yaxis_title="",
        legend_title="Canton", height=max(420, 34 * top_n), margin=dict(l=230),
    )

    return figure


# --------------------------------------------------------------------------------------
# Statistical model
# --------------------------------------------------------------------------------------

def odds_ratio_bar(effects: pd.DataFrame) -> go.Figure:
    """Effect size of each model feature, as an odds ratio per standard deviation."""
    figure = px.bar(
        effects, x="odds_ratio_per_1_sd", y="label", orientation="h",
        text="odds_ratio_per_1_sd", color="odds_ratio_per_1_sd",
        color_continuous_scale=config.SEQUENTIAL_SCALE,
        title="What predicts the presence of a Migros?",
    )

    figure.update_traces(texttemplate="%{text:.2f}x", textposition="outside", cliponaxis=False,
                         hovertemplate="<b>%{y}</b><br>Odds ratio per 1 SD: %{x:.2f}<extra></extra>")

    # Log scale: the population effect is an order of magnitude bigger than the others
    figure.update_layout(
        xaxis=dict(title="Odds ratio per 1 standard deviation (log scale)", type="log"),
        yaxis_title="", coloraxis_showscale=False, height=400, margin=dict(l=230),
    )

    # Reference line at 1 = the feature has no effect
    figure.add_vline(x=1, line_dash="dash", line_width=1, line_color=config.COLORS["muted"])

    return figure


def probability_bar(candidates: pd.DataFrame) -> go.Figure:
    """Predicted probability that each shortlisted postcode should contain a Migros."""
    plot_data = candidates.copy()
    plot_data["probability_pct"] = plot_data["predicted_probability"] * 100
    plot_data = plot_data.sort_values("probability_pct")

    figure = px.bar(
        plot_data, x="probability_pct", y="location_label", orientation="h",
        text="probability_pct", color="probability_pct",
        color_continuous_scale=config.SEQUENTIAL_SCALE,
        custom_data=["population", "avg_income_per_taxpayer", "external_competitor_count"],
        title="Modelled probability that this postcode should have a Migros",
    )

    figure.update_traces(
        texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Model probability: %{x:.1f}%<br>"
            "Population: %{customdata[0]:,.0f}<br>"
            "Average income: CHF %{customdata[1]:,.0f}<br>"
            "External competitors: %{customdata[2]:,.0f}"
            "<extra></extra>"
        ),
    )

    figure.update_layout(xaxis=dict(title="Predicted probability (%)", range=[0, 112]),
                         yaxis_title="", coloraxis_showscale=False, height=420, margin=dict(l=200))

    return figure


# --------------------------------------------------------------------------------------
# Catchment
# --------------------------------------------------------------------------------------

def catchment_bars(profile: pd.DataFrame) -> go.Figure:
    """Store counts by ring around one postcode."""
    long_form = profile.melt(
        id_vars="radius_km",
        value_vars=["migros_stores", "denner_stores", "external_competitors"],
        var_name="store_type", value_name="stores",
    )
    long_form["store_type"] = long_form["store_type"].replace({
        "migros_stores": "Migros", "denner_stores": "Denner", "external_competitors": "External competitors",
    })

    figure = px.bar(
        long_form, x="radius_km", y="stores", color="store_type", barmode="group", text="stores",
        color_discrete_map={"Migros": config.BRAND_COLORS["Migros"],
                            "Denner": config.BRAND_COLORS["Denner"],
                            "External competitors": config.COLORS["secondary"]},
        title=f"Supermarket coverage around {profile['location'].iloc[0]}",
    )

    figure.update_traces(textposition="outside", cliponaxis=False,
                         hovertemplate="<b>%{fullData.name}</b><br>Within %{x} km: %{y:,.0f} stores<extra></extra>")
    figure.update_layout(xaxis=dict(title="Catchment radius (km)", tickvals=sorted(profile["radius_km"])),
                         yaxis_title="Number of stores", legend_title="Store type", height=420)

    return figure


def catchment_comparison(profiles: list[pd.DataFrame]) -> go.Figure:
    """Two locations side by side on population, Migros Group coverage and competition."""
    panels = [
        ("population", "Catchment population", ",.0f"),
        ("migros_group_per_100k", "Migros Group per 100k", ".1f"),
        ("competitors_per_100k", "Competitors per 100k", ".1f"),
    ]

    figure = make_subplots(rows=1, cols=3, horizontal_spacing=0.09,
                           subplot_titles=[title for _, title, _ in panels])

    # One colour per location, reused across all three panels so the eye can track it
    palette = [config.COLORS["secondary"], config.COLORS["primary"]]

    for index, profile in enumerate(profiles):
        location = profile["location"].iloc[0]

        for column, (field, _, fmt) in enumerate(panels, start=1):
            figure.add_trace(
                go.Scatter(
                    x=profile["radius_km"], y=profile[field],
                    mode="lines+markers+text",
                    name=location, legendgroup=location,
                    showlegend=(column == 1),  # one legend entry per location, not per panel
                    line=dict(width=3, color=palette[index % len(palette)]),
                    marker=dict(size=10, color=palette[index % len(palette)]),
                    text=profile[field], texttemplate="%{text:" + fmt + "}", textposition="top center",
                    hovertemplate=f"<b>{location}</b><br>Radius: %{{x}} km<br>Value: %{{y:{fmt}}}<extra></extra>",
                ),
                row=1, col=column,
            )

    figure.update_layout(title="Catchment comparison", height=460, legend_title="Location")
    figure.update_xaxes(title_text="Radius (km)", tickvals=sorted(profiles[0]["radius_km"]))
    figure.update_yaxes(tickformat=",", row=1, col=1)

    return figure


# --------------------------------------------------------------------------------------
# Site selection maps
# --------------------------------------------------------------------------------------

def site_map(result: dict, village_name: str) -> go.Figure:
    """Score surface, existing stores and the recommended sites, zoomed to the village."""
    surface, sites = result["surface"], result["sites"]
    competitors, demand = result["competitors"], result["demand"]

    figure = go.Figure()

    # Layer 1: the score surface, each grid cell weighted by its own score
    figure.add_trace(DensityMapTrace(
        lat=surface["latitude"], lon=surface["longitude"], z=surface["site_score"],
        radius=22, colorscale="YlOrRd", opacity=0.55, hoverinfo="skip",
        colorbar=dict(title="Site score"),
    ))

    # Layer 2: where the local customers live
    local_demand = demand[demand["is_local"]]
    figure.add_trace(ScatterMapTrace(
        lat=local_demand["latitude"], lon=local_demand["longitude"], mode="markers",
        marker=dict(size=16, color=config.COLORS["secondary"], opacity=0.7),
        text=local_demand["name"], customdata=local_demand["population"],
        hovertemplate="<b>%{text}</b><br>Residents: %{customdata:,.0f}<extra></extra>",
        name="Demand centres",
    ))

    # Layer 3: the competition, one trace per brand so the legend can switch them off
    nearby = competitors[competitors["distance_from_centre_km"] <= 4]
    for brand, brand_stores in nearby.groupby("store_group"):
        figure.add_trace(ScatterMapTrace(
            lat=brand_stores["latitude"], lon=brand_stores["longitude"], mode="markers",
            marker=dict(size=12, color=config.BRAND_COLORS.get(brand, config.COLORS["muted"])),
            text=brand + " - " + brand_stores["street"].fillna("address unknown").astype(str),
            hovertemplate="<b>%{text}</b><extra></extra>", name=brand,
        ))

    # Layer 4: the recommendations, added last so they sit on top
    figure.add_trace(ScatterMapTrace(
        lat=sites["latitude"], lon=sites["longitude"], mode="markers+text",
        marker=dict(size=24, color="#00E5FF"),
        text=["#" + str(rank) for rank in sites["site_rank"]],
        textfont=dict(size=14, color="#04101E"),
        customdata=sites[["captured_population", "nearest_existing_store", "distance_to_nearest_migros_km"]].to_numpy(),
        hovertemplate=(
            "<b>Suggested site %{text}</b><br>"
            "%{lat:.5f}, %{lon:.5f}<br>"
            "Expected customers: %{customdata[0]:,.0f}<br>"
            "Nearest store: %{customdata[1]}<br>"
            "Nearest Migros: %{customdata[2]:.2f} km"
            "<extra></extra>"
        ),
        name="Recommended sites",
    ))

    _apply_map_layout(figure, *result["centre"], zoom=13.4)
    figure.update_layout(title=f"Where to open the new store in {village_name}", height=640,
                         margin=dict(l=10, r=10, t=60, b=10),
                         legend=dict(x=0.01, y=0.99))

    return figure


def regional_map(result: dict) -> go.Figure:
    """The wider network: why the chosen village is a gap in the first place."""
    competitors, demand, sites = result["competitors"], result["demand"], result["sites"]

    figure = go.Figure()

    for brand, brand_stores in competitors.groupby("store_group"):
        figure.add_trace(ScatterMapTrace(
            lat=brand_stores["latitude"], lon=brand_stores["longitude"], mode="markers",
            marker=dict(size=10, color=config.BRAND_COLORS.get(brand, config.COLORS["muted"])),
            text=brand + " - " + brand_stores["city"].fillna("").astype(str),
            hovertemplate="<b>%{text}</b><extra></extra>", name=brand,
        ))

    # Square-root sizing stops a small town from swamping every village on the map
    figure.add_trace(ScatterMapTrace(
        lat=demand["latitude"], lon=demand["longitude"], mode="markers",
        marker=dict(size=np.sqrt(demand["population"]) / 4 + 6, color=config.COLORS["secondary"], opacity=0.4),
        text=demand["name"], customdata=demand["population"],
        hovertemplate="<b>%{text}</b><br>Residents: %{customdata:,.0f}<extra></extra>",
        name="Demand (population)",
    ))

    best = sites.iloc[0]
    figure.add_trace(ScatterMapTrace(
        lat=[best["latitude"]], lon=[best["longitude"]], mode="markers+text",
        marker=dict(size=22, color="#00E5FF"), text=["NEW"],
        textfont=dict(size=12, color="#04101E"),
        hovertemplate="<b>Recommended new Migros</b><br>%{lat:.5f}, %{lon:.5f}<extra></extra>",
        name="Recommended site",
    ))

    _apply_map_layout(figure, *result["centre"], zoom=10.9)
    figure.update_layout(title="Regional context: the existing supermarket network", height=600,
                         margin=dict(l=10, r=10, t=60, b=10), legend=dict(x=0.01, y=0.99))

    return figure

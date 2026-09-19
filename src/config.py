"""Central configuration: file paths, model constants and the single plot theme.

Everything that controls how the app looks or what it assumes lives here, so a change
made once is applied to every chart and every page.
"""

from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

# Project root = the folder that contains app.py (this file sits in <root>/src)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

MASTER_DATA_FILE = DATA_DIR / "migros_master_data.csv"
STORES_FILE = DATA_DIR / "switzerland_supermarkets_clean.csv"
LOCALITIES_ZIP = DATA_DIR / "ortschaftenverzeichnis_plz_4326_csv.zip"

# --------------------------------------------------------------------------------------
# Analysis constants
# --------------------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371  # mean Earth radius, converts angular distance to kilometres

# Cantons carried forward from the canton-level screening into the postcode analysis
SCREENED_CANTONS = ["ZH", "SZ", "AG"]

# Default weights of the opportunity score (the app lets the user change them)
DEFAULT_WEIGHTS = {"population": 0.40, "income": 0.25, "distance": 0.25, "low_competition": 0.10}

# Features of the logistic regression. Distance to the nearest Migros is deliberately
# absent: it is zero for every postcode that has one and would leak the target.
LOGIT_FEATURES = ["log_population", "income_10k", "external_competitor_count", "log_catchment_population"]

LOGIT_FEATURE_LABELS = {
    "log_population": "Population (log10)",
    "income_10k": "Income per taxpayer (CHF 10k)",
    "external_competitor_count": "External competitors in postcode",
    "log_catchment_population": "Population within 5 km (log10)",
}

# Huff gravity model - used for the within-Siebnen site search
HUFF_DISTANCE_DECAY = 2.0        # shoppers' distance sensitivity (2.0 is the grocery standard)
HUFF_DISTANCE_FLOOR_KM = 0.2     # avoids division by zero when a store sits on a demand point
NEW_STORE_ATTRACTIVENESS = 1.0   # a full-format Migros, same pull as an existing Migros or Coop

# Relative pulling power by store format (sales area proxy)
STORE_ATTRACTIVENESS = {"Migros": 1.0, "Coop": 1.0, "Aldi": 0.8, "Lidl": 0.8, "Denner": 0.5}

# Postcode 8854 covers two settlements; this is the assumed split of its residents
SIEBNEN_POPULATION_SHARE = 0.70
SIEBNEN_POSTCODE = "8854"
SIEBNEN_LOCALITY_ZIP_ID = 5088   # identifier of Siebnen in the official locality directory

# --------------------------------------------------------------------------------------
# Single visual theme, applied to every Plotly figure in the app
# --------------------------------------------------------------------------------------

COLORS = {
    "background": "#FFFFFF",
    "surface": "#F8FAFC",
    "text": "#334155",
    "muted": "#64748B",
    "grid": "#E2E8F0",
    "primary": "#F58220",    # Migros orange, the accent of the whole app
    "secondary": "#4A86E8",  # blue used for comparisons and highlights
    "positive": "#2C854B",
    "negative": "#D42222",
}

# Brand colours stay fixed so a brand always looks the same across charts
BRAND_COLORS = {
    "Migros": "#F58220",
    "Coop": "#E30613",
    "Denner": "#19A831",
    "Aldi": "#120DA9",
    "Lidl": "#8B0E70",
}

# Sequential scale for "more is more" quantities (population, income, scores)
SEQUENTIAL_SCALE = "Oranges"

# Diverging scale used where a midpoint matters
DIVERGING_SCALE = "RdBu_r"

FONT_FAMILY = "Arial, Helvetica, sans-serif"


def register_theme() -> None:
    """Register and activate the shared Plotly template.

    Called once at start-up; afterwards every figure inherits fonts, colours, grid
    styling and margins, so no chart has to repeat its own layout block.
    """
    template = go.layout.Template()

    template.layout = go.Layout(
        font=dict(family=FONT_FAMILY, size=13, color=COLORS["text"]),
        title=dict(font=dict(size=19, color=COLORS["text"]), x=0.02, xanchor="left"),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["surface"],
        colorway=[COLORS["primary"], COLORS["secondary"], COLORS["positive"],
                  COLORS["negative"], COLORS["muted"]],
        xaxis=dict(gridcolor=COLORS["grid"], zeroline=False, linecolor=COLORS["grid"],
                   title=dict(font=dict(size=13, color=COLORS["muted"]))),
        yaxis=dict(gridcolor=COLORS["grid"], zeroline=False, linecolor=COLORS["grid"],
                   title=dict(font=dict(size=13, color=COLORS["muted"]))),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor=COLORS["grid"], borderwidth=1),
        margin=dict(l=70, r=40, t=70, b=60),
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=12)),
    )

    pio.templates["migros"] = template

    # "plotly_white" supplies sensible defaults; "migros" overrides them
    pio.templates.default = "plotly_white+migros"

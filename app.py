"""Migros expansion analysis - Streamlit front end.

Run with:  streamlit run app.py

This file only wires things together: page setup, data loading, navigation. The
analysis lives in ``src/analysis.py``, the figures in ``src/charts.py`` and the page
content in ``src/views.py``.
"""

from __future__ import annotations

import streamlit as st

from src import analysis, config, views
from src.data_loader import load_localities, load_master_data, load_stores

# --------------------------------------------------------------------------------------
# Page setup - must be the first Streamlit call in the script
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Migros Expansion Analysis",
    page_icon="🛒",
    layout="wide",                      # the charts and tables need the full width
    initial_sidebar_state="expanded",
)

# Registers the shared Plotly template, so every figure is styled identically
config.register_theme()


# --------------------------------------------------------------------------------------
# Data - loaded once and cached for the whole session
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Preparing the analysis...")
def prepare_shortlist(_master, cantons: tuple[str, ...]) -> tuple:
    """Candidate postcodes plus the four best under the default weights.

    The shortlist is recomputed from the data rather than hard-coded, so the app stays
    correct if the underlying tables are ever refreshed. The leading underscore on
    ``_master`` tells Streamlit not to hash the dataframe on every rerun.
    """
    candidates = analysis.build_opportunity_table(_master, cantons)
    ranked = analysis.score_opportunities(candidates, config.DEFAULT_WEIGHTS)
    return candidates, ranked["postal_code"].head(4).tolist()


master = load_master_data()
stores = load_stores()
localities = load_localities()

candidates, shortlist = prepare_shortlist(master, tuple(config.SCREENED_CANTONS))


# --------------------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------------------

PAGES = [
    "National picture",
    "1. Canton screening",
    "2. Postcode gaps",
    "3. Weighted ranking",
    "4. Statistical model",
    "5. Catchment check",
    "6. Site selection",
    "Conclusion",
]

with st.sidebar:
    st.title("🛒 Migros Expansion")
    st.caption("Where should Migros open its next supermarket in Switzerland?")

    # A radio rather than tabs: the analysis is a sequence, and a radio shows the order
    page = st.radio("Analysis step", PAGES, label_visibility="collapsed")

    st.markdown("---")
    st.caption(
        f"**Data**\n\n"
        f"{len(master):,} postcodes · {len(stores):,} supermarkets\n\n"
        f"Sources: OpenStreetMap, Swiss Federal Statistical Office, Federal Tax Administration"
    )

st.title("Migros Store Expansion Analysis")
st.caption("From 3,170 postcodes down to one street corner in Siebnen")


# --------------------------------------------------------------------------------------
# Routing - each page is a single render call
# --------------------------------------------------------------------------------------

if page == PAGES[0]:
    views.render_overview(master)
elif page == PAGES[1]:
    views.render_cantons(master)
elif page == PAGES[2]:
    views.render_gaps(candidates)
elif page == PAGES[3]:
    views.render_ranking(candidates)
elif page == PAGES[4]:
    views.render_model(master, shortlist)
elif page == PAGES[5]:
    views.render_catchment(master, shortlist)
elif page == PAGES[6]:
    views.render_sites(master, stores, localities)
else:
    views.render_conclusion(master, shortlist)

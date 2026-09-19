"""Loading and preparing the datasets produced by notebooks 01-06.

Every loader is wrapped in ``st.cache_data`` so the CSVs are read once per session
instead of on every widget interaction, which is what keeps the app responsive.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import BallTree

from . import config


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points, or between a point and an array.

    Latitude/longitude differences are not distances, so every proximity calculation in
    the app goes through this function.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * config.EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner="Loading postcode master data...")
def load_master_data() -> pd.DataFrame:
    """Read the postcode master table (one row per postcode) and add model features."""
    # postal_code as string so leading zeros survive the round-trip
    df = pd.read_csv(config.MASTER_DATA_FILE, dtype={"postal_code": "string"})

    # Only postcodes with complete income, municipality and coordinate data can be scored
    df = df[df["complete_location_data"] == True].copy().reset_index(drop=True)

    # --- surrounding demand -------------------------------------------------------
    # A postcode is not an island: the population living within 5 km also decides
    # whether a store is viable, and it separates urban from rural context.
    coordinates_rad = np.radians(df[["postcode_latitude", "postcode_longitude"]].to_numpy())
    tree = BallTree(coordinates_rad, metric="haversine")
    neighbours = tree.query_radius(coordinates_rad, r=5 / config.EARTH_RADIUS_KM)

    population = df["population"].to_numpy()
    df["catchment_population_5km"] = [population[idx].sum() for idx in neighbours] - df["population"]

    # --- model features -----------------------------------------------------------
    # Both population variables span several orders of magnitude, so they are logged.
    df["log_population"] = np.log10(df["population"] + 1)
    df["log_catchment_population"] = np.log10(df["catchment_population_5km"] + 1)

    # One unit = CHF 10,000, which makes the odds ratio readable
    df["income_10k"] = df["avg_income_per_taxpayer"] / 10000

    # Binary target of the logistic regression
    df["has_migros_flag"] = df["has_migros"].astype(int)

    # Label used on charts, because one locality name can cover several postcodes
    df["location_label"] = df["locality_name"] + " (" + df["postal_code"] + ")"

    return df


@st.cache_data(show_spinner="Loading supermarket locations...")
def load_stores() -> pd.DataFrame:
    """Read the cleaned OpenStreetMap supermarket table and attach format weights."""
    stores = pd.read_csv(config.STORES_FILE)

    # Pulling power by format, used by the Huff model; unknown formats get a middle value
    stores["attractiveness"] = stores["store_group"].map(config.STORE_ATTRACTIVENESS).fillna(0.6)

    return stores


@st.cache_data(show_spinner="Loading locality directory...")
def load_localities() -> pd.DataFrame:
    """Read the official Swiss locality directory straight from its ZIP archive.

    It gives sub-postcode geography: which municipalities share a postcode and what
    share of its addresses each one holds. That is what makes a within-village
    analysis possible at all.
    """
    with zipfile.ZipFile(config.LOCALITIES_ZIP) as archive:
        # The archive holds one CSV plus a folder entry, so the CSV is picked explicitly
        csv_name = [name for name in archive.namelist() if name.lower().endswith(".csv")][0]
        with archive.open(csv_name) as handle:
            localities = pd.read_csv(handle, sep=";")  # semicolon-separated source file

    # "51.158 %" is text in the source; convert it to a 0-1 fraction
    localities["address_share"] = (
        localities["Adressenanteil"].astype(str).str.replace("%", "", regex=False).str.strip().astype(float) / 100
    )

    return localities


@st.cache_data(show_spinner=False)
def national_kpis(master: pd.DataFrame) -> dict:
    """Headline numbers shown on the overview page."""
    total_population = master["population"].sum()
    population_without_migros = master.loc[master["migros_count"] == 0, "population"].sum()

    return {
        "postcodes": len(master),
        "total_population": total_population,
        "migros_stores": int(master["migros_count"].sum()),
        "postcodes_with_migros": int((master["migros_count"] > 0).sum()),
        "postcodes_without_migros": int((master["migros_count"] == 0).sum()),
        "population_without_migros": population_without_migros,
        "share_without_migros": population_without_migros / total_population * 100,
    }


@st.cache_data(show_spinner=False)
def brand_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Store counts per brand, plus the Migros Group total (Migros + Denner)."""
    counts = {
        "Migros": master["migros_count"].sum(),
        "Coop": master["coop_count"].sum(),
        "Denner": master["denner_count"].sum(),
        "Aldi": master["aldi_count"].sum(),
        "Lidl": master["lidl_count"].sum(),
    }

    summary = pd.DataFrame({"brand": list(counts), "stores": list(counts.values())})

    # Share of all mapped supermarket locations
    summary["share_pct"] = summary["stores"] / summary["stores"].sum() * 100

    return summary.sort_values("stores", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def canton_summary(master: pd.DataFrame) -> pd.DataFrame:
    """Aggregate postcodes into cantons for the screening view."""
    # Population-weighted income: a plain mean would let tiny postcodes count as much as cities
    working = master.copy()
    working["income_x_population"] = working["avg_income_per_taxpayer"] * working["population"]

    cantons = working.groupby("primary_canton", as_index=False).agg(
        population=("population", "sum"),
        income_x_population=("income_x_population", "sum"),
        migros_count=("migros_count", "sum"),
        denner_count=("denner_count", "sum"),
        coop_count=("coop_count", "sum"),
        aldi_count=("aldi_count", "sum"),
        lidl_count=("lidl_count", "sum"),
    )

    cantons["avg_income_per_taxpayer"] = cantons["income_x_population"] / cantons["population"]

    # Denner belongs to Migros Group, so it is not counted as external competition
    cantons["competitor_count"] = cantons["coop_count"] + cantons["aldi_count"] + cantons["lidl_count"]

    # Densities make large and small cantons comparable
    cantons["migros_per_100k"] = cantons["migros_count"] / cantons["population"] * 100000
    cantons["competitors_per_100k"] = cantons["competitor_count"] / cantons["population"] * 100000

    cantons["canton_name"] = cantons["primary_canton"].map(CANTON_NAMES)

    return cantons.sort_values("population", ascending=False).reset_index(drop=True)


CANTON_NAMES = {
    "AG": "Aargau", "AI": "Appenzell Innerrhoden", "AR": "Appenzell Ausserrhoden", "BE": "Bern",
    "BL": "Basel-Landschaft", "BS": "Basel-Stadt", "FR": "Fribourg", "GE": "Geneva", "GL": "Glarus",
    "GR": "Graubunden", "JU": "Jura", "LU": "Lucerne", "NE": "Neuchatel", "NW": "Nidwalden",
    "OW": "Obwalden", "SG": "St. Gallen", "SH": "Schaffhausen", "SO": "Solothurn", "SZ": "Schwyz",
    "TG": "Thurgau", "TI": "Ticino", "UR": "Uri", "VD": "Vaud", "VS": "Valais", "ZG": "Zug",
    "ZH": "Zurich",
}

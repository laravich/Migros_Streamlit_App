"""The analytical core: opportunity scoring, the logistic model, catchment profiles
and the within-village site search.

Nothing in this module knows about Streamlit widgets - it takes dataframes and
parameters and returns dataframes, which keeps it testable and reusable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

from . import config
from .data_loader import haversine_km


# --------------------------------------------------------------------------------------
# 1. Weighted opportunity score (the notebook 07 approach)
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_opportunity_table(master: pd.DataFrame, cantons: tuple[str, ...]) -> pd.DataFrame:
    """Postcodes in the screened cantons that have no Migros, with their distance to one.

    The distance is measured to the nearest postcode that *does* contain a Migros, which
    is accurate enough for screening; the final site work uses real store coordinates.
    """
    candidates = master[master["primary_canton"].isin(cantons) & (master["migros_count"] == 0)].copy()

    covered = master[master["migros_count"] > 0]

    # A BallTree searches nearest neighbours on the sphere in O(log n) instead of
    # comparing every candidate against every covered postcode
    tree = BallTree(np.radians(covered[["postcode_latitude", "postcode_longitude"]]), metric="haversine")
    distance, index = tree.query(np.radians(candidates[["postcode_latitude", "postcode_longitude"]]), k=1)

    # BallTree returns radians; multiply by the Earth radius for kilometres
    candidates["nearest_migros_distance_km"] = distance.flatten() * config.EARTH_RADIUS_KM
    candidates["nearest_migros_postcode"] = covered.iloc[index.flatten()]["postal_code"].to_numpy()

    return candidates.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def score_opportunities(candidates: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Rank candidate postcodes with the four-factor weighted score.

    Weights are a parameter rather than a constant so the app can expose them as
    sliders - which also makes the subjectivity of this method visible to the viewer.
    """
    scored = candidates.copy()

    # Min-max scaling puts residents, francs, kilometres and store counts on one 0-1 range
    raw_columns = ["population", "avg_income_per_taxpayer", "nearest_migros_distance_km", "external_competitor_count"]
    scaled = MinMaxScaler().fit_transform(scored[raw_columns])
    scored[["population_score", "income_score", "distance_score", "competitor_level"]] = scaled

    # For the first three a high value is good; for competitors a low value was treated as good
    scored["low_competition_score"] = 1 - scored["competitor_level"]

    scored["opportunity_score"] = 100 * (
        scored["population_score"] * weights["population"]
        + scored["income_score"] * weights["income"]
        + scored["distance_score"] * weights["distance"]
        + scored["low_competition_score"] * weights["low_competition"]
    )

    scored = scored.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    scored["rank_label"] = "#" + scored["rank"].astype(str) + "  " + scored["location_label"]

    return scored


# --------------------------------------------------------------------------------------
# 2. Logistic regression
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner="Fitting the logistic regression...")
def fit_logit(master: pd.DataFrame) -> dict:
    """Fit the coverage model on all postcodes and return coefficients, metrics, scores.

    The model answers "does a postcode with this profile normally contain a Migros?".
    Applied to a postcode that has none, a high probability marks a genuine gap.
    """
    X = master[config.LOGIT_FEATURES]
    y = master["has_migros_flag"]

    # Hold out 30% to check the model generalises; stratify keeps the class balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

    # statsmodels needs an explicit intercept column
    trained = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=0)

    test_probability = trained.predict(sm.add_constant(X_test))
    test_prediction = (test_probability >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, test_prediction),
        "roc_auc": roc_auc_score(y_test, test_probability),
        "confusion": pd.DataFrame(
            confusion_matrix(y_test, test_prediction),
            index=["Actual: no Migros", "Actual: has Migros"],
            columns=["Predicted: no Migros", "Predicted: has Migros"],
        ),
    }

    # The hold-out confirmed the model, so refit on everything for stable coefficients
    model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)

    # Raw coefficients are log-odds on four different scales, so they are also expressed
    # as an odds ratio per one standard deviation, which makes the effects comparable
    standard_deviation = X.std()
    effects = pd.DataFrame({
        "feature": config.LOGIT_FEATURES,
        "label": [config.LOGIT_FEATURE_LABELS[f] for f in config.LOGIT_FEATURES],
        "coefficient": model.params[config.LOGIT_FEATURES].to_numpy(),
        "p_value": model.pvalues[config.LOGIT_FEATURES].to_numpy(),
        "odds_ratio_per_1_sd": np.exp(model.params[config.LOGIT_FEATURES] * standard_deviation).to_numpy(),
    }).sort_values("odds_ratio_per_1_sd", ascending=True)

    # Probability for every postcode; the gap index only applies where no Migros exists
    scored = master.copy()
    scored["predicted_probability"] = model.predict(sm.add_constant(X))
    scored["coverage_gap_index"] = np.where(scored["migros_count"] == 0, scored["predicted_probability"] * 100, np.nan)

    return {
        "effects": effects,
        "metrics": metrics,
        "pseudo_r2": model.prsquared,
        "observations": int(model.nobs),
        "scored": scored,
    }


# --------------------------------------------------------------------------------------
# 3. Catchment profile of a single postcode
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def catchment_profile(master: pd.DataFrame, postal_code: str, radii: tuple[int, ...] = (2, 5, 10)) -> pd.DataFrame:
    """Population and store counts inside rings around one postcode centre.

    Postcode-level counts alone are misleading: a postcode can show zero Migros while
    being surrounded by them. This is the check that separates a real gap from a
    narrow miss.
    """
    centre = master.loc[master["postal_code"] == postal_code].iloc[0]

    distance = haversine_km(
        centre["postcode_latitude"], centre["postcode_longitude"],
        master["postcode_latitude"], master["postcode_longitude"],
    )

    rows = []
    for radius in radii:
        inside = master[distance <= radius]

        population = inside["population"].sum()
        migros = inside["migros_count"].sum()
        denner = inside["denner_count"].sum()
        competitors = inside["external_competitor_count"].sum()

        rows.append({
            "radius_km": radius,
            "postcodes": inside["postal_code"].nunique(),
            "population": population,
            "migros_stores": migros,
            "denner_stores": denner,
            "external_competitors": competitors,
            # Densities let a village be compared with a city ring
            "migros_per_100k": migros / population * 100000 if population else np.nan,
            "migros_group_per_100k": (migros + denner) / population * 100000 if population else np.nan,
            "competitors_per_100k": competitors / population * 100000 if population else np.nan,
        })

    profile = pd.DataFrame(rows)
    profile["location"] = f"{centre['locality_name']} ({postal_code})"

    return profile


# --------------------------------------------------------------------------------------
# 4. Within-village site search (Huff gravity model)
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_demand_points(master: pd.DataFrame, localities: pd.DataFrame, postal_code: str,
                        neighbour_radius_km: float = 8.0) -> pd.DataFrame:
    """Where the customers live, at a finer resolution than the postcode itself.

    Inside the target postcode the residents are spread over its sub-areas using the
    official address shares; everything within ``neighbour_radius_km`` is added at its
    own postcode centre as spill-over demand.
    """
    target = master.loc[master["postal_code"] == postal_code].iloc[0]
    total_population = float(target["population"])

    parts = localities[localities["PLZ4"] == int(postal_code)]

    rows = []
    for _, part in parts.iterrows():
        # A postcode can hold two settlements; the main one keeps the assumed majority share
        if part["ZIP_ID"] == config.SIEBNEN_LOCALITY_ZIP_ID:
            weight = config.SIEBNEN_POPULATION_SHARE * part["address_share"]
            is_local = True
        else:
            weight = (1 - config.SIEBNEN_POPULATION_SHARE) * part["address_share"]
            is_local = False

        rows.append({
            "name": f"{part['Ortschaftsname']} ({part['Gemeindename']})",
            "latitude": part["N"],
            "longitude": part["E"],
            "population": total_population * weight,
            "is_local": is_local,
        })

    # Neighbouring postcodes: people from the next village also shop at a new store
    distance = haversine_km(target["postcode_latitude"], target["postcode_longitude"],
                            master["postcode_latitude"], master["postcode_longitude"])
    neighbours = master[(distance < neighbour_radius_km) & (master["postal_code"] != postal_code)]

    for _, neighbour in neighbours.iterrows():
        rows.append({
            "name": neighbour["location_label"],
            "latitude": neighbour["postcode_latitude"],
            "longitude": neighbour["postcode_longitude"],
            "population": float(neighbour["population"]),
            "is_local": False,
        })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Scoring candidate sites...")
def search_sites(master: pd.DataFrame, stores: pd.DataFrame, localities: pd.DataFrame,
                 postal_code: str, number_of_sites: int = 3,
                 minimum_separation_km: float = 0.45) -> dict:
    """Score a 100 m grid over the village and return the best separated sites.

    Score = expected customers captured (Huff) x accessibility along the main retail road.
    """
    target = master.loc[master["postal_code"] == postal_code].iloc[0]
    centre_lat, centre_lon = target["postcode_latitude"], target["postcode_longitude"]

    demand = build_demand_points(master, localities, postal_code)
    local_demand = demand[demand["is_local"]]

    # Competition: every store within 10 km. Further away barely affects a local choice.
    stores = stores.copy()
    stores["distance_from_centre_km"] = haversine_km(centre_lat, centre_lon, stores["latitude"], stores["longitude"])
    competitors = stores[stores["distance_from_centre_km"] <= 10].copy()

    # --- the grid -----------------------------------------------------------------
    # 0.0007 deg latitude is about 78 m, 0.0010 deg longitude about 76 m at this latitude
    latitudes = np.arange(local_demand["latitude"].min() - 0.010, local_demand["latitude"].max() + 0.010, 0.0007)
    longitudes = np.arange(local_demand["longitude"].min() - 0.016, local_demand["longitude"].max() + 0.016, 0.0010)
    grid = np.array([(lat, lon) for lat in latitudes for lon in longitudes])

    # --- feasibility filters ------------------------------------------------------
    # 1: inside the built-up village rather than on the surrounding fields and slopes
    distance_to_village = np.min([haversine_km(grid[:, 0], grid[:, 1], row["latitude"], row["longitude"])
                                  for _, row in local_demand.iterrows()], axis=0)

    # 2: not on top of an existing competitor - that parcel is taken
    distance_to_store = np.min([haversine_km(grid[:, 0], grid[:, 1], row["latitude"], row["longitude"])
                                for _, row in competitors.iterrows()], axis=0)

    # 3: near the main retail road. Every supermarket in the village sits on it, so the
    # line connecting them is a usable proxy without an external road dataset.
    axis = competitors[competitors["distance_from_centre_km"] <= 2].sort_values("longitude")
    axis_points = axis[["latitude", "longitude"]].to_numpy()
    distance_to_axis = np.min([_distance_to_segment_km(grid, axis_points[i], axis_points[i + 1])
                               for i in range(len(axis_points) - 1)], axis=0)

    keep = (distance_to_village <= 0.9) & (distance_to_store >= 0.25) & (distance_to_axis <= 1.0)
    grid, distance_to_axis, distance_to_store = grid[keep], distance_to_axis[keep], distance_to_store[keep]

    # --- Huff model ---------------------------------------------------------------
    # Pull already exerted on each demand point by the existing network (constant per point)
    existing_pull = np.array([
        np.sum(competitors["attractiveness"].to_numpy() /
               np.maximum(haversine_km(row["latitude"], row["longitude"],
                                       competitors["latitude"].to_numpy(), competitors["longitude"].to_numpy()),
                          config.HUFF_DISTANCE_FLOOR_KM) ** config.HUFF_DISTANCE_DECAY)
        for _, row in demand.iterrows()
    ])

    captured = np.zeros(len(grid))
    for i, point in demand.reset_index(drop=True).iterrows():
        distance = np.maximum(haversine_km(grid[:, 0], grid[:, 1], point["latitude"], point["longitude"]),
                              config.HUFF_DISTANCE_FLOOR_KM)
        new_pull = config.NEW_STORE_ATTRACTIVENESS / distance ** config.HUFF_DISTANCE_DECAY

        # Share of this area's residents the new store would win from the existing network
        captured += point["population"] * new_pull / (new_pull + existing_pull[i])

    # A parcel far off the main road loses passing trade, so the raw capture is discounted
    accessibility = np.exp(-distance_to_axis / 0.35)

    surface = pd.DataFrame({
        "latitude": grid[:, 0],
        "longitude": grid[:, 1],
        "captured_population": captured,
        "distance_to_axis_km": distance_to_axis,
        "distance_to_nearest_store_km": distance_to_store,
        "accessibility": accessibility,
    })
    surface["site_score"] = surface["captured_population"] * surface["accessibility"]
    surface = surface.sort_values("site_score", ascending=False).reset_index(drop=True)

    # --- pick separated sites ------------------------------------------------------
    # The best cells are neighbours of each other, so a minimum spacing is enforced;
    # otherwise the "three options" would all sit on the same parcel.
    chosen: list[pd.Series] = []
    for _, cell in surface.iterrows():
        if all(haversine_km(cell["latitude"], cell["longitude"], other["latitude"], other["longitude"]) > minimum_separation_km
               for other in chosen):
            chosen.append(cell)
        if len(chosen) == number_of_sites:
            break

    sites = pd.DataFrame(chosen).reset_index(drop=True)
    sites["site_rank"] = sites.index + 1

    # Describe each site by a landmark, because coordinates alone are hard to picture
    migros_stores = competitors[competitors["store_group"] == "Migros"]
    landmarks, migros_distances = [], []
    for _, site in sites.iterrows():
        distances = haversine_km(site["latitude"], site["longitude"],
                                 competitors["latitude"].to_numpy(), competitors["longitude"].to_numpy())
        nearest = competitors.iloc[int(np.argmin(distances))]
        street = f"{nearest['street']} {nearest['house_number']}" if pd.notna(nearest["street"]) else "address unknown"
        landmarks.append(f"{nearest['name']}, {street} ({distances.min() * 1000:.0f} m)")
        migros_distances.append(haversine_km(site["latitude"], site["longitude"],
                                             migros_stores["latitude"].to_numpy(),
                                             migros_stores["longitude"].to_numpy()).min())

    sites["nearest_existing_store"] = landmarks
    sites["distance_to_nearest_migros_km"] = migros_distances

    return {
        "surface": surface,
        "sites": sites,
        "demand": demand,
        "competitors": competitors,
        "centre": (centre_lat, centre_lon),
    }


def _distance_to_segment_km(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Shortest distance from each point to a straight segment, in km.

    A local flat-earth approximation is enough here: the segments are a few hundred
    metres long, where curvature is irrelevant.
    """
    km_per_degree = 111.32
    lon_scale = np.cos(np.radians(points[:, 0].mean()))  # longitude degrees shrink with latitude

    px = (points[:, 1] - start[1]) * km_per_degree * lon_scale
    py = (points[:, 0] - start[0]) * km_per_degree
    bx = (end[1] - start[1]) * km_per_degree * lon_scale
    by = (end[0] - start[0]) * km_per_degree

    # Project onto the segment and clamp, so the closest point never lies beyond its ends
    t = np.clip((px * bx + py * by) / (bx * bx + by * by), 0, 1)

    return np.hypot(px - t * bx, py - t * by)

# Migros Store Expansion Analysis — Streamlit app

Interactive front end for the notebook pipeline (01–08): from 3,170 Swiss postcodes
down to a single recommended street corner in Siebnen (8854).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Project structure

```
.
├── app.py                  # entry point: page setup, data loading, navigation only
├── requirements.txt
├── .streamlit/
│   └── config.toml         # UI theme, matched to the chart theme
├── data/                   # inputs produced by notebooks 01–06
│   ├── migros_master_data.csv
│   ├── switzerland_supermarkets_clean.csv
│   └── ortschaftenverzeichnis_plz_4326_csv.zip
└── src/
    ├── config.py           # paths, model constants, the single Plotly theme
    ├── data_loader.py      # cached loading + feature engineering
    ├── analysis.py         # scoring, logistic regression, catchment, Huff site search
    ├── charts.py           # every figure, themed in one place
    └── views.py            # one render function per page
```

The separation matters: `analysis.py` never imports a widget, so the same functions can
be reused in a notebook or a test. `charts.py` owns all styling, so a colour changes in
one place. `views.py` decides what to show; `app.py` only routes.

## Pages

| Page | Shows |
|---|---|
| National picture | KPIs, stores per brand, population coverage |
| 1. Canton screening | Coverage vs purchasing power quadrant chart, full canton table |
| 2. Postcode gaps | Largest uncovered postcodes per screened canton (adjustable depth) |
| 3. Weighted ranking | Four-factor score with live weight sliders |
| 4. Statistical model | Logistic regression: metrics, odds ratios, coverage-gap probabilities |
| 5. Catchment check | Ring comparison of any two shortlisted postcodes |
| 6. Site selection | Huff-scored 100 m grid over Siebnen, map and downloadable coordinates |
| Conclusion | Recommendation and limitations |

Every chart is paired with the small table behind it, so any number on screen can be
checked against its source rows.

## Method in one paragraph

Canton-level screening narrows Switzerland to Zurich, Schwyz and Aargau. Within those,
postcodes without a Migros are ranked twice: once with a hand-weighted opportunity score,
and once with a logistic regression trained on all postcodes that learns what a "Migros
postcode" looks like (population, income, competitor presence, 5 km catchment). A high
model probability plus zero actual stores marks a genuine gap — Siebnen 8854. Its
catchment is then checked ring by ring, and a Huff gravity model scores a 100 m grid over
the village to pick the actual site.

## Known assumptions

- The population split inside postcode 8854 (70% Siebnen / 30% Galgenen) is an assumption.
- No land-use or zoning data: the model finds demand, not available parcels.
- The main retail road is inferred from existing store positions.
- Huff parameters (β = 2, format weights) are literature defaults.

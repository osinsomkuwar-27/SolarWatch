# SolarWatch — Solar Flare Prediction using Aditya-L1 (SoLEXS and HEL1OS)

Live solar flare nowcasting and forecasting console, built for ISRO's Bharatiya Antariksh Hackathon (BAH) 2026 — Problem Statement: forecasting solar flares using Aditya-L1 data. An end-to-end ML pipeline covering raw FITS ingestion, parsing, feature engineering, flare detection, model training, evaluation, and a backend/frontend layer for serving live predictions.

Live Demo: https://solar-ashen-two.vercel.app/

Python · FastAPI · React · TensorFlow · MIT License

## What is this?

SolarWatch is two things at once, by design:

1. A **deployed, working product** — a live dashboard that ingests real-time Pradan GOES-18 X-ray flux and runs it through a trained 1D CNN + LSTM model to nowcast current flare class and forecast M-class-or-larger (M+) flare probability at 1, 3, and 6-hour horizons.
2. An **Aditya-L1 data pipeline** — this repository turns raw FITS files from Aditya-L1's SoLEXS and HEL1OS X-ray spectrometer payloads into a trained, evaluable flare prediction model, with a documented plan to make Aditya-L1 the product's primary data source.

Load raw instrument data, parse it, build multi-day datasets, engineer features, detect flares, train a model, evaluate it, then serve predictions to a user-facing interface — while the live dashboard runs today on GOES-18 so the product is usable now, not only after Aditya-L1 integration is complete.

Built around ISRO's Aditya-L1 mission, India's first dedicated solar observation satellite, positioned at the Sun-Earth L1 Lagrange point.

## Why GOES-18 today, and Aditya-L1 next

GOES X-ray sensors are the long-standing standard for flare classification, and most existing research is built around them. We use GOES-18 as the production model's current data source because it's a continuously available, well-understood feed that lets the product work end-to-end today. We're building toward Aditya-L1 because:

- Aditya-L1 observes from L1, a different vantage point and continuity profile than GOES's geostationary orbit
- SoLEXS (soft X-ray) and HEL1OS (high-energy X-ray) together enable cross-instrument analysis that a single-instrument pipeline cannot
- Open-source tooling for Aditya-L1 data is far less developed than for GOES, so this pipeline fills a real gap
- As ISRO's first dedicated solar mission, and the actual subject of our hackathon problem statement, Aditya-L1 benefits from open pipelines that make its data usable beyond ISRO's internal teams

This is not a claim that Aditya-L1 is superior to GOES for all purposes — it's a statement of project scope and rollout order.

## Live Application

The deployed app has four views:

- **Live Dashboard** — real-time GOES-18 X-ray flux, current flare class, system status, and 1h/3h/6h M+ flare probability
- **History** — 7-day X-ray flux history and a catalog of detected C/M/X-class flare events
- **Model Insights** — full model architecture and per-class (A/B/C/M/X) precision/recall/F1 performance
- **Aditya-L1** — ISRO Aditya-L1 mission and SoLEXS instrument details, plus our integration roadmap

## Model

Current production model: a 1D CNN + LSTM, trained on 14 years (2010–2024) of NOAA GOES X-ray flux, validated on a held-out 12-month window covering the ascending phase of Solar Cycle 25, achieving 91.3% overall validation accuracy.
Model: 1D CNN + LSTM (Solar Flare Classifier)
Framework: TensorFlow

Input Layer
  Shape: (256, 2)
  256 timesteps × 2 channels [soft X-ray flux, hard X-ray flux]

Layer 1  : Conv1D    filters=64,  kernel_size=5, activation=ReLU
Layer 2  : Conv1D    filters=128, kernel_size=5, activation=ReLU
Layer 3  : MaxPool1D pool_size=2
Layer 4  : Conv1D    filters=128, kernel_size=3, activation=ReLU
Layer 5  : MaxPool1D pool_size=2
Layer 6  : LSTM      units=128, return_sequences=True
Layer 7  : LSTM      units=64
Layer 8  : Dense     units=32, activation=ReLU
           Dropout(rate=0.3)
Layer 9  : Dense     units=5, activation=Softmax
           Output classes: [A, B, C, M, X]

Training data : NOAA GOES X-ray flux, 14 years (2010–2024)
Validation    : Held-out 12-month window (Solar Cycle 25, ascending phase)
Validation accuracy : 91.3%

Derived outputs (served to frontend):
  - Current flare class (argmax of softmax)
  - M+ flare probability, 1-hour horizon
  - M+ flare probability, 3-hour horizon
  - M+ flare probability, 6-hour horizon

  Full per-class precision/recall/F1 is published on the Model Insights page of the live app.

## Aditya-L1 Integration Plan

1. Stream SoLEXS soft X-ray (1–22 keV) as input channel 1
2. Stream HEL1OS hard X-ray as input channel 2
3. Retrain the LSTM head on a combined GOES + Aditya-L1 tensor (256 × 4 channels)
4. Cross-validate predictions against SUIT UV imager active-region data
5. Switch the primary data source from NOAA GOES-18 to Aditya-L1 once L1 telemetry stabilizes

## Features

- Modular loaders for raw Aditya-L1 FITS files
- Dedicated FITS parsing for SoLEXS and HEL1OS payload formats
- Caching layer to avoid redundant reprocessing
- Multi-day dataset construction from daily FITS files
- Exploratory data analysis for SoLEXS and HEL1OS independently
- Cross-instrument analysis combining both payloads
- Flare detection on parsed light curve data
- Feature engineering tailored to X-ray flux time series
- Dataset generation for supervised ML training
- Model training and evaluation modules, with published per-class metrics
- Inference module for predictions on new data
- Backend integration layer (FastAPI) exposing pipeline outputs and predictions
- Deployed React frontend (Live Dashboard, History, Model Insights, Aditya-L1) for visualizing data and predictions

## Architecture

The pipeline runs as a sequence of stages, not a microservices architecture. Each stage is a separate, independently runnable module. The backend sits in front of the trained model to serve live predictions to the frontend; today it serves GOES-18-derived predictions, and will serve Aditya-L1-derived predictions once that path is production-ready.

### Stage Map

| Stage | Module location | Responsibility |
|---|---|---|
| Loading | `ml/` | Load raw SoLEXS / HEL1OS FITS files |
| Parsing | `ml/` | Parse instrument-specific FITS structures |
| Caching | `ml/` | Avoid reprocessing unchanged files |
| Multi-day construction | `ml/` | Assemble daily data into continuous time series |
| EDA | `ml/` | SoLEXS, HEL1OS, and cross-instrument analysis |
| Flare detection | `ml/` | Identify flare events in time series |
| Feature engineering | `ml/` | Derive model-ready features |
| Dataset generation | `ml/` | Produce labeled, ML-ready dataset |
| Training | `ml/` | Train flare prediction model |
| Evaluation | `ml/` | Evaluate trained model, produce per-class metrics |
| Inference | `ml/` | Generate predictions on new data |
| Backend | `backend/` | Serve predictions and pipeline outputs via API (FastAPI) |
| Frontend | `frontend/` | Live Dashboard, History, Model Insights, Aditya-L1 views |

> Exact module file names within `ml/` and `backend/` are not all enumerated here — refer to those directories directly.

## Folder Structure

```

SolarWatch/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── crud.py
│   │   ├── database.py
│   │   ├── main.py                  # FastAPI app entry point
│   │   ├── models.py                # DB / ORM models
│   │   ├── schemas.py                # Pydantic request/response schemas
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── api.py                # API route definitions
│   │   └── services/                 # (contents not confirmed — GitHub API
│   │                                  #  rate-limited during inspection)
│   ├── mock/
│   │   └── latest_prediction.json    # Mock prediction response for local/dev use
│   └── requirements.txt              # Backend-specific Python dependencies
│
├── config/                           # (contents not confirmed — GitHub API
│                                      #  rate-limited during inspection)
│
├── frontend/                         # React 19 / TanStack Start dashboard
│                                      # (subtree not enumerated in this pass;
│                                      #  package.json confirms React 19, TanStack
│                                      #  Start/Router, Tailwind v4, shadcn/ui, Recharts)
│
├── ml/
│   ├── __init__.py
│   ├── inference.py                  # Inference entry point
│   ├── train_and_save.py             # Model training + artifact saving script
│   ├── data/                         # (contents not confirmed — GitHub API
│   │                                  #  rate-limited during inspection)
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── build_dataset.py
│   │   ├── builder.py
│   │   └── config.py
│   ├── eda/
│   │   ├── __init__.py
│   │   ├── flare_detector.py         # Flare event detection logic
│   │   ├── light_curve_plotter.py    # Light curve visualization
│   │   ├── run_eda.py                # EDA entry point
│   │   ├── statistics.py             # Statistical analysis utilities
│   │   └── helios_eda/               # (contents not confirmed — rate-limited)
│   ├── features/                     # (contents not confirmed — GitHub API
│   │                                  #  rate-limited during inspection)
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── helios_loader.py          # HEL1OS FITS loader
│   │   ├── multi_day_loader.py       # Multi-day dataset assembly
│   │   └── solexs_loader.py          # SoLEXS FITS loader
│   ├── pipeline/
│   │   └── ml_mechanics_test.py      # Pipeline mechanics test script
│   └── utils/
│       ├── __init__.py
│       └── config.py                 # Shared ML utility config
│
├── tests/
│   └── unit/                         # (contents not confirmed — rate-limited)
│
├── .env.example
├── .gitignore
├── LICENSE
├── PRD.md
├── README.md
├── pyproject.toml
└── requirements.txt

```

## Setup

### Prerequisites

- Python 3.11+ (see `pyproject.toml`)
- Node.js and npm (for the frontend)
- Access to raw Aditya-L1 SoLEXS and HEL1OS FITS files (not bundled with this repository)
- Git

### 1. Clone the repo

```bash
git clone https://github.com/osinsomkuwar-27/SolarWatch.git
cd SolarWatch
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Set up environment variables

Copy `.env.example` to `.env` and fill in any required values:

```bash
cp .env.example .env
```

### 4. Prepare the Aditya-L1 dataset (optional — not needed to run the live GOES-18 dashboard)

Place raw SoLEXS and HEL1OS FITS files in the input directory expected by `config/`, then run loading, parsing, multi-day construction, feature extraction, flare detection, and dataset generation in that order (see "Running Individual Modules" below).

### 5. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

## Running Individual Modules

```bash
# Load and parse raw FITS files
python -m ml.loader --instrument solexs --date-range 2024-01-01:2024-01-31

# Run SoLEXS exploratory data analysis
python -m ml.eda_solexs

# Run HEL1OS exploratory data analysis
python -m ml.eda_hel1os

# Run cross-instrument analysis
python -m ml.cross_instrument_analysis

# Run flare detection
python -m ml.flare_detection

# Run feature engineering
python -m ml.feature_engineering

# Generate the final training dataset
python -m ml.dataset_builder
```

> Module names and flags are illustrative of the pipeline's stages. Confirm exact names and arguments against the actual files in `ml/`.

## Running the Complete Pipeline

```bash
python -m ml.pipeline --config config/config.yaml
```

> Replace with the actual entry point if one exists. Otherwise, run the individual modules above in order.

## Model Training

```bash
python -m ml.train --dataset-path <path-to-generated-dataset> --config config/config.yaml
```

## Evaluation

```bash
python -m ml.evaluate --model-path <path-to-trained-model> --dataset-path <path-to-test-dataset>
```

## Inference

```bash
python -m ml.infer --model-path <path-to-trained-model> --input <path-to-new-data>
```

## Backend

```bash
uvicorn backend.app:app --reload
```

> Confirm the actual entry point against the code in `backend/`. Built on FastAPI/Uvicorn per `requirements.txt`.

The backend is responsible for:

- Routing inference requests to the trained model
- Validating request data against the model's expected feature schema
- Serving processed pipeline data (live flux, historical flux, detected flares, model metrics) to the frontend
- Reading configuration for model artifact, dataset, and live feed source locations

## Tests

```bash
pytest
```

## Integrations

| Integration type | Description |
|---|---|
| External data (production) | Live NOAA SWPC GOES-18 X-ray flux feed |
| External data (pipeline) | Raw Aditya-L1 SoLEXS / HEL1OS FITS files, sourced from ISRO mission data |
| Internal (pipeline ↔ backend ↔ frontend) | Backend invokes trained models and reads pipeline outputs; frontend calls backend API |
| Tooling | `requirements.txt` / `pyproject.toml` (Python), npm (frontend), `tests/unit` (testing) |

> No third-party CI/CD, hosting, or monitoring integrations beyond Vercel (frontend) are documented yet — update this table as they're added.

## Output Directories

Pipeline stages produce: cached parsed FITS data, multi-day assembled time series, engineered feature sets, flare detection labels, final ML-ready datasets, trained model artifacts, and evaluation reports. Exact paths are defined in `config/`.

## Technologies Used

| Layer | Technology |
|---|---|
| Frontend | React 19, TypeScript, TanStack Start/Router, TanStack Query, Tailwind CSS v4, shadcn/ui, Recharts, Vite |
| Backend | Python, FastAPI, Uvicorn |
| ML pipeline | Python, TensorFlow, scikit-learn, Astropy, pandas, numpy, scipy |
| Data formats | FITS (SoLEXS, HEL1OS), live NOAA GOES-18 flux feed |
| Deployment | Vercel (frontend) |
| Testing | pytest |

## References

- ISRO Aditya-L1 Mission — https://www.isro.gov.in/Aditya_L1.html
- SoLEXS and HEL1OS payload documentation, as published by ISRO
- NOAA SWPC GOES X-ray flux — https://www.swpc.noaa.gov/

## Future Improvements

- Complete SoLEXS + HEL1OS channel integration and retrain the model on a 4-channel input tensor
- Switch the primary data source from GOES-18 to Aditya-L1
- Cross-validate with SUIT UV imager data
- Formalize the backend inference layer into a documented, versioned prediction API
- Add automated, threshold-based alerting on the live dashboard
- Expand cross-instrument analysis
- Add CI/CD and integration test coverage across pipeline, backend, and frontend
- Improve dataset versioning as multi-day data volume grows

## Team

Built by Team VIHANG for ISRO's Bharatiya Antariksh Hackathon 2026.

## License

MIT License — see [LICENSE](./LICENSE) for details.
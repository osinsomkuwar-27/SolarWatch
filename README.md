# Solar Flare Prediction using Aditya-L1 (SoLEXS and HEL1OS)

Predict solar flares from ISRO's Aditya-L1 mission data. An end-to-end ML pipeline covering raw FITS ingestion, parsing, feature engineering, flare detection, model training, evaluation, and a backend/frontend layer for serving predictions.

Python · FastAPI/Backend · React · MIT License

## What is this?

This repository turns raw FITS files from Aditya-L1's SoLEXS and HEL1OS X-ray spectrometer payloads into a trained, evaluable flare prediction model — and exposes that model through a backend API and a React frontend.

Load raw instrument data, parse it, build multi-day datasets, engineer features, detect flares, train a model, evaluate it, then serve predictions to a user-facing interface.

Built around ISRO's Aditya-L1 mission, India's first dedicated solar observation satellite, positioned at the Sun-Earth L1 Lagrange point.

## Why Aditya-L1 instead of GOES

GOES X-ray sensors are the long-standing standard for flare classification, and most existing research is built around them. This project works with Aditya-L1 data instead because:

- Aditya-L1 observes from L1, a different vantage point and continuity profile than GOES's geostationary orbit
- SoLEXS (soft X-ray) and HEL1OS (high-energy X-ray) together enable cross-instrument analysis that a single-instrument pipeline cannot
- Open-source tooling for Aditya-L1 data is far less developed than for GOES, so this pipeline fills a real gap
- As ISRO's first dedicated solar mission, Aditya-L1 benefits from open pipelines that make its data usable beyond ISRO's internal teams

This is not a claim that Aditya-L1 is superior to GOES for all purposes — it is a statement of project scope.

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
- Model training and evaluation modules
- Inference module for predictions on new data
- Backend integration layer exposing pipeline outputs and predictions
- React frontend (in development on a separate branch) for visualizing data and predictions

## Architecture

The pipeline runs as a sequence of stages, not a microservices architecture. Each stage is a separate, independently runnable module, and the backend sits in front of the trained model to serve predictions to the frontend.

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
| Evaluation | `ml/` | Evaluate trained model |
| Inference | `ml/` | Generate predictions on new data |
| Backend | `backend/` | Serve predictions and pipeline outputs via API |
| Frontend | `shreeja/frontend` branch | Visualize data and predictions for end users |

> Exact module file names within `ml/` and `backend/` are not enumerated here — refer to those directories directly.

## Folder Structure

```
solar/
├── backend/                # Backend integration and prediction-serving layer
├── config/                 # Centralized configuration for the pipeline
├── frontend/                # React frontend (UI for data visualization and predictions)
├── ml/                      # Core ML pipeline: loaders, parsing, EDA, feature
│                              # engineering, flare detection, dataset generation,
│                              # training, evaluation, and inference modules
├── tests/
│   └── unit/                # Unit tests
├── .env.example              # Example environment variable configuration
├── pyproject.toml            # Project metadata and build configuration
├── requirements.txt          # Python dependencies
├── README.md
└── LICENSE
```

## Setup

### Prerequisites

- Python 3.x (see `pyproject.toml` for the supported version range)
- Node.js and a package manager (for the frontend, on `shreeja/frontend`)
- Access to raw Aditya-L1 SoLEXS and HEL1OS FITS files (not bundled with this repository)
- Git

### 1. Clone the repo

```bash
git clone https://github.com/osinsomkuwar-27/solar.git
cd solar
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

### 4. Prepare the dataset

Place raw SoLEXS and HEL1OS FITS files in the input directory expected by `config/`, then run loading, parsing, multi-day construction, feature extraction, flare detection, and dataset generation in that order (see "Running Individual Modules" below).

### 5. Check out the frontend (optional, in development)

```bash
git fetch origin shreeja/frontend
git checkout shreeja/frontend
cd <frontend-directory>
npm install
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
python -m backend.app
```

> Confirm the actual entry point and framework (e.g. Flask, FastAPI) against the code in `backend/`.

The backend is responsible for:

- Routing inference requests to the trained model
- Validating request data against the model's expected feature schema
- Serving processed pipeline data (light curves, detected flares) to the frontend
- Reading configuration for model artifact and dataset locations

## Frontend

```bash
git checkout shreeja/frontend
cd <frontend-directory>
npm install
npm start
```

> The frontend branch has not been merged into `main`; confirm directory name and scripts directly on that branch.

The frontend is responsible for:

- Visualizing SoLEXS and HEL1OS light curves and EDA outputs
- Displaying detected flare events
- Requesting and displaying predictions from the backend
- Providing a user-facing entry point into pipeline outputs

## Integrations

| Integration type | Description |
|---|---|
| External data | Raw Aditya-L1 SoLEXS / HEL1OS FITS files, sourced from ISRO mission data |
| Internal (pipeline ↔ backend ↔ frontend) | Backend invokes trained models and reads pipeline outputs; frontend calls backend API |
| Tooling | `requirements.txt` / `pyproject.toml` (Python), npm/yarn (frontend), `tests/unit` (testing) |

> No third-party CI/CD, hosting, or monitoring integrations are documented yet — update this table as they're added.

## Output Directories

Pipeline stages produce: cached parsed FITS data, multi-day assembled time series, engineered feature sets, flare detection labels, final ML-ready datasets, trained model artifacts, and evaluation reports. Exact paths are defined in `config/`.

## Technologies Used

| Layer | Technology |
|---|---|
| ML pipeline | Python |
| Data format | FITS (SoLEXS, HEL1OS) |
| Backend | Python (framework per `backend/`) |
| Frontend | React (`shreeja/frontend` branch) |

## References

- ISRO Aditya-L1 Mission — https://www.isro.gov.in/Aditya_L1.html
- SoLEXS and HEL1OS payload documentation, as published by ISRO

## Future Improvements

- Merge `shreeja/frontend` into `main`
- Formalize the backend inference layer into a documented prediction API
- Expand cross-instrument analysis
- Add CI/CD and integration test coverage across pipeline, backend, and frontend
- Improve dataset versioning as multi-day data volume grows

## License

MIT License — see [LICENSE](./LICENSE) for details.
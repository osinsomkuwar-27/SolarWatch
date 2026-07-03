# Product Requirements Document: SolarWatch — Solar Flare Prediction Pipeline

## Project Goal

Build and maintain a modular, end-to-end machine learning pipeline and product that:

- Ingests raw FITS data from ISRO's Aditya-L1 SoLEXS and HEL1OS payloads
- Produces clean, structured, multi-day time series data for both instruments
- Produces engineered features suitable for flare prediction modeling
- Detects flare events derived from instrument data
- Produces a labeled, ML-ready dataset
- Trains and evaluates flare prediction models
- Serves a production nowcasting/forecasting model to end users today, using NOAA GOES-18 X-ray flux as an interim, proven data source
- Exposes predictions and pipeline outputs through a backend integration layer
- Presents data and predictions to end users through a deployed frontend application

The pipeline must be reproducible, modular (each stage independently runnable and testable), and extensible to future instruments, models, backend endpoints, and frontend features without requiring redesign of existing modules. The product must remain usable end-to-end (live GOES-18-based dashboard) while the Aditya-L1 data path is built and validated in parallel.

## Functional Requirements

### Data Ingestion Requirements

- The system must load raw FITS files for SoLEXS and HEL1OS from a configurable input directory.
- The system must support loading data for a single day or a configurable multi-day date range.
- The system must distinguish between SoLEXS and HEL1OS file formats during ingestion.
- The system must support incremental ingestion, allowing newly added files to be loaded without reprocessing previously ingested files.
- The system must surface clear errors when expected files are missing or unreadable.
- The production system must separately ingest a live NOAA SWPC GOES-18 X-ray flux feed for the currently deployed dashboard, independent of the Aditya-L1 FITS ingestion path.

### FITS File Validation

- The system must validate that ingested files conform to expected FITS structure before parsing.
- The system must validate the presence of required headers and data extensions specific to SoLEXS and HEL1OS formats.
- The system must reject or quarantine files that fail validation, with a clear reason logged.
- The system must not silently proceed with partially corrupted or malformed FITS files.

### Multi-day Loading

- The system must assemble single-day parsed outputs into continuous multi-day datasets.
- The system must correctly handle gaps in coverage (missing days) without corrupting the resulting time series.
- The system must preserve timestamp continuity and timezone/time-standard consistency across assembled days.
- The system must support re-running multi-day assembly when new days are added to the input set.

### SoLEXS Processing

- The system must parse SoLEXS-specific FITS structures into a structured, analysis-ready representation.
- The system must support exploratory data analysis specific to SoLEXS soft X-ray (1–22 keV) measurements.
- The system must expose SoLEXS-derived light curves for downstream feature engineering and flare detection.

### HEL1OS Processing

- The system must parse HEL1OS-specific FITS structures into a structured, analysis-ready representation.
- The system must support exploratory data analysis specific to HEL1OS high-energy X-ray measurements.
- The system must expose HEL1OS-derived light curves for downstream feature engineering and flare detection.

### Cross Instrument Analysis

- The system must support joint analysis of temporally aligned SoLEXS and HEL1OS data.
- The system must handle differences in sampling cadence and coverage between the two instruments when aligning data.
- The system must surface discrepancies between instruments (e.g., non-overlapping coverage windows) rather than silently dropping data.

### Feature Engineering

- The system must derive features from SoLEXS and HEL1OS time series suitable for flare prediction modeling.
- The system must support feature extraction independently for each instrument and for cross-instrument combinations.
- The system must keep feature engineering logic decoupled from flare detection and dataset generation logic.
- The system must allow new features to be added without modifying upstream parsing or loading modules.

### Dataset Builder

- The system must combine engineered features and flare detection labels into a single ML-ready dataset.
- The system must support configurable train/validation/test splitting.
- The system must version or timestamp generated datasets to support reproducibility of training runs.
- The system must avoid data leakage across time when constructing splits for a time-series prediction task.

### ML Training

- The system must train flare prediction models using the dataset produced by the dataset builder.
- The system must support configuration-driven training (e.g., model parameters defined in `config/` rather than hardcoded).
- The system must persist trained model artifacts in a retrievable, identifiable form.
- The system must log training configuration alongside each trained model artifact for reproducibility.
- The production model (currently a 1D CNN + LSTM, input shape 256 timesteps × 2 channels [soft, hard] flux, 5-class softmax output [A, B, C, M, X]) must remain retrainable against a new input schema (256 × 4 channels) once Aditya-L1 channels are added, without requiring a redesign of the training entry point.

### Model Evaluation

- The system must evaluate trained models against held-out test data.
- The system must report evaluation results in a structured, inspectable format, including per-class precision, recall, and F1 for each flare class (A, B, C, M, X).
- The system must support evaluating multiple model artifacts against the same test dataset for comparison.
- The system must not silently pass evaluation when input data does not match the model's expected feature schema.
- Evaluation reports must be exposed to end users in a transparent, human-readable form (see Frontend Requirements — Model Insights).

### Inference Requirements

- The system must support running inference using a trained model artifact against new, unseen data.
- The system must validate that input data for inference matches the feature schema expected by the model.
- The system must support both batch inference and integration with the backend layer for on-demand prediction requests.
- The production inference path must return, at minimum: current flare class, current status, and M+ flare probability at 1-hour, 3-hour, and 6-hour horizons.

### Backend Requirements

- The backend must expose an API surface that allows external callers (including the frontend) to request predictions and retrieve pipeline outputs.
- The backend must validate and sanitize all incoming request payloads before passing them to the inference module.
- The backend must decouple model-serving logic from training/evaluation logic, so that retraining a model does not require redeploying the backend.
- The backend must return structured, well-defined error responses when a request cannot be fulfilled (e.g., invalid input, missing model artifact).
- The backend must be configurable (model artifact paths, dataset paths, serving parameters, live feed source) through the centralized configuration module rather than hardcoded values.
- The backend must support switching its live data source (currently NOAA GOES-18) to Aditya-L1 via configuration once that path is production-ready, without requiring frontend changes beyond the response schema.

### Frontend Requirements

- The frontend must allow users to view live X-ray flux, current flare class, and system status (Live Dashboard).
- The frontend must allow users to view 1-hour, 3-hour, and 6-hour M+ flare probability forecasts.
- The frontend must allow users to view historical X-ray flux (rolling 7-day window) and a catalog of detected C/M/X-class flare events (History).
- The frontend must allow users to view model architecture and per-class performance metrics for transparency (Model Insights).
- The frontend must present Aditya-L1 mission and instrument context, and the team's data-source integration roadmap (Aditya-L1 page).
- The frontend must allow users to view SoLEXS and HEL1OS light curves and associated exploratory data analysis outputs once that data path is live.
- The frontend must allow users to request and view predictions by communicating with the backend API.
- The frontend must handle backend errors (e.g., failed requests, validation errors) gracefully, presenting clear feedback to the user rather than failing silently.
- The frontend's API integration points (base URLs, endpoints) must be configurable per environment rather than hardcoded, to support local development, staging, and production backends.
- The frontend must be deployed and kept in sync with `main`; any active development branch must be merged promptly to avoid drift between deployed behavior and repository state.

### Visualization Requirements

- The system must support visualization of SoLEXS and HEL1OS light curves as part of exploratory data analysis.
- The system must support visualization of detected flare events overlaid on instrument time series.
- The frontend must render live flux, historical flux, and flare probability visualizations sourced from backend-provided data rather than duplicating pipeline computation in the frontend.
- Visualization components in the ML pipeline must remain separable from core data processing logic so that the pipeline can run headlessly (without generating plots) when required.

### Reporting Requirements

- The system must support generating evaluation reports summarizing model performance on test data, including per-class precision/recall/F1.
- Reports must reference the dataset version, training window, and model artifact used, to ensure traceability.
- The system must avoid embedding unverified or placeholder metrics in generated reports or in the Model Insights page.

### Configuration Requirements

- The system must centralize configuration (paths, parameters, instrument settings, live feed source, backend serving settings) in the `config/` module rather than scattering configuration across individual scripts.
- The system must support environment-specific configuration via `.env` files, following the pattern established in `.env.example`.
- The frontend must support its own environment-specific configuration (e.g., backend API base URL) independent of the Python-side `.env` configuration.
- The system must allow configuration overrides without modifying source code.

### Logging Requirements

- The system must log key pipeline events (ingestion start/end, validation failures, parsing errors, training start/completion, evaluation results) at each stage.
- The backend must log incoming requests, validation failures, and inference errors with sufficient context for debugging.
- The system must distinguish log severity levels (e.g., info, warning, error) so that failures are not silently mixed with normal operation logs.
- Logs must include sufficient context (file names, date ranges, module name, request identifiers where applicable) to support debugging without re-running the pipeline.

### Performance Requirements

- The caching layer must avoid redundant reprocessing of previously parsed FITS files.
- Multi-day dataset construction must scale to the date ranges required for model training without requiring full reprocessing of unrelated days.
- Feature engineering and dataset generation must be runnable incrementally as new data becomes available, rather than requiring a full pipeline rerun from raw FITS files each time.
- The backend must respond to prediction and data requests within a reasonable time for interactive frontend use, given the size of the underlying dataset and model.
- The live dashboard's polling/feed mechanism must not degrade page responsiveness under normal operation.

### Error Handling

- The system must fail loudly (with clear error messages) on invalid input, missing files, or schema mismatches, rather than failing silently.
- The system must isolate failures at one stage (e.g., a single corrupted FITS file) from halting processing of unrelated, valid data where feasible.
- The system must provide actionable error messages that reference the specific file, date, or module involved.
- The backend must translate internal pipeline/model errors into well-formed API error responses rather than leaking internal stack traces to callers.
- The frontend must handle and display backend error responses without crashing or leaving the user without feedback.

### Scalability

- The pipeline must be structured so that additional days, date ranges, or instrument data can be incorporated without redesigning existing modules.
- The architecture must support extension to additional instruments or data sources in the future without requiring changes to unrelated modules (e.g., adding a new instrument should not require changes to the dataset builder's core logic beyond integration points).
- The backend must be structured so that additional endpoints (e.g., new prediction types, new data views) can be added without restructuring existing endpoints.
- The frontend must be structured so that new visualizations or views can be added without requiring changes to unrelated components.

### Security

- Credentials, API keys, and environment-specific secrets must be managed via `.env` files (backend) and frontend environment configuration, and must not be committed to version control.
- The backend must validate and sanitize all inputs received for inference and data requests.
- Access to any exposed prediction API must be controlled appropriately for the deployment context (e.g., authentication/authorization at the backend layer, as applicable).
- The frontend must not embed sensitive credentials or secrets in client-side code; any required keys must be proxied through the backend.

### Deployment

- The backend integration module must expose trained model inference in a form consumable by external callers, including the frontend.
- Deployment configuration (e.g., model artifact location, serving parameters, live feed source, frontend API base URL) must be managed through configuration rather than hardcoded into backend or frontend code.
- The pipeline components (training/evaluation) must be separable from the serving components (backend/inference) so that retraining does not require redeploying the entire backend.
- The frontend must be deployable independently of the backend's deployment cycle, communicating only through its configured API integration point.
- The frontend is currently deployed to Vercel and must remain deployable there or to an equivalent static/SSR host without structural changes to `backend/`, `config/`, or `ml/`, beyond well-defined API integration points.

### Future Enhancements

- Complete the SoLEXS + HEL1OS channel integration and retrain the production model on a combined GOES + Aditya-L1 tensor.
- Switch the primary live data source from NOAA GOES-18 to Aditya-L1 once L1 telemetry is stable.
- Cross-validate flare predictions against SUIT UV imager active-region data.
- Add automated, threshold-based alerting on the live dashboard.
- Extend cross-instrument analysis to support additional derived diagnostics beyond current SoLEXS/HEL1OS comparisons.
- Formalize the backend inference module into a documented, versioned prediction API.
- Expand automated testing beyond current unit test coverage to include integration tests across pipeline, backend, and frontend.
- Evaluate additional model architectures within the existing training and evaluation framework.
- Improve dataset versioning and lineage tracking as the volume of multi-day data grows.
- Add CI/CD integration for automated testing and deployment of backend and frontend components.

---

This document describes functional and non-functional requirements for the SolarWatch pipeline and product, including its currently deployed Pradan-based live dashboard, its backend integration layer, its React frontend, and its in-development Aditya-L1 SoLEXS/HEL1OS data pipeline. It does not introduce new modules beyond those already present or planned in the repository (configuration, utilities, loaders, FITS parsing, caching, SoLEXS EDA, HEL1OS EDA, cross-instrument analysis, flare detection, feature engineering, dataset generation, model training, evaluation, inference, backend integration, and the frontend application).
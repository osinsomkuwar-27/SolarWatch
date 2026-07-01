import { queryOptions } from "@tanstack/react-query";

export const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export interface DataQuality {
  status: string;
  reason: string | null;
  lookback_seconds_available: number;
  lookback_seconds_required: number;
}

export interface Nowcast {
  flare_probability: number;
  flare_class: string;
  confidence: number;
  is_flare_active: boolean;
}

export interface Forecast {
  flare_probability_30min: number;
  flare_probability_60min: number;
  predicted_class: string;
  estimated_onset_minutes: number | null;
}

export interface RawFeatures {
  slx_counts: number | null;
  hardness_ratio: number | null;
  hardness_smoothed: number | null;
  dCR_dt: number | null;
  d2CR_dt2: number | null;
  ema_60s: number | null;
  ema_300s: number | null;
  neupert_corr: number | null;
  flare_phase: string | null;
  cdte_broadband: number | null;
  czt_broadband: number | null;
  photon_index_fit: number | null;
}

export interface Prediction {
  id: number;
  timestamp: string;
  model_version: string;
  data_quality: DataQuality;
  nowcast: Nowcast;
  forecast: Forecast;
  raw_features: RawFeatures;
  created_at: string;
}

export interface LightCurvePoint {
  timestamp: string;
  slx_counts: number | null;
  cdte_broadband: number | null;
  czt_broadband: number | null;
  hardness_ratio: number | null;
  flare_phase: string | null;
}

export interface StatusResponse {
  scheduler_running: boolean;
  last_prediction_at: string | null;
  last_file_modified_at: string | null;
  total_predictions_today: number;
  alert_active: boolean;
}

async function apiFetch<T>(path: string): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`API fetch failed on ${path}: ${response.statusText} (${response.status})`);
  }
  return response.json();
}

export const api = {
  getLatestPrediction: () => apiFetch<Prediction>("/api/latest"),
  getRecentPredictions: (n = 100) => apiFetch<Prediction[]>("/api/predictions?n=" + n),
  getRecentLightCurve: (n = 300) => apiFetch<LightCurvePoint[]>("/api/lightcurve?n=" + n),
  getStatus: () => apiFetch<StatusResponse>("/api/status"),
};

// React Query Options
export const apiQueries = {
  latestPrediction: () =>
    queryOptions({
      queryKey: ["latestPrediction"],
      queryFn: () => api.getLatestPrediction(),
      refetchInterval: 10000,
    }),
  recentPredictions: (n = 100) =>
    queryOptions({
      queryKey: ["recentPredictions", n],
      queryFn: () => api.getRecentPredictions(n),
      refetchInterval: 15000,
    }),
  lightCurve: (n = 300) =>
    queryOptions({
      queryKey: ["lightCurve", n],
      queryFn: () => api.getRecentLightCurve(n),
      refetchInterval: 15000,
    }),
  status: () =>
    queryOptions({
      queryKey: ["status"],
      queryFn: () => api.getStatus(),
      refetchInterval: 10000,
    }),
};

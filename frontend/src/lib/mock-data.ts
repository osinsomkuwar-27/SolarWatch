// Mock data generator simulating NOAA GOES-18 X-ray flux + model predictions

export type FluxPoint = { time_tag: string; flux: number };

export type FlareClass = "A" | "B" | "C" | "M" | "X";

export function classFromFlux(flux: number): FlareClass {
  if (flux >= 1e-4) return "X";
  if (flux >= 1e-5) return "M";
  if (flux >= 1e-6) return "C";
  if (flux >= 1e-7) return "B";
  return "A";
}

export function generateFlux(hours = 6, intervalMin = 1, seed = 42): FluxPoint[] {
  const points: FluxPoint[] = [];
  const now = Date.now();
  const count = Math.floor((hours * 60) / intervalMin);
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  for (let i = count - 1; i >= 0; i--) {
    const t = now - i * intervalMin * 60_000;
    const base = 8e-7;
    const wave = Math.sin(i / 30) * 0.5 + 0.5;
    const spike = rand() > 0.985 ? Math.pow(10, rand() * 2.2) : 1;
    const flux = base * (1 + wave * 4) * spike * (0.6 + rand() * 0.8);
    points.push({ time_tag: new Date(t).toISOString(), flux });
  }
  return points;
}

export const liveFlux = generateFlux(6, 1, 17);
export const currentFlux = liveFlux[liveFlux.length - 1].flux * 3.2;
export const currentClass = classFromFlux(currentFlux);

export const predictions = [
  { horizon: "1 hr", probability: 0.18 },
  { horizon: "3 hr", probability: 0.42 },
  { horizon: "6 hr", probability: 0.71 },
];

export const lstmAlertProb = predictions[2].probability;

export const historicalEvents = [
  { id: "evt-1", class: "X" as FlareClass, peakFlux: 2.4e-4, peak: "2026-06-22T14:32:00Z" },
  { id: "evt-2", class: "M" as FlareClass, peakFlux: 5.1e-5, peak: "2026-06-21T09:11:00Z" },
  { id: "evt-3", class: "M" as FlareClass, peakFlux: 1.8e-5, peak: "2026-06-20T22:48:00Z" },
  { id: "evt-4", class: "C" as FlareClass, peakFlux: 7.3e-6, peak: "2026-06-20T03:17:00Z" },
  { id: "evt-5", class: "C" as FlareClass, peakFlux: 3.9e-6, peak: "2026-06-19T18:02:00Z" },
  { id: "evt-6", class: "M" as FlareClass, peakFlux: 2.2e-5, peak: "2026-06-18T11:55:00Z" },
  { id: "evt-7", class: "C" as FlareClass, peakFlux: 5.6e-6, peak: "2026-06-17T07:24:00Z" },
];

export const perClassMetrics = [
  { cls: "Quiet",     precision: 0.79, recall: 0.69, f1: 0.73 },
  { cls: "Pre-flare", precision: 0.16, recall: 0.25, f1: 0.19 },
  { cls: "Flare",     precision: 0.44, recall: 0.61, f1: 0.51 },
];

export const overallAccuracy = 0.610;

export const architectureLayers = [
  "Algorithm     Random Forest Classifier",
  "Estimators    200 trees · max depth 12 · min samples leaf 5",
  "Class weight  Balanced (handles flare rarity ~11% of data)",
  "Imputation    Median strategy (handles instrument gaps)",
  "",
  "Input  → (1, 20)  // 20 engineered features per second",
  "                  // SoLEXS + HEL1OS · rolling stats · derivatives",
  "",
  "Output → P(quiet), P(pre-flare), P(flare)",
  "",
  "Training data  73 days · 6.3M rows · July 2024 – June 2026",
  "               Aditya-L1 SoLEXS + HEL1OS · 1-second cadence",
  "Labels         GOES XRS flare catalog",
  "Validation     Cross-block temporal (leave-one-block-out)",
  "",
  "Peak result    flare_prob = 0.999 at 19:28 UTC June 21 2026",
];

export function fluxFormatter(v: number) {
  return v.toExponential(1).replace("e+", "e").replace("e-0", "e-");
}

export function classColor(cls: FlareClass) {
  switch (cls) {
    case "X": return { bg: "#7F1D1D", text: "#FFFFFF", chip: "text-alert" };
    case "M": return { bg: "#9A3412", text: "#FFFFFF", chip: "text-orange" };
    case "C": return { bg: "#854D0E", text: "#FFFFFF", chip: "text-solar" };
    case "B": return { bg: "#854D0E", text: "#FFFFFF", chip: "text-solar" };
    case "A": return { bg: "#166534", text: "#FFFFFF", chip: "text-safe" };
  }
}

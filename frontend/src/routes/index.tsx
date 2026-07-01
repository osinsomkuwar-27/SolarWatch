import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiQueries } from "@/lib/api";
import { AlertPanel } from "@/components/AlertPanel";
import { FlareClassBadge } from "@/components/FlareClassBadge";
import { FluxChart } from "@/components/FluxChart";
import { PredictionCards } from "@/components/PredictionCards";
import { SolarCycleWidget } from "@/components/SolarCycleWidget";
import { fluxFormatter } from "@/lib/mock-data";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Live Dashboard — SolarWatch" },
      {
        name: "description",
        content:
          "Live solar X-ray flux, current flare class, and 1/3/6-hour M+ flare probability from the SolarWatch CNN+LSTM model.",
      },
      { property: "og:title", content: "Live Dashboard — SolarWatch" },
      {
        property: "og:description",
        content: "Real-time mission-control view of solar flare risk.",
      },
    ],
  }),
  component: Dashboard,
});

function Dashboard() {
  const [timeStr, setTimeStr] = useState<string>("");

  const {
    data: latestPrediction,
    error: latestError,
    isLoading: latestLoading,
  } = useQuery(apiQueries.latestPrediction());

  const { data: status } = useQuery(apiQueries.status());
  const { data: lightCurve } = useQuery(apiQueries.lightCurve(300));

  useEffect(() => {
    const tick = () => {
      const d = new Date(Math.floor(Date.now() / 60000) * 60000);
      setTimeStr(d.toUTCString().replace("GMT", "UTC"));
    };
    tick();
    const id = setInterval(tick, 30_000);
    return () => clearInterval(id);
  }, []);

  if (latestLoading) {
    return (
      <div className="flex h-[80vh] items-center justify-center font-mono text-sm text-text-muted">
        Connecting to SolarWatch telemetry...
      </div>
    );
  }

  if (latestError) {
    return (
      <div className="mx-auto max-w-[1200px] px-8 py-10 flex flex-col gap-6 font-mono text-destructive">
        <h1 className="text-2xl font-bold">Telemetry Connection Lost</h1>
        <p>{latestError.message || "Failed to load live forecast data from the backend."}</p>
      </div>
    );
  }

  // Derive parameters from live API data
  const isFlareActive = latestPrediction?.nowcast.is_flare_active ?? false;
  const currentClass = (latestPrediction?.nowcast.flare_class ?? "A") as any;
  const currentFlux = latestPrediction?.raw_features.slx_counts ?? 0;
  const nowcastProb = latestPrediction?.nowcast.flare_probability ?? 0;
  const forecast30 = latestPrediction?.forecast.flare_probability_30min ?? 0;
  const forecast60 = latestPrediction?.forecast.flare_probability_60min ?? 0;

  // Format light curve points for FluxChart
  const chartData = lightCurve
    ? lightCurve.map((pt) => ({
        time_tag: pt.timestamp,
        flux: pt.slx_counts || 0,
      }))
    : [];

  const showAdvisory = forecast60 >= 0.6 || isFlareActive;
  const advisoryProb = isFlareActive ? nowcastProb : forecast60;

  return (
    <div className="mx-auto max-w-[1200px] px-4 sm:px-8 py-6 sm:py-10 flex flex-col gap-8">
      <header className="flex flex-col gap-2 fade-in">
        <span className="font-mono text-xs uppercase tracking-[0.22em] text-sky">
          Live · Nowcast
        </span>
        <h1 className="font-serif text-5xl text-foreground leading-[1.02]">
          Solar weather <em className="text-sky">at a glance</em>
        </h1>
        <p className="text-base text-text-muted max-w-2xl">
          Real-time solar flare forecasting and nowcasting for ISRO satellite protection. Alerts fire 15–45 minutes before impact.
        </p>
      </header>

      {showAdvisory && <AlertPanel probability={advisoryProb} />}

      <section className="card-surface card-tint-sky p-6 fade-in">
        <div className="flex flex-col md:flex-row md:items-center gap-8">
          <FlareClassBadge cls={currentClass} />
          <div className="flex-1 grid grid-cols-1 sm:grid-cols-3 gap-6">
            <div>
              <div className="font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
                {currentFlux > 1.0 ? "SoLEXS Count Rate" : "Current flux (1–8 Å)"}
              </div>
              <div className="font-serif text-3xl text-solar mt-1">
                {currentFlux > 1.0 ? currentFlux.toLocaleString() : fluxFormatter(currentFlux)}
                <span className="text-text-muted text-lg font-sans ml-1">
                  {currentFlux > 1.0 ? "counts/s" : "W/m²"}
                </span>
              </div>
            </div>
            <div>
              <div className="font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
                Active region
              </div>
              <div className="font-serif text-2xl text-foreground mt-1">AR 13842</div>
              <div className="text-sm text-text-muted">
                {latestPrediction?.raw_features.flare_phase
                  ? `Phase: ${latestPrediction.raw_features.flare_phase}`
                  : "βγδ magnetic complexity"}
              </div>
            </div>
            <div>
              <div className="font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
                Last update
              </div>
              <div
                className="font-mono text-sm text-foreground mt-1 min-h-[1.25rem]"
                suppressHydrationWarning
              >
                {latestPrediction
                  ? new Date(latestPrediction.timestamp).toUTCString().replace("GMT", "UTC")
                  : "—"}
              </div>
              <div className="text-sm text-text-muted">
                {status?.scheduler_running ? "Live stream active" : "Offline data feed"}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section>
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="font-serif text-xl text-foreground">Forecast horizons</h2>
          <span className="font-mono text-xs uppercase tracking-wider text-text-faint">
            M+ probability
          </span>
        </div>
        <PredictionCards
          nowcastProbability={nowcastProb}
          forecast30MinProbability={forecast30}
          forecast60MinProbability={forecast60}
        />
      </section>

      <FluxChart data={chartData} />

      <SolarCycleWidget />
    </div>
  );
}

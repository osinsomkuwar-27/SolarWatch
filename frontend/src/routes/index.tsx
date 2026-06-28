import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { AlertPanel } from "@/components/AlertPanel";
import { FlareClassBadge } from "@/components/FlareClassBadge";
import { FluxChart } from "@/components/FluxChart";
import { PredictionCards } from "@/components/PredictionCards";
import { SolarCycleWidget } from "@/components/SolarCycleWidget";
import {
  currentClass,
  currentFlux,
  fluxFormatter,
  liveFlux,
  lstmAlertProb,
} from "@/lib/mock-data";

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
  useEffect(() => {
    const tick = () => {
      const d = new Date(Math.floor(Date.now() / 60000) * 60000);
      setTimeStr(d.toUTCString().replace("GMT", "UTC"));
    };
    tick();
    const id = setInterval(tick, 30_000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="mx-auto max-w-[1200px] px-8 py-10 flex flex-col gap-8">
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

      {lstmAlertProb >= 0.6 && <AlertPanel probability={lstmAlertProb} />}

      <section className="card-surface card-tint-sky p-6 fade-in">
        <div className="flex flex-col md:flex-row md:items-center gap-8">
          <FlareClassBadge cls={currentClass} />
          <div className="flex-1 grid grid-cols-1 sm:grid-cols-3 gap-6">
            <div>
              <div className="font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
                Current flux (1–8 Å)
              </div>
              <div className="font-serif text-3xl text-solar mt-1">
                {fluxFormatter(currentFlux)}
                <span className="text-text-muted text-lg font-sans ml-1">W/m²</span>
              </div>
            </div>
            <div>
              <div className="font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
                Active region
              </div>
              <div className="font-serif text-2xl text-foreground mt-1">AR 13842</div>
              <div className="text-sm text-text-muted">βγδ magnetic complexity</div>
            </div>
            <div>
              <div className="font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
                Last update
              </div>
              <div
                className="font-mono text-sm text-foreground mt-1 min-h-[1.25rem]"
                suppressHydrationWarning
              >
                {timeStr || "—"}
              </div>
              <div className="text-sm text-text-muted">GOES-18 primary feed</div>
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
        <PredictionCards />
      </section>

      <FluxChart data={liveFlux} />

      <SolarCycleWidget />
    </div>
  );
}

import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiQueries } from "@/lib/api";
import { FluxChart } from "@/components/FluxChart";
import { fluxFormatter, classColor } from "@/lib/mock-data";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "History — SolarWatch" },
      {
        name: "description",
        content:
          "Browse the past 7 days of solar X-ray flux and a catalog of detected C/M/X-class flare events.",
      },
      { property: "og:title", content: "History — SolarWatch" },
      {
        property: "og:description",
        content: "Past 7 days of solar X-ray flux and the SolarWatch flare catalog.",
      },
    ],
  }),
  component: History,
});

const windows = [
  { label: "24 hours", hours: 24 },
  { label: "3 days", hours: 72 },
  { label: "7 days", hours: 168 },
] as const;

function History() {
  const [hours, setHours] = useState<number>(72);
  const numPoints = hours * 60;

  const { data: lightCurveData, isLoading: lcLoading } = useQuery(
    apiQueries.lightCurve(numPoints)
  );
  const { data: predictions, isLoading: predLoading } = useQuery(
    apiQueries.recentPredictions(500)
  );

  const chartData = useMemo(() => {
    if (!lightCurveData) return [];
    return lightCurveData.map((pt) => ({
      time_tag: pt.timestamp,
      flux: pt.slx_counts || 0,
    }));
  }, [lightCurveData]);

  // Extract discrete flare events from continuous predictions
  const historicalEvents = useMemo(() => {
    if (!predictions) return [];
    const events: Array<{ id: string; class: any; peakFlux: number; peak: string }> = [];
    let lastPeakTime = 0;
    
    // Sort chronological first to group correctly
    const sorted = [...predictions].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    for (const p of sorted) {
      const isFlare = p.nowcast.is_flare_active || ["C", "M", "X"].includes(p.nowcast.flare_class);
      if (isFlare) {
        const t = new Date(p.timestamp).getTime();
        // If it's more than 30 minutes since the last event, start a new one
        if (t - lastPeakTime > 30 * 60 * 1000) {
          events.push({
            id: `evt-${p.id}`,
            class: p.nowcast.flare_class,
            peakFlux: p.raw_features.slx_counts || 0,
            peak: p.timestamp,
          });
          lastPeakTime = t;
        } else if (events.length > 0) {
          // Update current event peak if this point is stronger
          const lastEvt = events[events.length - 1];
          if ((p.raw_features.slx_counts || 0) > lastEvt.peakFlux) {
            lastEvt.peakFlux = p.raw_features.slx_counts || 0;
            lastEvt.class = p.nowcast.flare_class;
            lastEvt.peak = p.timestamp;
          }
        }
      }
    }
    return events.reverse(); // Newest first
  }, [predictions]);

  const currentWindow = windows.find((w) => w.hours === hours)!;

  if (lcLoading || predLoading) {
    return (
      <div className="flex h-[80vh] items-center justify-center font-mono text-sm text-text-muted">
        Loading historical archive...
      </div>
    );
  }

  const isCountRate = chartData.some((d) => d.flux > 1.0);

  return (
    <div className="mx-auto max-w-[1200px] px-4 sm:px-8 py-6 sm:py-10 flex flex-col gap-8">
      <header className="flex flex-col md:flex-row md:items-end justify-between gap-6 fade-in">
        <div>
          <span className="font-mono text-xs uppercase tracking-[0.22em] text-sky">
            Archive
          </span>
          <h1 className="font-serif text-4xl text-foreground leading-[1.05] mt-2">
            Historical <em className="text-text-muted">flux & flares</em>
          </h1>
          <p className="text-base text-text-muted mt-2 max-w-xl">
            Browse solar X-ray flux history and past flare events. Identify patterns before major solar activity.
          </p>
        </div>
        <div className="inline-flex rounded-md border border-border bg-surface p-1">
          {windows.map((w) => {
            const active = hours === w.hours;
            return (
              <button
                key={w.hours}
                onClick={() => setHours(w.hours)}
                className={`px-3 py-1.5 text-sm font-mono uppercase tracking-wider rounded-sm transition-colors ${
                  active
                    ? "bg-sky/15 text-sky"
                    : "text-text-muted hover:text-foreground"
                }`}
              >
                {w.label}
              </button>
            );
          })}
        </div>
      </header>

      <FluxChart
        data={chartData}
        height={360}
        title={isCountRate ? "SoLEXS Count Rate History" : `X-ray flux — past ${currentWindow.label}`}
        subtitle={isCountRate ? "SoLEXS · Soft X-ray · counts/s" : "GOES-18 · 1–8 Å · W/m²"}
      />

      <section className="fade-in">
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="font-serif text-xl text-foreground">Flare catalog</h2>
          <span className="font-mono text-xs uppercase tracking-wider text-text-faint">
            {historicalEvents.length} events detected
          </span>
        </div>
        {historicalEvents.length === 0 ? (
          <div className="card-surface p-8 text-center font-mono text-sm text-text-muted">
            No flare events detected in the current archive.
          </div>
        ) : (
          <div className="card-surface divide-y divide-border overflow-hidden">
            {historicalEvents.map((e) => {
              const c = classColor(e.class) || { chip: "text-text-muted" };
              return (
                <div
                  key={e.id}
                  className="flex items-center justify-between px-5 py-4 hover:bg-tertiary/40 transition-colors"
                >
                  <div className="flex items-center gap-5">
                    <span
                      className={`font-serif text-xl ${c.chip}`}
                      style={{ lineHeight: 1 }}
                    >
                      {e.class}
                    </span>
                    <div>
                      <div className="text-base text-foreground">
                        Peak {new Date(e.peak).toUTCString().replace("GMT", "UTC")}
                      </div>
                      <div className="font-mono text-xs text-text-faint mt-0.5">
                        {e.id.toUpperCase()}
                      </div>
                    </div>
                  </div>
                  <span className="font-mono text-base text-text-muted">
                    {isCountRate ? `${e.peakFlux.toLocaleString()} counts/s` : `${fluxFormatter(e.peakFlux)} W/m²`}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}


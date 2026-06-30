import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { FluxChart } from "@/components/FluxChart";
import { generateFlux, historicalEvents, fluxFormatter, classColor } from "@/lib/mock-data";

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
  const data = useMemo(() => generateFlux(hours, hours > 48 ? 10 : 5, hours + 3), [hours]);
  const currentWindow = windows.find((w) => w.hours === hours)!;

  return (
    <div className="mx-auto max-w-[1200px] px-8 py-10 flex flex-col gap-8">
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
        data={data}
        height={360}
        title={`X-ray flux — past ${currentWindow.label}`}
        subtitle="GOES-18 · 1–8 Å · W/m²"
      />

      <section className="fade-in">
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="font-serif text-xl text-foreground">Flare catalog</h2>
          <span className="font-mono text-xs uppercase tracking-wider text-text-faint">
            {historicalEvents.length} events
          </span>
        </div>
        <div className="card-surface divide-y divide-border overflow-hidden">
          {historicalEvents.map((e) => {
            const c = classColor(e.class);
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
                  {fluxFormatter(e.peakFlux)} W/m²
                </span>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}

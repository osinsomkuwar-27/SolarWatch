import { createFileRoute } from "@tanstack/react-router";
import { architectureLayers, overallAccuracy, perClassMetrics } from "@/lib/mock-data";

export const Route = createFileRoute("/insights")({
  head: () => ({
    meta: [
      { title: "Model Insights — SolarWatch" },
      {
        name: "description",
        content:
          "Inside the SolarWatch 1D CNN + LSTM model: per-class precision, recall, F1, and the full layer architecture.",
      },
      { property: "og:title", content: "Model Insights — SolarWatch" },
      {
        property: "og:description",
        content: "Per-class metrics and architecture of the SolarWatch CNN+LSTM forecaster.",
      },
    ],
  }),
  component: Insights,
});

function Insights() {
  return (
    <div className="mx-auto max-w-[960px] px-4 sm:px-8 py-6 sm:py-10 flex flex-col gap-8">
      <header className="fade-in">
        <span className="font-mono text-xs uppercase tracking-[0.22em] text-sky">
          Model
        </span>
        <h1 className="font-serif text-4xl text-foreground leading-[1.05] mt-2">
          Inside the <em className="text-text-muted">forecaster</em>
        </h1>
        <p className="text-base text-text-muted mt-2 max-w-xl">
          A 1D CNN reads short-term shape; a stacked LSTM learns temporal context. Together
          they nowcast flare class up to six hours ahead.
        </p>
      </header>

      <section className="card-surface p-8 fade-in flex flex-col md:flex-row md:items-center gap-8">
        <div className="flex-1">
          <div className="font-mono text-xs uppercase tracking-[0.22em] text-text-faint">
            Validation accuracy
          </div>
          <div
            className="font-serif mt-2"
            style={{ fontSize: 96, color: "var(--accent-sky)", letterSpacing: "-0.04em", lineHeight: 1 }}
          >
            {(overallAccuracy * 100).toFixed(1)}
            <span className="text-3xl text-text-muted">%</span>
          </div>
        </div>
        <p className="flex-1 text-base text-text-muted leading-relaxed">
          Trained on 14 years of NOAA GOES X-ray flux (2010–2024). Validated on a held-out
          12-month window covering the ascending phase of Solar Cycle 25.
        </p>
      </section>

      <section className="fade-in">
        <h2 className="font-serif text-xl text-foreground mb-3">Per-class metrics</h2>
        <div className="card-surface overflow-hidden">
          <table className="w-full text-base">
            <thead>
              <tr className="border-b border-border">
                {["Class", "Precision", "Recall", "F1"].map((h) => (
                  <th
                    key={h}
                    className="text-left font-mono text-xs uppercase tracking-[0.18em] text-text-faint px-5 py-3"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {perClassMetrics.map((row) => (
                <tr key={row.cls} className="border-b border-border last:border-0">
                  <td className="px-5 py-4 font-serif text-lg text-foreground">{row.cls}</td>
                  <td className="px-5 py-4 font-mono text-base text-text-muted">{row.precision.toFixed(3)}</td>
                  <td className="px-5 py-4 font-mono text-base text-text-muted">{row.recall.toFixed(3)}</td>
                  <td className="px-5 py-4 font-mono text-base text-sky">{row.f1.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="fade-in">
        <h2 className="font-serif text-xl text-foreground mb-3">Architecture</h2>
        <div className="card-surface p-6">
          <pre className="font-mono text-sm leading-[1.7] text-text-muted whitespace-pre-wrap">
{architectureLayers.join("\n")}
          </pre>
        </div>
      </section>
    </div>
  );
}

import { predictions } from "@/lib/mock-data";

function severity(p: number) {
  if (p > 0.6) return { color: "var(--alert-red)", label: "High", bar: "#991B1B", tint: "card-tint-rose" };
  if (p > 0.3) return { color: "var(--accent-amber)", label: "Elevated", bar: "#92400E", tint: "card-tint-amber" };
  return { color: "var(--safe-green)", label: "Low", bar: "#166534", tint: "card-tint-sage" };
}

export function PredictionCards() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {predictions.map((p) => {
        const pct = Math.round(p.probability * 100);
        const s = severity(p.probability);
        return (
          <div key={p.horizon} className={`card-surface card-hover ${s.tint} p-5 fade-in`}>
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
                {p.horizon} horizon
              </span>
              <span
                className="font-mono text-xs uppercase tracking-wider"
                style={{ color: s.color }}
              >
                {s.label}
              </span>
            </div>
            <div className="mt-3 flex items-baseline gap-2">
              <span
                className="font-serif"
                style={{ fontSize: 44, color: s.color, lineHeight: 1, letterSpacing: "-0.02em" }}
              >
                {pct}
              </span>
              <span className="text-text-muted text-lg font-serif">%</span>
            </div>
            <div className="text-sm text-text-muted mt-1">M-class or greater probability</div>
            <div className="mt-4 h-[6px] rounded-full bg-white border border-foreground overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{ width: `${pct}%`, backgroundColor: s.bar }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

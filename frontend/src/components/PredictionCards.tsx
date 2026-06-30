import { predictions as mockPredictions } from "@/lib/mock-data";

function severity(p: number) {
  if (p > 0.6) return { color: "#B91C1C", label: "high", bg: "#FEE2E2" };
  if (p > 0.3) return { color: "#1D4ED8", label: "elevated", bg: "#DBEAFE" };
  return { color: "#166534", label: "low", bg: "#DCFCE7" };
}

type Props = {
  nowcastProbability?: number;
  forecast30MinProbability?: number;
  forecast60MinProbability?: number;
};

export function PredictionCards({
  nowcastProbability,
  forecast30MinProbability,
  forecast60MinProbability,
}: Props) {
  const predictions =
    nowcastProbability !== undefined &&
    forecast30MinProbability !== undefined &&
    forecast60MinProbability !== undefined
      ? [
          { horizon: "Nowcast", probability: nowcastProbability },
          { horizon: "30 min", probability: forecast30MinProbability },
          { horizon: "60 min", probability: forecast60MinProbability },
        ]
      : mockPredictions;

  const maxBarHeight = 150;
  return (
    <div className="p-6 pb-3.5 fade-in rounded-none border-2 border-black">
      <div className="flex justify-between items-baseline mb-6">
        <h3 className="font-serif text-lg text-foreground">Forecast horizons</h3>
        <span className="font-mono text-xs uppercase tracking-wider text-text-faint">
          M-class or greater probability
        </span>
      </div>

      <div className="flex items-end justify-center gap-16 py-6">
        <div className="flex items-end gap-24 relative" style={{ height: 190, width: 460 }}>
          <div className="absolute left-0 right-0 bottom-0 h-[3px] bg-black z-[1]" />
          <span className="absolute left-[-34px] bottom-[-4px] font-mono text-xs font-bold uppercase tracking-wider text-black">
            0%
          </span>

          {predictions.map((p) => {
            const pct = Math.round(p.probability * 100);
            const s = severity(p.probability);
            const visual = pct === 42 ? { color: "#166534", label: s.label, bg: "#DCFCE7" } : s;
            const barHeight = p.probability * maxBarHeight;
            const showAdvisory = visual.label === "high";

            return (
              <div
                key={p.horizon}
                className="flex flex-col items-center flex-1 max-w-[120px] relative z-[2]"
              >
                <div
                  className="relative border-2 border-b-0"
                  style={{ width: 84, height: barHeight, background: visual.bg, borderColor: visual.color }}
                >
                  <span
                    className="absolute -top-10 left-1/2 -translate-x-1/2 font-serif text-4xl font-bold whitespace-nowrap"
                    style={{ color: visual.color }}
                  >
                    {pct}%
                  </span>

                  {showAdvisory && (
                    <div
                      className="absolute z-[1]"
                      style={{ left: "100%", marginLeft: 12, top: 8, width: 150 }}
                    >
                      <span
                        className="block font-mono text-xs uppercase tracking-wider whitespace-nowrap mb-2 font-bold"
                        style={{ color: visual.color }}
                      >
                        advisory · 60%
                      </span>
                      <div className="relative" style={{ height: 16 }}>
                        <div
                          className="absolute"
                          style={{
                            left: 11,
                            right: 0,
                            top: 7,
                            height: 3,
                            background:
                              "repeating-linear-gradient(to left, #B91C1C 0 10px, transparent 10px 10px)",
                            backgroundSize: "20px 3px",
                            animation: "marchAnt 0.7s linear infinite",
                          }}
                        />
                        <div
                          className="absolute"
                          style={{
                            left: 0,
                            top: 1,
                            width: 0,
                            height: 0,
                            borderTop: "7px solid transparent",
                            borderBottom: "7px solid transparent",
                            borderRight: "11px solid #B91C1C",
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
                <span className="absolute -bottom-[26px] font-mono text-xs font-bold uppercase tracking-wider text-black whitespace-nowrap">
                  {p.horizon} horizon
                </span>
              </div>
            );
          })}
        </div>

        <div className="flex flex-col gap-3 pb-[1px]">
          {predictions.map((p) => {
            const s = severity(p.probability);
            return (
              <div key={p.horizon} className="flex items-center gap-2 text-base">
                <span className="h-3 w-3 rounded-full shrink-0" style={{ background: s.color }} />
                <span className="text-text-muted">
                  {p.horizon} — <span className="font-bold text-foreground">{s.label}</span>
                </span>
              </div>
            );
          })}
        </div>
      </div>


      <div className="flex gap-7 mt-8 pt-3.5 border-t border-border">
        <div>
          <div className="font-mono text-xs uppercase tracking-wider text-text-faint">Model</div>
          <div className="text-sm text-foreground mt-0.5">1D CNN + LSTM</div>
        </div>
        <div>
          <div className="font-mono text-xs uppercase tracking-wider text-text-faint">Source</div>
          <div className="text-sm text-foreground mt-0.5">Aditya-L1 · ISSDC</div>
        </div>
      </div>

      <style>{`
        @keyframes marchAnt {
          from { background-position: 0 0; }
          to   { background-position: -20px 0; }
        }
      `}</style>
    </div>
  );
}
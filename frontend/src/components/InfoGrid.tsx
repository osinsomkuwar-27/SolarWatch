const items = [
  { label: "Primary feed", value: "NOAA GOES-18", meta: "operational", tint: "card-tint-sky" },
  { label: "Cadence", value: "60 second nowcast", meta: "<1 s latency", tint: "card-tint-amber" },
  { label: "Model", value: "1D CNN + LSTM", meta: "v0.4 · trained 2026-05", tint: "card-tint-sage" },
];

export function InfoGrid() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
      {items.map((i) => (
        <div key={i.label} className={`card-surface card-hover ${i.tint} p-5`}>
          <div className="font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
            {i.label}
          </div>
          <div className="font-serif text-xl text-foreground mt-2">{i.value}</div>
          <div className="text-sm text-text-muted mt-1">{i.meta}</div>
        </div>
      ))}
    </div>
  );
}

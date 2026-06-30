export function SolarCycleWidget({ percent = 47 }: { percent?: number }) {
  return (
    <div className="p-5 fade-in border-2 border-black rounded-2xl bg-white">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="font-serif text-base text-foreground">Solar Cycle</h3>
        <span className="font-mono text-xs uppercase tracking-wider text-text-faint">
          2019 → ~2030
        </span>
      </div>
      <div className="h-2.5 rounded-full bg-white border border-foreground overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            width: `${percent}%`,
            background: "linear-gradient(90deg, #16A34A, #DC2626)",
          }}
        />
      </div>
      <div className="flex items-center justify-between mt-2">
        <span className="font-mono text-sm text-text-muted">{percent}% complete</span>
        <span className="text-sm text-foreground">Near maximum activity</span>
      </div>
    </div>
  );
}
export function AlertPanel({ probability }: { probability: number }) {
  return (
    <div
      className="flex items-center gap-3 px-4 py-3 fade-in"
      style={{
        borderLeft: "3px solid #991B1B",
        background: "transparent",
      }}
      role="alert"
    >
      <div
        style={{
          width: "7px",
          height: "7px",
          borderRadius: "50%",
          background: "#991B1B",
          flexShrink: 0,
        }}
      />
      <span className="font-mono text-xs uppercase tracking-widest" style={{ color: "#991B1B" }}>
        Advisory
      </span>
      <span className="font-mono text-xs" style={{ color: "#888" }}>
        ·
      </span>
     <span className="font-mono text-sm" style={{ color: "#1a1a1a" }}>
        M-class flare imminent — satellite safe mode recommended
      </span>
      <span
        className="font-mono text-sm font-bold ml-auto"
        style={{ color: "#1a1a1a", whiteSpace: "nowrap" }}
      >
        {Math.round(probability * 100)}%
      </span>
    </div>
  );
}
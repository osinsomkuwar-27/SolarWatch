import type { FlareClass } from "@/lib/mock-data";

const palette: Record<FlareClass, { ring: string; text: string; dot: string }> = {
  A: { ring: "rgba(52, 211, 153, 0.45)", text: "var(--safe-green)", dot: "var(--safe-green)" },
  B: { ring: "rgba(56, 189, 248, 0.45)", text: "var(--accent-sky)", dot: "var(--accent-sky)" },
  C: { ring: "rgba(251, 191, 36, 0.55)", text: "var(--accent-amber)", dot: "var(--accent-amber)" },
  M: { ring: "rgba(251, 146, 60, 0.7)",  text: "#FB923C",            dot: "#FB923C" },
  X: { ring: "rgba(248, 113, 113, 0.85)", text: "var(--alert-red)",   dot: "var(--alert-red)" },
};

export function FlareClassBadge({ cls, size = 88 }: { cls: FlareClass; size?: number }) {
  const c = palette[cls];
  const intense = cls === "M" || cls === "X";
  return (
    <div
      className={`relative flex items-center justify-center rounded-full select-none ${intense ? "pulse-glow" : ""}`}
      style={{
        width: size,
        height: size,
        background:
          "radial-gradient(circle at 30% 30%, rgba(255,255,255,0.06), rgba(255,255,255,0) 60%), var(--bg-card)",
        boxShadow: `inset 0 0 0 1px ${c.ring}`,
      }}
      aria-label={`Current flare class ${cls}`}
    >
      <span
        className="font-serif"
        style={{ fontSize: size * 0.5, color: c.text, lineHeight: 1 }}
      >
        {cls}
      </span>
      <span
        className="absolute bottom-2 h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: c.dot, boxShadow: `0 0 8px ${c.dot}` }}
      />
    </div>
  );
}

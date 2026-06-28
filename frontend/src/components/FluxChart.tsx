import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import type { FluxPoint } from "@/lib/mock-data";
import { fluxFormatter } from "@/lib/mock-data";

type Props = {
  data: FluxPoint[];
  height?: number;
  title?: string;
  subtitle?: string;
};

export function FluxChart({
  data,
  height = 320,
  title = "Solar X-Ray Flux",
  subtitle = "GOES-18 · 1–8 Å · W/m²",
}: Props) {
  return (
    <div className="card-surface p-5 fade-in">
      <div className="flex items-end justify-between mb-4">
        <div>
          <h3 className="font-serif text-lg text-foreground">{title}</h3>
          <p className="font-mono text-xs uppercase tracking-[0.16em] text-text-faint mt-1">
            {subtitle}
          </p>
        </div>
        <div className="flex items-center gap-3 font-mono text-xs uppercase tracking-wider text-text-muted">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-solar" /> Flux
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-px w-3 bg-alert" /> M
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-px w-3 bg-sky" /> C
          </span>
        </div>
      </div>
      <div style={{ width: "100%", height }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
            <CartesianGrid stroke="rgba(11,25,41,0.06)" vertical={false} />
            <XAxis
              dataKey="time_tag"
              tickFormatter={(v) =>
                new Date(v).toLocaleTimeString("en-GB", {
                  hour: "2-digit",
                  minute: "2-digit",
                  timeZone: "UTC",
                })
              }
              tick={{ fill: "#4A4A4A", fontSize: 10, fontFamily: "JetBrains Mono" }}
              stroke="rgba(11,25,41,0.1)"
              minTickGap={50}
            />
            <YAxis
              scale="log"
              domain={[1e-9, 1e-3]}
              tickFormatter={(v) => fluxFormatter(v as number)}
              tick={{ fill: "#4A4A4A", fontSize: 10, fontFamily: "JetBrains Mono" }}
              stroke="rgba(11,25,41,0.1)"
              ticks={[1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}
              width={68}
            />
            <Tooltip
              contentStyle={{
                background: "#FFFFFF",
                border: "1px solid rgba(11,25,41,0.12)",
                borderRadius: 8,
                fontFamily: "JetBrains Mono",
                fontSize: 11,
                color: "#0B1929",
                boxShadow: "0 4px 16px rgba(11,25,41,0.08)",
              }}
              labelStyle={{ color: "#4A4A4A" }}
              labelFormatter={(v) => new Date(v).toUTCString()}
              formatter={(v) => [fluxFormatter(Number(v)), "W/m²"]}
            />
            <ReferenceLine y={1e-5} stroke="#B91C1C" strokeDasharray="3 4" strokeOpacity={0.7} />
            <ReferenceLine y={1e-6} stroke="#0369A1" strokeDasharray="3 4" strokeOpacity={0.5} />
            <Line
              type="monotone"
              dataKey="flux"
              stroke="#B45309"
              strokeWidth={1.75}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

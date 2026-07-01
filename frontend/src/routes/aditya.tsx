import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/aditya")({
  head: () => ({
    meta: [
      { title: "Aditya-L1 — SolarWatch" },
      {
        name: "description",
        content:
          "How SolarWatch integrates with ISRO's Aditya-L1 SoLEXS and HEL1OS X-ray instruments at the L1 Lagrange point.",
      },
      { property: "og:title", content: "Aditya-L1 Integration — SolarWatch" },
      {
        property: "og:description",
        content: "SolarWatch + ISRO Aditya-L1: combined soft and hard X-ray flare nowcasting.",
      },
    ],
  }),
  component: Aditya,
});

const heroStats = [
  { label: "Launch", value: "2 Sep 2023" },
  { label: "Orbit", value: "L1 Halo · 1.5M km" },
  { label: "Operator", value: "ISRO" },
  { label: "Mission life", value: "5+ years" },
];

const integrationPlan = [
  "Stream SoLEXS soft X-ray (1–22 keV) into the CNN input head as channel 1.",
  "Stream HEL1OS hard X-ray (10–150 keV) as channel 2 — captures impulsive flare onset.",
  "Re-train the LSTM head on combined Aditya-L1 + GOES tensor (256 × 4 channels).",
  "Cross-validate with the SUIT UV imager active-region masks for spatial context.",
  "Switch primary data source from NOAA GOES-18 to Aditya-L1 once L1 telemetry stabilises.",
];

function Aditya() {
  return (
    <div className="mx-auto max-w-[960px] px-4 sm:px-8 py-6 sm:py-10 flex flex-col gap-8">
      <header className="fade-in">
        <span className="font-mono text-xs uppercase tracking-[0.22em] text-sky">
          ISRO Mission
        </span>
        <h1 className="font-serif text-5xl text-foreground leading-[1] mt-2">
          Aditya<em className="text-text-muted">-L1</em>
        </h1>
        <p className="text-lg text-text-muted mt-4 max-w-2xl leading-relaxed">
          India's first dedicated solar observatory, stationed 1.5 million km sunward at the
          Sun–Earth L1 point — providing uninterrupted observation of the photosphere,
          chromosphere, and corona.
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mt-8 pt-6 border-t border-border">
          {heroStats.map((s) => (
            <div key={s.label}>
              <div className="font-mono text-xs uppercase tracking-[0.18em] text-text-faint">
                {s.label}
              </div>
              <div className="font-serif text-2xl text-foreground mt-1">{s.value}</div>
            </div>
          ))}
        </div>
      </header>

      <section className="card-surface p-6 fade-in">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="font-serif text-xl text-foreground">SoLEXS</h2>
          <span className="font-mono text-xs uppercase tracking-wider text-text-faint">
            Solar Low-Energy X-ray Spectrometer
          </span>
        </div>
        <p className="text-base text-text-muted leading-relaxed">
          Two SCD detectors covering 1–22 keV at ~180 eV resolution, sampling soft X-ray flux
          every second. Characterises the thermal phase of solar flares and feeds real-time
          spectroscopy to ground.
        </p>
        <div className="grid grid-cols-3 gap-3 mt-5">
          <Stat label="Energy" value="1–22 keV" />
          <Stat label="Cadence" value="1 s" />
          <Stat label="Resolution" value="~180 eV" />
        </div>
      </section>

      <section className="fade-in">
        <h2 className="font-serif text-xl text-foreground mb-4">Integration plan</h2>
        <ol className="card-surface divide-y divide-border overflow-hidden">
          {integrationPlan.map((item, i) => (
            <li key={i} className="flex gap-5 px-6 py-4">
              <span className="font-mono text-base text-sky w-6 shrink-0">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="text-base text-foreground leading-relaxed">{item}</span>
            </li>
          ))}
        </ol>
      </section>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-tertiary/40 px-3 py-2">
      <div className="font-mono text-xs uppercase tracking-wider text-text-faint">{label}</div>
      <div className="font-mono text-base text-sky mt-1">{value}</div>
    </div>
  );
}

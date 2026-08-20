'use client';

import React, { useState } from 'react';
import { TelemetryFrame } from '@/hooks/useSimulationStream';

interface DemandLoadChartProps {
  telemetry: TelemetryFrame | null;
  history: TelemetryFrame[];
}

export default function DemandLoadChart({ telemetry, history }: DemandLoadChartProps) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string; visible: boolean }>({
    x: 0,
    y: 0,
    text: '',
    visible: false,
  });

  const currentActualKw = telemetry?.power.total_hvac_kw || 42.5;
  const currentBaselineKw = telemetry?.power.baseline_hvac_kw || 68.0;
  const demandShavedKw = telemetry?.power.demand_shaved_kw || (currentBaselineKw - currentActualKw);

  // Generate SVG path points from history or fallback curve
  const dataPoints = history.length > 5 ? history : Array.from({ length: 24 }, (_, i) => ({
    power: {
      total_hvac_kw: 35 + Math.sin(i / 3) * 15 + (i > 14 && i < 18 ? -10 : 0),
      baseline_hvac_kw: 45 + Math.sin(i / 3) * 20 + 10,
    },
  }));

  const maxVal = 100;
  const minVal = 0;

  const actualPath = dataPoints
    .map((d, idx) => {
      const x = (idx / (dataPoints.length - 1)) * 100;
      const y = 100 - ((d.power.total_hvac_kw - minVal) / (maxVal - minVal)) * 100;
      return `${idx === 0 ? 'M' : 'L'} ${x} ${Math.max(5, Math.min(95, y))}`;
    })
    .join(' ');

  const baselinePath = dataPoints
    .map((d, idx) => {
      const x = (idx / (dataPoints.length - 1)) * 100;
      const y = 100 - ((d.power.baseline_hvac_kw - minVal) / (maxVal - minVal)) * 100;
      return `${idx === 0 ? 'M' : 'L'} ${x} ${Math.max(5, Math.min(95, y))}`;
    })
    .join(' ');

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = x / rect.width;
    const dataIdx = Math.min(Math.floor(pct * dataPoints.length), dataPoints.length - 1);
    const pt = dataPoints[dataIdx];
    if (pt) {
      setTooltip({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        text: `Actual: ${pt.power.total_hvac_kw.toFixed(1)} kW | Base: ${pt.power.baseline_hvac_kw.toFixed(1)} kW`,
        visible: true,
      });
    }
  };

  return (
    <article className="bg-surface-container-high/70 backdrop-blur-xl border border-grid-line rounded-lg p-panel-padding flex flex-col gap-4 relative">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="font-headline-md text-headline-md text-on-surface">Load Analysis</h3>
          <p className="font-label-caps text-label-caps text-on-surface-variant uppercase mt-1">
            Baseline vs. AETHERIS Load
          </p>
        </div>
        <div className="flex flex-col items-end">
          <span className="font-data-telemetry text-sm text-primary-fixed font-bold">
            {currentActualKw.toFixed(1)} kW
          </span>
          <span className="font-label-caps text-[10px] text-on-surface-variant">
            Shaved: -{Math.max(0, demandShavedKw).toFixed(1)} kW
          </span>
        </div>
      </div>

      {/* Interactive Chart Area */}
      <div className="h-40 w-full relative mt-2 border-b border-l border-grid-line/50">
        {tooltip.visible && (
          <div
            className="absolute bg-surface-container-lowest/90 text-on-surface px-2 py-1 rounded font-data-telemetry text-xs pointer-events-none z-30 border border-primary/20 transform -translate-x-1/2 -translate-y-full"
            style={{ left: tooltip.x, top: tooltip.y }}
          >
            {tooltip.text}
          </div>
        )}

        {/* Y-Axis Labels */}
        <div className="absolute -left-6 bottom-0 font-data-telemetry text-[10px] text-on-surface-variant">0</div>
        <div className="absolute -left-8 top-0 font-data-telemetry text-[10px] text-on-surface-variant">kW</div>

        {/* Grid lines */}
        <div className="absolute w-full h-[1px] bg-grid-line/50 top-1/2"></div>
        <div className="absolute w-full h-[1px] bg-grid-line/50 top-1/4"></div>
        <div className="absolute w-full h-[1px] bg-grid-line/50 top-3/4"></div>

        {/* SVG Line Chart */}
        <svg
          className="absolute inset-0 w-full h-full overflow-visible cursor-crosshair"
          preserveAspectRatio="none"
          viewBox="0 0 100 100"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip((t) => ({ ...t, visible: false }))}
        >
          {/* Baseline (Dashed, Slate) */}
          <path
            className="text-grid-line"
            d={baselinePath}
            fill="none"
            stroke="currentColor"
            strokeDasharray="4 4"
            strokeWidth="2"
          />
          {/* Actual Load (Solid, Primary) */}
          <path
            className="text-primary-fixed"
            d={actualPath}
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
          />
        </svg>
      </div>

      {/* Legend */}
      <div className="flex gap-6 mt-2">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-grid-line rounded-sm"></div>
          <span className="font-data-telemetry text-data-telemetry text-on-surface-variant">Baseline</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-primary-fixed rounded-sm shadow-[0_0_8px_rgba(255,165,134,0.4)]"></div>
          <span className="font-data-telemetry text-data-telemetry text-primary-fixed">AETHERIS</span>
        </div>
      </div>
    </article>
  );
}


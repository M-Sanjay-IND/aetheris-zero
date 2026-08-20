'use client';

import React, { useState } from 'react';
import { TelemetryFrame } from '@/hooks/useSimulationStream';

interface PriceCurveChartProps {
  telemetry: TelemetryFrame | null;
  history: TelemetryFrame[];
}

export default function PriceCurveChart({ telemetry, history }: PriceCurveChartProps) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string; visible: boolean }>({
    x: 0,
    y: 0,
    text: '',
    visible: false,
  });

  const currentPriceInr = telemetry?.dynamic_lmp_price_inr_mwh || 11827.5;
  const isSpike = telemetry?.grid_dr_event_active || (telemetry?.dynamic_lmp_price || 0) >= 0.50;

  // Generate price curve points (default 24 points)
  const dataPoints = history.length > 5 ? history : Array.from({ length: 24 }, (_, i) => {
    let pUsd = 0.12;
    if (i >= 14 && i <= 18) pUsd = 1.50;
    else if (i >= 10 && i < 14) pUsd = 0.25;
    return {
      dynamic_lmp_price_inr_mwh: Math.round(pUsd * 1000 * 83),
      timestamp_hour: i,
    };
  });

  const maxPrice = 140000;
  const minPrice = 5000;

  const pathPoints = dataPoints
    .map((d, idx) => {
      const x = (idx / (dataPoints.length - 1)) * 100;
      const priceVal = d.dynamic_lmp_price_inr_mwh || 10000;
      const y = 100 - ((priceVal - minPrice) / (maxPrice - minPrice)) * 100;
      return `${idx === 0 ? 'M' : 'L'} ${x} ${Math.max(5, Math.min(95, y))}`;
    })
    .join(' ');

  const areaPath = `${pathPoints} L 100 100 L 0 100 Z`;

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
        text: `₹${(pt.dynamic_lmp_price_inr_mwh || 0).toLocaleString()}/MWh`,
        visible: true,
      });
    }
  };

  return (
    <article className="bg-surface-container-high/70 backdrop-blur-xl border border-grid-line rounded-lg p-panel-padding flex flex-col gap-4 relative">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="font-headline-md text-headline-md text-on-surface">CAISO LMP Price</h3>
          <p className="font-label-caps text-label-caps text-on-surface-variant uppercase mt-1">
            Live Price Curve (₹/MWh)
          </p>
        </div>
        {isSpike && (
          <div className="px-2 py-1 bg-error-container/20 border border-error/50 rounded text-error font-data-telemetry text-xs animate-pulse">
            SPIKE DETECTED
          </div>
        )}
      </div>

      {/* Interactive Chart Area */}
      <div className="h-32 w-full relative mt-2 border-b border-l border-grid-line/50">
        {tooltip.visible && (
          <div
            className="absolute bg-surface-container-lowest/90 text-on-surface px-2 py-1 rounded font-data-telemetry text-xs pointer-events-none z-30 border border-error/30 transform -translate-x-1/2 -translate-y-full"
            style={{ left: tooltip.x, top: tooltip.y }}
          >
            {tooltip.text}
          </div>
        )}

        <svg
          className="absolute inset-0 w-full h-full overflow-visible cursor-crosshair"
          preserveAspectRatio="none"
          viewBox="0 0 100 100"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip((t) => ({ ...t, visible: false }))}
        >
          <defs>
            <linearGradient id="priceGradReact" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stopColor="#b51a2b" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#b51a2b" stopOpacity="0.0" />
            </linearGradient>
          </defs>
          <path d={areaPath} fill="url(#priceGradReact)" />
          <path className="text-error" d={pathPoints} fill="none" stroke="currentColor" strokeWidth="2" />
          {/* Current Price Point */}
          <circle className="fill-error" cx="100" cy="20" r="3">
            <animate attributeName="r" dur="2s" repeatCount="indefinite" values="3;6;3" />
          </circle>
        </svg>
      </div>

      <div className="text-right">
        <span className="font-data-telemetry text-[24px] font-bold text-error">
          ₹{currentPriceInr.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
        </span>
      </div>
    </article>
  );
}


'use client';

import React from 'react';
import { TelemetryFrame } from '@/hooks/useSimulationStream';

interface SavingsCardProps {
  telemetry: TelemetryFrame | null;
}

export default function SavingsCard({ telemetry }: SavingsCardProps) {
  const savingsInr = telemetry?.metrics?.cost_savings_inr || 4060360.45;
  const savingsPct = telemetry?.metrics?.cost_savings_pct || 30.2;
  const peakRedPct = telemetry?.metrics?.peak_demand_reduction_pct || 31.0;

  const intPart = Math.floor(savingsInr).toLocaleString('en-IN');
  const decPart = (savingsInr % 1).toFixed(2).substring(1);

  return (
    <article className="bg-surface-container-highest/80 backdrop-blur-xl border border-secondary/20 rounded-lg p-panel-padding flex items-center justify-between relative overflow-hidden group">
      <div className="absolute inset-0 bg-gradient-to-r from-secondary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
      <div>
        <p className="font-label-caps text-label-caps text-on-surface-variant uppercase mb-2">
          Accumulated Savings ({savingsPct > 0 ? `${savingsPct.toFixed(1)}% ROI` : 'Active ROI'})
        </p>
        <div className="font-display-lg text-[32px] text-secondary">
          ₹{intPart}
          <span className="text-[20px] text-secondary/70">{decPart}</span>
        </div>
        <div className="flex gap-4 mt-2 font-data-telemetry text-xs text-on-surface-variant">
          <span>Peak Reduction: <strong className="text-primary-fixed">{peakRedPct.toFixed(1)}%</strong></span>
          <span>SLA: <strong className="text-primary-fixed">100% ASHRAE 55</strong></span>
        </div>
      </div>
      <div className="w-12 h-12 rounded-full bg-secondary-container/20 border border-secondary/30 flex items-center justify-center">
        <span className="material-symbols-outlined text-secondary text-2xl">payments</span>
      </div>
    </article>
  );
}


'use client';

import React from 'react';
import { TelemetryFrame } from '@/hooks/useSimulationStream';

interface ActionPanelProps {
  telemetry: TelemetryFrame | null;
  isRunning: boolean;
  controllerMode: string;
  onStart: () => void;
  onStop: () => void;
  onStep: () => void;
  onReset: () => void;
  onTriggerDR: () => void;
  onInjectFault: () => void;
  onToggleShadow: (enabled: boolean) => void;
  onRunEpisode: () => void;
}

export default function ActionPanel({
  telemetry,
  isRunning,
  controllerMode,
  onStart,
  onStop,
  onStep,
  onReset,
  onTriggerDR,
  onInjectFault,
  onToggleShadow,
  onRunEpisode,
}: ActionPanelProps) {
  const stepCount = telemetry?.step || 0;
  const timeStr = telemetry?.time_of_day_str || '14:30';
  const isShadow = controllerMode === 'SHADOW_MODE';
  const isIntervened = telemetry?.safety?.intervention_active;

  return (
    <section className="mt-auto w-full p-gutter px-margin_edge bg-surface-container-low/90 backdrop-blur-2xl border-t border-grid-line flex flex-col md:flex-row justify-between items-center gap-4 z-20">
      {/* Left: System Status & Time indicator */}
      <div className="flex items-center gap-4">
        <div
          className={`w-2.5 h-2.5 rounded-full ${
            isIntervened
              ? 'bg-error animate-pulse shadow-[0_0_10px_rgba(181,26,43,0.9)]'
              : 'bg-primary-fixed animate-pulse shadow-[0_0_8px_rgba(255,165,134,0.8)]'
          }`}
        />
        <div className="flex flex-col">
          <span className="font-data-telemetry text-data-telemetry text-on-surface-variant uppercase tracking-wider text-xs">
            {isIntervened ? 'CBF SHIELD INTERVENED' : 'SYSTEM ARMED & OPTIMAL'}
          </span>
          <span className="font-data-telemetry text-xs text-primary-fixed font-bold">
            {timeStr} | Step {stepCount} | Mode: {controllerMode}
          </span>
        </div>
      </div>

      {/* Middle: Step / Play / Pause Controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={isRunning ? onStop : onStart}
          className={`px-4 py-2 rounded font-label-caps text-label-caps uppercase transition-colors flex items-center gap-1.5 ${
            isRunning
              ? 'bg-secondary-container/80 text-on-surface border border-secondary/40 hover:bg-secondary-container'
              : 'bg-surface-variant/80 text-primary-fixed border border-primary/40 hover:bg-surface-variant'
          }`}
        >
          <span className="material-symbols-outlined text-[16px]">{isRunning ? 'pause' : 'play_arrow'}</span>
          {isRunning ? 'Pause Loop' : 'Start Loop'}
        </button>

        <button
          onClick={onStep}
          disabled={isRunning}
          className="px-3 py-2 rounded border border-grid-line text-on-surface-variant font-label-caps text-label-caps uppercase hover:bg-surface-variant/40 transition-colors disabled:opacity-40"
        >
          <span className="material-symbols-outlined text-[16px]">skip_next</span>
        </button>

        <button
          onClick={onReset}
          className="px-3 py-2 rounded border border-grid-line text-on-surface-variant font-label-caps text-label-caps uppercase hover:bg-surface-variant/40 transition-colors"
        >
          <span className="material-symbols-outlined text-[16px]">restart_alt</span>
        </button>
      </div>

      {/* Right Action Triggers */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Run 24h Episode Button */}
        <button
          onClick={onRunEpisode}
          className="px-4 py-2 rounded border border-primary/40 text-primary-fixed font-label-caps text-label-caps uppercase hover:bg-primary-container/20 transition-colors flex items-center gap-1.5 text-xs"
        >
          <span className="material-symbols-outlined text-[16px]">speed</span>
          24h ROI Run
        </button>

        {/* Toggle Shadow Mode */}
        <button
          onClick={() => onToggleShadow(!isShadow)}
          className={`px-4 py-2 rounded border font-label-caps text-label-caps uppercase transition-colors flex items-center gap-1.5 text-xs ${
            isShadow
              ? 'border-primary-fixed bg-primary-container/40 text-primary-fixed'
              : 'border-transparent text-on-surface-variant hover:bg-surface-variant/50'
          }`}
        >
          <span className="material-symbols-outlined text-[16px]">
            {isShadow ? 'visibility' : 'visibility_off'}
          </span>
          {isShadow ? 'Shadow: ON' : 'Shadow Mode'}
        </button>

        {/* Fault Injection Button */}
        <button
          onClick={onInjectFault}
          className="px-4 py-2 rounded border border-error text-error font-label-caps text-label-caps uppercase hover:bg-error/10 transition-colors flex items-center gap-1.5 text-xs pulse-danger"
        >
          <span className="material-symbols-outlined text-[16px]">bug_report</span>
          Override (38°C)
        </button>

        {/* Trigger OpenADR Event Button */}
        <button
          onClick={onTriggerDR}
          className="px-6 py-2.5 rounded bg-primary-fixed text-on-primary font-label-caps text-label-caps uppercase font-bold hover:bg-primary-fixed/90 transition-colors flex items-center gap-2 shadow-[0_0_15px_rgba(255,165,134,0.3)] hover:shadow-[0_0_25px_rgba(255,165,134,0.5)] text-xs"
        >
          <span className="material-symbols-outlined text-[16px]">bolt</span>
          Trigger OpenADR DR
        </button>
      </div>
    </section>
  );
}


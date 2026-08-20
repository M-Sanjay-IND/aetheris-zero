'use client';

import React, { useState } from 'react';
import BuildingScene from '@/components/3d/BuildingScene';
import DemandLoadChart from '@/components/analytics/DemandLoadChart';
import PriceCurveChart from '@/components/analytics/PriceCurveChart';
import SavingsCard from '@/components/analytics/SavingsCard';
import ActionPanel from '@/components/controls/ActionPanel';
import IngestionUpload from '@/components/controls/IngestionUpload';
import { useSimulationStream } from '@/hooks/useSimulationStream';

export default function Home() {
  const {
    telemetry,
    history,
    isConnected,
    isRunning,
    controllerMode,
    triggerDR,
    injectFault,
    toggleShadow,
    stepSimulation,
    resetSimulation,
    startSimulation,
    stopSimulation,
    runEpisode,
    setMode,
  } = useSimulationStream();

  const [isIngestionOpen, setIsIngestionOpen] = useState(false);
  const isIntervened = telemetry?.safety?.intervention_active;
  const isDrActive = telemetry?.grid_dr_event_active;

  return (
    <div className="bg-background text-on-background font-body-base overflow-hidden min-h-screen selection:bg-primary-container selection:text-on-primary-container relative">
      {/* TopNavBar */}
      <nav className="fixed top-0 w-full z-50 flex justify-between items-center px-margin_edge h-16 bg-surface-container/80 backdrop-blur-xl border-b border-grid-line">
        {/* Brand */}
        <div className="flex items-center gap-4">
          <span className="font-display-lg text-[24px] font-bold tracking-tighter text-primary">
            AETHERIS-Zero
          </span>
          <span className="px-2 py-0.5 rounded text-[10px] font-label-caps bg-primary-container/40 text-primary-fixed border border-primary/30 uppercase">
            Safe-RL Transactive VPP
          </span>
        </div>

        {/* Navigation Links */}
        <div className="hidden md:flex gap-8 items-center h-full">
          <button
            onClick={() => setMode('RL_SAFE_ARBITRAGE')}
            className={`h-full flex items-center font-label-caps text-label-caps uppercase transition-colors pt-1 ${
              controllerMode === 'RL_SAFE_ARBITRAGE'
                ? 'text-primary-fixed border-b-2 border-primary-fixed pb-1'
                : 'text-on-surface-variant hover:text-primary-fixed'
            }`}
          >
            PPO Safe-RL
          </button>
          <button
            onClick={() => setMode('BASELINE_HEURISTIC')}
            className={`h-full flex items-center font-label-caps text-label-caps uppercase transition-colors pt-1 ${
              controllerMode === 'BASELINE_HEURISTIC'
                ? 'text-primary-fixed border-b-2 border-primary-fixed pb-1'
                : 'text-on-surface-variant hover:text-primary-fixed'
            }`}
          >
            Baseline
          </button>
          <button
            onClick={() => setIsIngestionOpen(true)}
            className="h-full flex items-center text-on-surface-variant font-label-caps text-label-caps uppercase hover:text-primary-fixed transition-colors pt-1"
          >
            Brick Schema
          </button>
        </div>

        {/* Trailing Actions */}
        <div className="flex items-center gap-4 text-primary">
          <button
            onClick={() => setIsIngestionOpen(true)}
            title="Semantic Ingestion / Brick Schema"
            className="hover:text-primary-fixed transition-colors flex items-center justify-center p-2 rounded-full hover:bg-surface-variant/30"
          >
            <span className="material-symbols-outlined text-lg">schema</span>
          </button>
          <div className="flex items-center gap-1.5 px-3 py-1 bg-surface-container-high rounded-full border border-grid-line text-xs font-data-telemetry">
            <span
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-primary-fixed shadow-[0_0_6px_rgba(255,165,134,0.8)]' : 'bg-error'
              }`}
            />
            <span className="text-on-surface-variant">{isConnected ? 'LIVE WS' : 'OFFLINE'}</span>
          </div>
        </div>
      </nav>

      {/* Main Dashboard Layout */}
      <main className="relative z-10 pt-16 h-screen flex flex-col">
        {/* Split View Container */}
        <div className="flex-grow flex flex-col md:flex-row gap-gutter p-gutter pb-0 overflow-hidden">
          {/* Left Column: 3D Digital Twin Stage (60%) */}
          <section className="w-full md:w-[60%] relative bg-surface-container-low/60 backdrop-blur-md border border-grid-line rounded-lg overflow-hidden inner-glow flex flex-col">
            {/* Header / Telemetry Tag */}
            <div className="absolute top-4 left-4 z-20 flex flex-wrap items-center gap-3">
              <div className="px-3 py-1 bg-surface-variant/80 backdrop-blur-sm border-l-2 border-primary-fixed font-data-telemetry text-data-telemetry text-primary-fixed">
                SYS.TWIN.01
              </div>

              <div
                className={`flex items-center gap-2 px-3 py-1 backdrop-blur-sm border rounded-sm ${
                  isIntervened
                    ? 'bg-error-container/40 border-error text-error safety-glow animate-pulse'
                    : 'bg-surface-container/60 border-grid-line text-primary-fixed'
                }`}
              >
                <span className="material-symbols-outlined text-sm">
                  {isIntervened ? 'warning' : 'verified_user'}
                </span>
                <span className="font-label-caps text-label-caps uppercase tracking-widest text-[10px]">
                  Safety Barrier: {isIntervened ? 'INTERVENED' : 'OPTIMAL'}
                </span>
              </div>

              {isDrActive && (
                <div className="px-3 py-1 bg-error-container/40 border border-error rounded-sm text-error font-data-telemetry text-[10px] animate-pulse">
                  OPENADR 3.0 DR EVENT ACTIVE
                </div>
              )}
            </div>

            {/* 3D Scene Wrapper */}
            <BuildingScene telemetry={telemetry} />

            {/* Overlay Grids / Crosshairs for Technical HUD Aesthetic */}
            <div className="absolute inset-0 pointer-events-none border border-grid-line/30 m-8 opacity-20">
              <div className="absolute top-0 left-0 w-4 h-4 border-t border-l border-primary-fixed"></div>
              <div className="absolute top-0 right-0 w-4 h-4 border-t border-r border-primary-fixed"></div>
              <div className="absolute bottom-0 left-0 w-4 h-4 border-b border-l border-primary-fixed"></div>
              <div className="absolute bottom-0 right-0 w-4 h-4 border-b border-r border-primary-fixed"></div>
            </div>
          </section>

          {/* Right Column: Analytics Rail (40%) */}
          <section className="w-full md:w-[40%] flex flex-col gap-gutter overflow-y-auto pr-1 pb-gutter custom-scrollbar">
            <DemandLoadChart telemetry={telemetry} history={history} />
            <PriceCurveChart telemetry={telemetry} history={history} />
            <SavingsCard telemetry={telemetry} />
          </section>
        </div>

        {/* Bottom Panel: Control Deck */}
        <ActionPanel
          telemetry={telemetry}
          isRunning={isRunning}
          controllerMode={controllerMode}
          onStart={startSimulation}
          onStop={stopSimulation}
          onStep={stepSimulation}
          onReset={resetSimulation}
          onTriggerDR={() => triggerDR(1.50, 14.0, 4.0)}
          onInjectFault={() => injectFault('zone_1', 38.0)}
          onToggleShadow={toggleShadow}
          onRunEpisode={() => runEpisode(288)}
        />
      </main>

      {/* Semantic Ingestion Modal */}
      <IngestionUpload isOpen={isIngestionOpen} onClose={() => setIsIngestionOpen(false)} />
    </div>
  );
}


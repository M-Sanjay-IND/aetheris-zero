import { useState, useEffect, useRef, useCallback } from 'react';

export interface ZoneState {
  temp_c: number;
  setpoint_c: number;
  pmv: number;
  ppd: number;
  comfort_compliant: boolean;
  occupancy: number;
  cooling_load_kw: number;
  hex_color?: string;
  heat_intensity?: number;
}

export interface PowerState {
  chiller_kw: number;
  supply_fan_kw: number;
  base_load_kw: number;
  total_hvac_kw: number;
  baseline_hvac_kw: number;
  demand_shaved_kw: number;
}

export interface SafetyState {
  cbf_qp_active: boolean;
  intervention_active: boolean;
  shield_status: string;
  dwell_time_remaining_sec: number;
  max_slew_rate_c_per_step: number;
  min_comfort_limit_c: number;
  max_comfort_limit_c: number;
}

export interface MetricsState {
  cumulative_energy_actual_kwh: number;
  cumulative_energy_baseline_kwh: number;
  cumulative_cost_actual: number;
  cumulative_cost_baseline: number;
  cost_savings_usd: number;
  cost_savings_inr: number;
  cost_savings_pct: number;
  peak_demand_reduction_pct: number;
  ashrae55_comfort_violation_count: number;
}

export interface TelemetryFrame {
  step: number;
  timestamp_hour: number;
  time_of_day_str: string;
  ambient_temp_c: number;
  solar_irradiance_wm2: number;
  dynamic_lmp_price: number;
  dynamic_lmp_price_inr_mwh: number;
  grid_dr_event_active: boolean;
  active_dr_event_id?: string | null;
  controller_mode?: string;
  zones: Record<string, ZoneState>;
  power: PowerState;
  safety: SafetyState;
  metrics: MetricsState;
}

export function useSimulationStream(wsUrl?: string) {
  const [telemetry, setTelemetry] = useState<TelemetryFrame | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [controllerMode, setControllerMode] = useState<string>('RL_SAFE_ARBITRAGE');
  const [history, setHistory] = useState<TelemetryFrame[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<any>(null);

  const url = wsUrl || (typeof window !== 'undefined' 
    ? `ws://${window.location.hostname}:8000/ws/telemetry` 
    : 'ws://localhost:8000/ws/telemetry');

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'INITIAL_STATE' || data.type === 'TELEMETRY_UPDATE') {
            const frame: TelemetryFrame = data.telemetry;
            // Enrich with INR conversion ($1 = ₹83)
            if (!frame.dynamic_lmp_price_inr_mwh) {
              frame.dynamic_lmp_price_inr_mwh = Math.round(frame.dynamic_lmp_price * 1000 * 83);
            }
            if (!frame.metrics.cost_savings_inr) {
              frame.metrics.cost_savings_inr = Math.round(frame.metrics.cost_savings_usd * 83 * 100) / 100;
            }
            setTelemetry(frame);
            setHistory((prev) => {
              const updated = [...prev, frame];
              return updated.length > 50 ? updated.slice(updated.length - 50) : updated;
            });
          } else if (data.type === 'LOOP_STARTED') {
            setIsRunning(true);
          } else if (data.type === 'LOOP_STOPPED') {
            setIsRunning(false);
          } else if (data.type === 'MODE_CHANGED' || data.type === 'SHADOW_MODE_TOGGLED') {
            if (data.controller_mode) setControllerMode(data.controller_mode);
          }
        } catch (e) {
          console.error('Error parsing telemetry message', e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        reconnectTimeoutRef.current = setTimeout(connect, 2000);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch (e) {
      console.error('WebSocket connection failed', e);
    }
  }, [url]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const sendAction = useCallback((action: string, params: Record<string, any> = {}) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action, params, timestamp: Date.now() }));
    }
  }, []);

  const triggerDR = useCallback((priceSpike = 1.50, startHour = 14.0, durationHours = 4.0) => {
    sendAction('TRIGGER_OPENADR_EVENT', { price_spike: priceSpike, start_hour: startHour, duration_hours: durationHours });
  }, [sendAction]);

  const injectFault = useCallback((zoneId = 'zone_1', targetTemp = 38.0) => {
    sendAction('INJECT_MALICIOUS_SETPOINT', { zone_id: zoneId, target_temp: targetTemp });
  }, [sendAction]);

  const toggleShadow = useCallback((enabled: boolean) => {
    sendAction('TOGGLE_SHADOW_MODE', { enabled });
  }, [sendAction]);

  const stepSimulation = useCallback(() => {
    sendAction('STEP_SIMULATION');
  }, [sendAction]);

  const resetSimulation = useCallback(() => {
    sendAction('RESET_SIMULATION');
    setHistory([]);
  }, [sendAction]);

  const startSimulation = useCallback(() => {
    sendAction('START_SIMULATION');
    setIsRunning(true);
  }, [sendAction]);

  const stopSimulation = useCallback(() => {
    sendAction('STOP_SIMULATION');
    setIsRunning(false);
  }, [sendAction]);

  const runEpisode = useCallback((totalSteps = 288) => {
    sendAction('RUN_EPISODE', { total_steps: totalSteps });
  }, [sendAction]);

  const setMode = useCallback((mode: string) => {
    sendAction('SET_CONTROLLER_MODE', { mode });
    setControllerMode(mode);
  }, [sendAction]);

  return {
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
  };
}


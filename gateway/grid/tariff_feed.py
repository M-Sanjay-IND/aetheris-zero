"""
TariffFeed: Dynamic Wholesale Electricity LMP Tariff Generator & Feed for CAISO / ERCOT.
Supplies real-time prices, 24h forward forecast vectors for RL pre-cooling arbitrage,
and interactive price spike injection hooks for live pitch demonstrations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class TariffFeed:
    """
    Wholesale Locational Marginal Pricing (LMP) feed simulator.
    Provides sub-hourly price interpolation, lookahead vectors for safe-RL arbitrage,
    and runtime price spike event triggers.
    """

    DEFAULT_CAISO_HOURLY = [
        0.085, 0.080, 0.075, 0.072, 0.075, 0.090,  # 00:00 - 05:00 Off-peak
        0.145, 0.195, 0.220,                       # 06:00 - 08:00 Morning peak
        0.120, 0.065, 0.045, 0.040, 0.050,         # 09:00 - 13:00 Solar duck curve / pre-cooling window
        1.500, 1.500, 1.500, 1.500,                # 14:00 - 17:00 Critical DR event spike ($1.50/kWh)
        0.420, 0.380, 0.290,                       # 18:00 - 20:00 Evening peak
        0.180, 0.125, 0.095                        # 21:00 - 23:00 Night shoulder
    ]

    def __init__(
        self,
        json_file_path: Optional[Union[str, Path]] = None,
        base_hourly_prices: Optional[List[float]] = None,
        market_name: str = "CAISO_TH_SP15_GEN",
    ):
        self.market_name = market_name
        self.nominal_hourly_prices: List[float] = []
        self.active_hourly_prices: List[float] = []
        self.spikes: List[Dict[str, float]] = []

        if json_file_path and Path(json_file_path).exists():
            self.load_from_json(json_file_path)
        elif base_hourly_prices:
            self.nominal_hourly_prices = list(base_hourly_prices)
            self.active_hourly_prices = list(base_hourly_prices)
        else:
            self.nominal_hourly_prices = list(self.DEFAULT_CAISO_HOURLY)
            self.active_hourly_prices = list(self.DEFAULT_CAISO_HOURLY)

    def load_from_json(self, json_file_path: Union[str, Path]) -> None:
        """Load 24-hour pricing curve from JSON file."""
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prices = [0.0] * 24
        for item in data:
            hr = int(item.get("hour", 0))
            pr = float(item.get("price", 0.10))
            if 0 <= hr < 24:
                prices[hr] = pr
        self.nominal_hourly_prices = prices
        self.active_hourly_prices = list(prices)

    def reset(self) -> None:
        """Reset active prices and remove injected spikes."""
        self.active_hourly_prices = list(self.nominal_hourly_prices)
        self.spikes.clear()

    def inject_spike(self, start_hour: float, duration_hours: float = 4.0, spike_price: float = 1.50) -> None:
        """
        Dynamically inject a critical pricing spike event.
        Used for testing OpenADR load shedding and thermal arbitrage.
        """
        self.spikes.append({
            "start_hour": start_hour,
            "end_hour": start_hour + duration_hours,
            "spike_price": spike_price,
        })
        start_idx = int(start_hour) % 24
        end_idx = int(start_hour + duration_hours) % 24
        if start_idx <= end_idx:
            for h in range(start_idx, end_idx):
                self.active_hourly_prices[h] = max(self.active_hourly_prices[h], spike_price)
        else:
            for h in range(start_idx, 24):
                self.active_hourly_prices[h] = max(self.active_hourly_prices[h], spike_price)
            for h in range(0, end_idx):
                self.active_hourly_prices[h] = max(self.active_hourly_prices[h], spike_price)

    def get_price_by_hour(self, hour: float) -> float:
        """Get electricity price ($/kWh) at continuous hour of the day [0.0 - 24.0)."""
        norm_hour = hour % 24.0
        h_floor = int(norm_hour)
        h_ceil = (h_floor + 1) % 24
        alpha = norm_hour - h_floor

        p_floor = self.active_hourly_prices[h_floor]
        p_ceil = self.active_hourly_prices[h_ceil]

        # Check explicit spike intervals
        for sp in self.spikes:
            if sp["start_hour"] <= norm_hour <= sp["end_hour"]:
                return float(sp["spike_price"])

        # Linear interpolation between hourly knots
        price = (1.0 - alpha) * p_floor + alpha * p_ceil
        return float(round(price, 4))

    def get_price(self, timestamp_sec: float | int) -> float:
        """Get electricity price ($/kWh) at timestamp in seconds."""
        hour = (float(timestamp_sec) / 3600.0) % 24.0
        return self.get_price_by_hour(hour)

    def get_forecast_horizon(
        self,
        current_step: int,
        horizon_steps: int = 96,
        step_sec: int = 300,
    ) -> List[float]:
        """
        Generate forward lookahead price vector for RL and PINN forward simulation.
        By default returns 96 steps @ 5-min intervals (24 hours lookahead).
        """
        forecast = []
        for i in range(horizon_steps):
            future_sec = (current_step + i) * step_sec
            forecast.append(self.get_price(future_sec))
        return forecast

    def to_chart_data(self, resolution_minutes: int = 15) -> List[Dict[str, Any]]:
        """Export full 24h price curve formatted for frontend chart components."""
        steps = int(24 * 60 / resolution_minutes)
        data = []
        for s in range(steps):
            t_min = s * resolution_minutes
            hr = t_min / 60.0
            price = self.get_price_by_hour(hr)
            regime = "OFF_PEAK"
            if price >= 1.00:
                regime = "CRITICAL_DR_EVENT"
            elif price >= 0.30:
                regime = "PEAK"
            elif hr in [10, 11, 12, 13] or price <= 0.06:
                regime = "SOLAR_SURPLUS"

            data.append({
                "time_str": f"{int(hr):02d}:{int((hr % 1) * 60):02d}",
                "hour": round(hr, 2),
                "price_usd_per_kwh": price,
                "price_usd_per_mwh": round(price * 1000.0, 2),
                "regime": regime,
            })
        return data

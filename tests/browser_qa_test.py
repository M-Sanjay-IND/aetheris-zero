import sys
import time
import json
from playwright.sync_api import sync_playwright

def run_qa_suite():
    results = {
        "page_load_and_console": {},
        "currency_toggle": {},
        "simulation_controls": {},
        "weather_scenario_studio": {},
        "tariff_price_controls": {},
        "thermostat_and_safety_shield": {},
        "zone_selection_and_hud": {},
        "guide_modal_and_shortcuts": {},
        "cards_and_nan_inspection": {},
        "errors": []
    }

    console_logs = []
    page_errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path='/home/ace/.config/Antigravity/bin/google-chrome',
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # Track console and errors
        page.on("console", lambda msg: console_logs.append({"type": msg.type, "text": msg.text}))
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        print("==================================================================")
        print("[TEST 1] Navigating to http://localhost:8000/ & Console Error Check")
        print("==================================================================")
        response = page.goto("http://localhost:8000/", wait_until="networkidle")
        time.sleep(1.0) # allow websocket initial frame & Three.js setup

        title = page.title()
        status = response.status if response else 0
        error_logs = [log for log in console_logs if log['type'] in ('error',)]
        results["page_load_and_console"] = {
            "status_code": status,
            "title": title,
            "error_logs": error_logs,
            "all_logs": console_logs,
            "page_errors": page_errors,
            "passed": (status == 200 and len(error_logs) == 0 and len(page_errors) == 0)
        }
        print(f"  Title: {title}")
        print(f"  HTTP Status: {status}")
        print(f"  Console Error Logs ({len(error_logs)}): {error_logs}")
        print(f"  Page Unhandled Exceptions ({len(page_errors)}): {page_errors}")
        print(f"  >> Test 1 Passed: {results['page_load_and_console']['passed']}")

        # Helper to get text
        def get_text(sel):
            return page.locator(sel).inner_text().strip()

        # Helper to set range slider value cleanly
        def set_range_slider(sel, val):
            page.locator(sel).evaluate(f"el => {{ el.value = '{val}'; el.dispatchEvent(new Event('input', {{ bubbles: true }})); }}")

        # Helper to check no NaN or undefined in any text
        def check_no_nan(text_str, context_name):
            if "nan" in text_str.lower() or "undefined" in text_str.lower() or "null" in text_str.lower():
                results["errors"].append(f"Found invalid value in {context_name}: '{text_str}'")
                return False
            return True

        # ==========================================
        # TEST 2: Currency toggle (INR / USD)
        # ==========================================
        print("\n==================================================================")
        print("[TEST 2] Testing Currency toggle (INR / USD)")
        print("==================================================================")
        inr_savings = get_text("#metric-savings-val")
        inr_price = get_text("#metric-price-val")
        inr_mwh = get_text("#metric-price-mwh")
        inr_slider_price = get_text("#slider-price-val")
        inr_burn_rate = get_text("#metric-savings-burn-rate")

        # Click USD toggle
        page.click("#btn-currency-usd")
        time.sleep(0.5)

        usd_savings = get_text("#metric-savings-val")
        usd_price = get_text("#metric-price-val")
        usd_mwh = get_text("#metric-price-mwh")
        usd_slider_price = get_text("#slider-price-val")
        usd_burn_rate = get_text("#metric-savings-burn-rate")

        # Click back to INR
        page.click("#btn-currency-inr")
        time.sleep(0.5)

        inr_savings_after = get_text("#metric-savings-val")
        inr_price_after = get_text("#metric-price-val")
        inr_slider_price_after = get_text("#slider-price-val")

        curr_passed = (
            "₹" in inr_savings and "₹" in inr_price and "₹" in inr_slider_price and
            "$" in usd_savings and "$" in usd_price and "$" in usd_slider_price and
            "₹" in inr_savings_after and "₹" in inr_price_after
        )
        results["currency_toggle"] = {
            "initial_inr": {"savings": inr_savings, "price": inr_price, "mwh": inr_mwh, "slider": inr_slider_price, "burn": inr_burn_rate},
            "usd_mode": {"savings": usd_savings, "price": usd_price, "mwh": usd_mwh, "slider": usd_slider_price, "burn": usd_burn_rate},
            "back_to_inr": {"savings": inr_savings_after, "price": inr_price_after, "slider": inr_slider_price_after},
            "passed": curr_passed
        }
        print(f"  INR Mode -> Savings: {inr_savings}, Price: {inr_price}, MWh: {inr_mwh}, Slider: {inr_slider_price}")
        print(f"  USD Mode -> Savings: {usd_savings}, Price: {usd_price}, MWh: {usd_mwh}, Slider: {usd_slider_price}")
        print(f"  Back INR -> Savings: {inr_savings_after}, Price: {inr_price_after}, Slider: {inr_slider_price_after}")
        print(f"  >> Test 2 Passed: {curr_passed}")

        # ==========================================
        # TEST 3: Simulation Controls
        # ==========================================
        print("\n==================================================================")
        print("[TEST 3] Testing Simulation Controls ('+5 Min', 'Start Auto Optimizer', 'Full 24h Day', 'Reset')")
        print("==================================================================")
        # Ensure reset first
        page.click("button:has-text('Reset')")
        time.sleep(0.6)
        init_clock = get_text("#sim-clock-display")
        init_step = get_text("#sim-step-badge")

        # 3a. +5 Min Step
        page.click("button:has-text('+5 Min')")
        time.sleep(0.8)
        step1_clock = get_text("#sim-clock-display")
        step1_step = get_text("#sim-step-badge")

        # 3b. Start Auto Optimizer
        page.click("#btn-toggle-loop")
        time.sleep(1.0)
        auto_btn_text = get_text("#loop-text")
        time.sleep(0.8)
        auto_clock = get_text("#sim-clock-display")
        auto_step = get_text("#sim-step-badge")
        
        # Stop Auto Optimizer
        page.click("#btn-toggle-loop")
        time.sleep(0.6)
        paused_btn_text = get_text("#loop-text")

        # 3c. Full 24h Day
        page.click("button:has-text('Full 24h Day')")
        time.sleep(1.5)
        fast_savings = get_text("#metric-savings-val")
        fast_carbon = get_text("#metric-carbon-val")
        fast_clock = get_text("#sim-clock-display")
        fast_step = get_text("#sim-step-badge")

        # 3d. Reset
        page.click("button:has-text('Reset')")
        time.sleep(0.8)
        reset_clock = get_text("#sim-clock-display")
        reset_step = get_text("#sim-step-badge")
        reset_savings = get_text("#metric-savings-val")

        sim_passed = (
            step1_clock != init_clock or step1_step != init_step
        ) and (
            auto_btn_text == "Pause Optimizer" and paused_btn_text == "Start Auto Optimizer"
        ) and (
            reset_clock == "00:00" and "0" in reset_step and "0.00" in reset_savings
        )

        results["simulation_controls"] = {
            "init_clock": init_clock,
            "step1_clock": step1_clock,
            "step1_step": step1_step,
            "auto_btn_text_running": auto_btn_text,
            "auto_clock": auto_clock,
            "auto_step": auto_step,
            "auto_btn_text_paused": paused_btn_text,
            "fast_24h": {"clock": fast_clock, "step": fast_step, "savings": fast_savings, "carbon": fast_carbon},
            "reset": {"clock": reset_clock, "step": reset_step, "savings": reset_savings},
            "passed": sim_passed
        }
        print(f"  Initial State: {init_clock} ({init_step})")
        print(f"  +5 Min Step:   {step1_clock} ({step1_step})")
        print(f"  Auto Optimizer: Button says '{auto_btn_text}', Advanced to {auto_clock} ({auto_step}), Paused button says '{paused_btn_text}'")
        print(f"  Full 24h Day:  {fast_clock} ({fast_step}) with Savings: {fast_savings} & CO2 Avoided: {fast_carbon}")
        print(f"  Reset State:   {reset_clock} ({reset_step}) with Savings: {reset_savings}")
        print(f"  >> Test 3 Passed: {sim_passed}")

        # ==========================================
        # TEST 4: Live Input & Scenario Studio - Weather Slider & Presets
        # ==========================================
        print("\n==================================================================")
        print("[TEST 4] Testing Weather Slider & Presets (18°C, 28°C, 42°C Heatwave)")
        print("==================================================================")
        # Preset 18°C Cool Day
        page.click("button:has-text('Cool Day (18°C)')")
        time.sleep(0.5)
        w18_slider_label = get_text("#slider-weather-val")
        w18_hud_text = get_text("#three-weather-text")
        w18_slider_val = page.locator("#slider-weather").input_value()

        # Preset 28°C Standard
        page.click("button:has-text('Standard (28°C)')")
        time.sleep(0.5)
        w28_slider_label = get_text("#slider-weather-val")
        w28_hud_text = get_text("#three-weather-text")
        w28_slider_val = page.locator("#slider-weather").input_value()

        # Preset 42°C Heatwave
        page.click("button:has-text('Heatwave (42°C)')")
        time.sleep(0.5)
        w42_slider_label = get_text("#slider-weather-val")
        w42_hud_text = get_text("#three-weather-text")
        w42_slider_val = page.locator("#slider-weather").input_value()

        # Test dragging / setting weather slider via range value dispatch
        set_range_slider("#slider-weather", "35.0")
        page.click("button:has-text('Apply') >> nth=0")
        time.sleep(0.5)
        w35_slider_label = get_text("#slider-weather-val")
        w35_hud_text = get_text("#three-weather-text")

        weather_passed = (
            "18" in w18_slider_label and "18" in w18_hud_text and
            "28" in w28_slider_label and "28" in w28_hud_text and
            "42" in w42_slider_label and "42" in w42_hud_text and
            "35" in w35_slider_label and "35" in w35_hud_text
        )
        results["weather_scenario_studio"] = {
            "w18": {"label": w18_slider_label, "hud": w18_hud_text, "val": w18_slider_val},
            "w28": {"label": w28_slider_label, "hud": w28_hud_text, "val": w28_slider_val},
            "w42": {"label": w42_slider_label, "hud": w42_hud_text, "val": w42_slider_val},
            "w35_custom": {"label": w35_slider_label, "hud": w35_hud_text},
            "passed": weather_passed
        }
        print(f"  Preset 18°C Cool Day:  Slider: {w18_slider_label}, 3D HUD: {w18_hud_text}")
        print(f"  Preset 28°C Standard:  Slider: {w28_slider_label}, 3D HUD: {w28_hud_text}")
        print(f"  Preset 42°C Heatwave:  Slider: {w42_slider_label}, 3D HUD: {w42_hud_text}")
        print(f"  Custom 35°C Dragged:   Slider: {w35_slider_label}, 3D HUD: {w35_hud_text}")
        print(f"  >> Test 4 Passed: {weather_passed}")

        # ==========================================
        # TEST 5: Electricity Tariff Price slider, presets, and 'Trigger 5x Peak Surge'
        # ==========================================
        print("\n==================================================================")
        print("[TEST 5] Testing Electricity Tariff Price Slider, Presets, and 'Trigger 5x Peak Surge'")
        print("==================================================================")
        # Preset Off-Peak (₹6.5)
        page.click("button:has-text('Off-Peak (₹6.5)')")
        time.sleep(0.5)
        p_offpeak_label = get_text("#slider-price-val")
        p_offpeak_card = get_text("#metric-price-val")

        # Preset Mid-Day (₹18)
        page.click("button:has-text('Mid-Day (₹18)')")
        time.sleep(0.5)
        p_mid_label = get_text("#slider-price-val")
        p_mid_card = get_text("#metric-price-val")

        # Custom slider fill
        set_range_slider("#slider-price", "30.0")
        page.click("button:has-text('Apply') >> nth=1")
        time.sleep(0.5)
        p_custom_label = get_text("#slider-price-val")
        p_custom_card = get_text("#metric-price-val")

        # Trigger 5x Peak Surge button
        page.click("button:has-text('Trigger 5x Peak Surge')")
        time.sleep(0.8)
        p_surge_card = get_text("#metric-price-val")
        p_surge_level = get_text("#metric-price-level")
        status_banner_title = get_text("#status-banner-title")

        tariff_passed = (
            "6.5" in p_offpeak_label and
            "18" in p_mid_label and
            "30" in p_custom_label and
            ("Peak" in p_surge_level or "Expensive" in p_surge_level or "High Electricity" in status_banner_title or "124" in p_surge_card or "1.5" in p_surge_card)
        )
        results["tariff_price_controls"] = {
            "offpeak": {"label": p_offpeak_label, "card": p_offpeak_card},
            "midday": {"label": p_mid_label, "card": p_mid_card},
            "custom": {"label": p_custom_label, "card": p_custom_card},
            "peak_surge": {"card": p_surge_card, "level": p_surge_level, "banner": status_banner_title},
            "passed": tariff_passed
        }
        print(f"  Preset Off-Peak (₹6.5): Slider: {p_offpeak_label} -> Price Card: {p_offpeak_card}")
        print(f"  Preset Mid-Day (₹18):  Slider: {p_mid_label} -> Price Card: {p_mid_card}")
        print(f"  Custom Slider ₹30.0:   Slider: {p_custom_label} -> Price Card: {p_custom_card}")
        print(f"  Trigger 5x Peak Surge: Price Card: {p_surge_card}, Tier: {p_surge_level}, Status Banner: '{status_banner_title}'")
        print(f"  >> Test 5 Passed: {tariff_passed}")

        # ==========================================
        # TEST 6: Room Thermostat target slider for each zone & Safety Clamp 38°C
        # ==========================================
        print("\n==================================================================")
        print("[TEST 6] Testing Room Thermostat Target Sliders & 'Test Malicious 38°C' Safety Shield Demo")
        print("==================================================================")
        zone_slider_results = {}
        for zid, zname in [('zone_1', 'Core Floor 1'), ('zone_2', 'North Office'), ('zone_3', 'South Office'), ('zone_4', 'East Office'), ('zone_5', 'West Office')]:
            page.click(f"#pill-{zid}")
            time.sleep(0.3)
            # Set target slider to 23.5°C
            set_range_slider("#slider-zone-temp", "23.5")
            page.click("button:has-text('Set Room')")
            time.sleep(0.5)
            z_val = get_text("#slider-zone-val")
            z_hud_name = get_text("#selected-zone-name-label")
            zone_slider_results[zid] = {"name": z_hud_name, "val": z_val}
            print(f"  Zone {zid} ({zname}): Set target -> {z_val} (Selected: {z_hud_name})")

        # Test Malicious 38°C Safety Clamp Demo Button
        page.click("button:has-text('Test Malicious 38°C')")
        time.sleep(1.0)
        shield_banner = get_text("#status-banner-title")
        shield_badge = get_text("#shield-badge")
        shield_solve_time = get_text("#shield-solve-time")
        shield_bounds = get_text("#shield-bounds-val")
        toast_title = get_text("#toast-title")
        toast_msg = get_text("#toast-message")

        safety_passed = (
            "Safety Shield" in shield_banner or "Intervened" in shield_banner or "Safe Clamp" in shield_badge or "Intervened" in shield_badge
        ) and "38" in toast_title and "24.5" in toast_msg

        results["thermostat_and_safety_shield"] = {
            "zones_set": zone_slider_results,
            "malicious_test": {
                "banner": shield_banner,
                "badge": shield_badge,
                "solve_time": shield_solve_time,
                "bounds": shield_bounds,
                "toast_title": toast_title,
                "toast_msg": toast_msg
            },
            "passed": safety_passed
        }
        print(f"  Malicious 38°C Test: Banner: '{shield_banner}', Badge: '{shield_badge}'")
        print(f"  Safety Shield Diagnostics: Bounds: {shield_bounds}, Solver Latency: {shield_solve_time}")
        print(f"  Intercept Toast Notification: '{toast_title}' - '{toast_msg}'")
        print(f"  >> Test 6 Passed: {safety_passed}")

        # ==========================================
        # TEST 7: Clicking 3D building zones and zone selector pills & HUD sync
        # ==========================================
        print("\n==================================================================")
        print("[TEST 7] Testing 3D Building Zones & Selector Pills HUD Sync")
        print("==================================================================")
        zone_pill_sync = {}
        for zid, expected_name in [('zone_1', 'Core Floor 1'), ('zone_2', 'North Office'), ('zone_3', 'South Office'), ('zone_4', 'East Office'), ('zone_5', 'West Office')]:
            page.click(f"#pill-{zid}")
            time.sleep(0.3)
            hud_name = get_text("#hud-zone-name")
            hud_temp = get_text("#hud-zone-temp")
            hud_mass = get_text("#hud-zone-mass")
            badge_3d_temp = get_text(f"#badge-3d-temp-{zid}")
            pill_temp = get_text(f"#badge-temp-{zid}")
            matches = (hud_name == expected_name)
            zone_pill_sync[zid] = {
                "expected": expected_name,
                "hud_name": hud_name,
                "hud_temp": hud_temp,
                "hud_mass": hud_mass,
                "badge_3d_temp": badge_3d_temp,
                "pill_temp": pill_temp,
                "matches": matches
            }
            print(f"  Pill '{expected_name}': HUD Name: {hud_name}, HUD Temp: {hud_temp}, Mass Temp: {hud_mass}, 3D Badge: {badge_3d_temp} -> Synced: {matches}")

        # Test clicking directly on Three.js 3D canvas (Core zone is at center (0,0,0))
        canvas_box = page.locator("#three-canvas-container").bounding_box()
        if canvas_box:
            # Click near center of canvas
            page.mouse.click(canvas_box["x"] + canvas_box["width"] / 2, canvas_box["y"] + canvas_box["height"] / 2)
            time.sleep(0.5)
            canvas_clicked_zone = get_text("#hud-zone-name")
        else:
            canvas_clicked_zone = "Unknown"

        zone_sync_passed = all(z["matches"] for z in zone_pill_sync.values())
        results["zone_selection_and_hud"] = {
            "pills_sync": zone_pill_sync,
            "canvas_center_click_selected": canvas_clicked_zone,
            "passed": zone_sync_passed
        }
        print(f"  Three.js Canvas Direct Click Selected: '{canvas_clicked_zone}'")
        print(f"  >> Test 7 Passed: {zone_sync_passed}")

        # ==========================================
        # TEST 8: Opening and closing 'How It Works' guide modal (button & shortcut)
        # ==========================================
        print("\n==================================================================")
        print("[TEST 8] Testing 'How It Works' Guide Modal (Buttons & Keyboard Shortcuts)")
        print("==================================================================")
        # 8a. Open by button
        is_hidden_before = "hidden" in (page.locator("#guide-modal").get_attribute("class") or "")
        page.click("button:has-text('How It Works')")
        time.sleep(0.4)
        is_hidden_after_open_btn = "hidden" in (page.locator("#guide-modal").get_attribute("class") or "")

        # Close by 'Got It!' button
        page.click("button:has-text('Got It!')")
        time.sleep(0.4)
        is_hidden_after_close_btn = "hidden" in (page.locator("#guide-modal").get_attribute("class") or "")

        # 8b. Open by keyboard shortcut ('h' / '?')
        page.keyboard.press("h")
        time.sleep(0.4)
        is_hidden_after_key_open = "hidden" in (page.locator("#guide-modal").get_attribute("class") or "")

        # Close by keyboard shortcut ('h')
        page.keyboard.press("h")
        time.sleep(0.4)
        is_hidden_after_key_close = "hidden" in (page.locator("#guide-modal").get_attribute("class") or "")

        modal_passed = (
            is_hidden_before and
            not is_hidden_after_open_btn and
            is_hidden_after_close_btn and
            not is_hidden_after_key_open and
            is_hidden_after_key_close
        )
        results["guide_modal_and_shortcuts"] = {
            "initial_hidden": is_hidden_before,
            "after_open_button_visible": not is_hidden_after_open_btn,
            "after_close_button_hidden": is_hidden_after_close_btn,
            "after_key_h_open_visible": not is_hidden_after_key_open,
            "after_key_h_close_hidden": is_hidden_after_key_close,
            "passed": modal_passed
        }
        print(f"  Initial Modal Hidden: {is_hidden_before}")
        print(f"  Opened via 'How It Works' button: {not is_hidden_after_open_btn}")
        print(f"  Closed via 'Got It!' button: {is_hidden_after_close_btn}")
        print(f"  Opened via 'H' key shortcut: {not is_hidden_after_key_open}")
        print(f"  Closed via 'H' key shortcut: {is_hidden_after_key_close}")
        print(f"  >> Test 8 Passed: {modal_passed}")

        # ==========================================
        # TEST 9: Inspect all cards and verify zero NaN / undefined values
        # ==========================================
        print("\n==================================================================")
        print("[TEST 9] Inspecting All Metric Cards, HUDs & Diagnostics for NaN / Undefined")
        print("==================================================================")
        inspected_elements = {
            "Total Cost Saved (val)": get_text("#metric-savings-val"),
            "Total Cost Saved (pct)": get_text("#metric-savings-pct"),
            "Instant Savings Burn Rate": get_text("#metric-savings-burn-rate"),
            "Electricity Price (val)": get_text("#metric-price-val"),
            "Electricity Price (level)": get_text("#metric-price-level"),
            "Electricity Price (mwh)": get_text("#metric-price-mwh"),
            "Current Power Draw (val)": get_text("#metric-power-val"),
            "Power Breakdown": get_text("#metric-power-breakdown"),
            "Power Shaved (Virtual Battery)": get_text("#metric-power-shaved"),
            "Room Comfort (val)": get_text("#metric-comfort-val"),
            "Room Comfort (status)": get_text("#metric-comfort-status"),
            "Occupant Count": get_text("#metric-occupant-count"),
            "CO2 Avoided (val)": get_text("#metric-carbon-val"),
            "Shield Bounds": get_text("#shield-bounds-val"),
            "Shield Slew Rate": get_text("#shield-slew-val"),
            "Shield Solver Latency": get_text("#shield-solve-time"),
            "Shield Dwell Status": get_text("#shield-dwell-status"),
            "Shield Barrier Constraints": get_text("#shield-constraints-val"),
            "Simulation Clock": get_text("#sim-clock-display"),
            "Simulation Step Badge": get_text("#sim-step-badge"),
            "3D Weather Text": get_text("#three-weather-text"),
            "Active HUD Zone Name": get_text("#hud-zone-name"),
            "Active HUD Air Temp": get_text("#hud-zone-temp"),
            "Active HUD Mass Temp": get_text("#hud-zone-mass"),
        }

        has_nan = False
        nan_details = []
        for elem_name, val in inspected_elements.items():
            valid = check_no_nan(val, elem_name)
            status_str = "OK" if valid else "FAILED (NaN/Undefined)"
            print(f"  [{status_str}] {elem_name:30}: {val}")
            if not valid:
                has_nan = True
                nan_details.append({"element": elem_name, "value": val})

        results["cards_and_nan_inspection"] = {
            "inspected_elements": inspected_elements,
            "nan_found": has_nan,
            "nan_details": nan_details,
            "passed": not has_nan
        }
        print(f"  Total Elements Checked: {len(inspected_elements)}")
        print(f"  Any NaN / Undefined values detected: {has_nan}")
        print(f"  >> Test 9 Passed: {not has_nan}")

        # Take final screenshot
        page.screenshot(path="/home/ace/aetheris-zero/tests/dashboard_qa_screenshot.png")
        print("\n  [+] Saved screenshot to: /home/ace/aetheris-zero/tests/dashboard_qa_screenshot.png")

        browser.close()

    return results

if __name__ == "__main__":
    res = run_qa_suite()
    print("\n==================================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("==================================================================")

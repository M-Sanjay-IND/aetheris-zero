---
name: Violet Dusk Precision
colors:
  surface: '#25052b'
  surface-dim: '#25052b'
  surface-bright: '#4f2c54'
  surface-container-lowest: '#1f0126'
  surface-container-low: '#2e0d34'
  surface-container: '#331238'
  surface-container-high: '#3e1d44'
  surface-container-highest: '#4a284f'
  on-surface: '#ffd6ff'
  on-surface-variant: '#d1c4ba'
  inverse-surface: '#ffd6ff'
  inverse-on-surface: '#45234b'
  outline: '#9a8f85'
  outline-variant: '#4e453d'
  surface-tint: '#dcc2a8'
  primary: '#fffbff'
  on-primary: '#3d2d1b'
  primary-container: '#f6dbc0'
  on-primary-container: '#735f4a'
  inverse-primary: '#6e5b46'
  secondary: '#ffafd5'
  on-secondary: '#541b3c'
  secondary-container: '#713455'
  on-secondary-container: '#f0a1c7'
  tertiary: '#fffaff'
  on-tertiary: '#313129'
  tertiary-container: '#e3dfd4'
  on-tertiary-container: '#64625a'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#f9dec3'
  primary-fixed-dim: '#dcc2a8'
  on-primary-fixed: '#261908'
  on-primary-fixed-variant: '#554430'
  secondary-fixed: '#ffd8e8'
  secondary-fixed-dim: '#ffafd5'
  on-secondary-fixed: '#3a0526'
  on-secondary-fixed-variant: '#6f3253'
  tertiary-fixed: '#e6e2d8'
  tertiary-fixed-dim: '#cac6bc'
  on-tertiary-fixed: '#1c1c15'
  on-tertiary-fixed-variant: '#48473f'
  background: '#25052b'
  on-background: '#ffd6ff'
  surface-variant: '#4a284f'
  dusk-violet: '#502D55'
  muted-rose: '#935073'
  soft-peach: '#F6DBC0'
  off-white: '#F8F4E9'
  surface-overlay: rgba(80, 45, 85, 0.7)
  grid-line: rgba(248, 244, 233, 0.15)
typography:
  display-lg:
    fontFamily: Hanken Grotesk
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Hanken Grotesk
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Hanken Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-base:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  data-telemetry:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0.02em
  label-caps:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.1em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 16px
  margin-edge: 24px
  panel-padding: 20px
  grid-column-gap: 16px
---

## Brand & Style
The design system evolves the **Cyber-Physical** aesthetic into a more sophisticated, atmospheric monitoring environment. It maintains the technical precision required for heavy industrial and energy management but softens the "Electric Cyan" neon-futurism in favor of a "Twilight Industrial" mood.

The visual language is defined by **Atmospheric Technicality**:
- **Sophisticated Monitoring:** Utilizing a deep violet foundation to reduce eye strain in 24/7 control room environments while maintaining high-contrast telemetry in soft peach and off-white.
- **Glassmorphic Depth:** Translucent layers and background blurs create a tiered hierarchy that feels like a physical heads-up display (HUD) overlaid on digital twins or data streams.
- **Precision Engineering:** Despite the warmer color palette, the system retains its industrial rigor through the use of monospaced fonts, grid-aligned elements, and sharp, machined edges.
- **Localized Context:** All financial data and energy cost metrics are denominated in Indian Rupee (₹), ensuring regional relevance for infrastructure monitoring.

## Colors
The "Violet Dusk" palette is optimized for ergonomic dark-mode usage, providing a high-contrast yet soothing interface for technical monitoring.

- **Primary (Soft Peach):** The core interactive color. Used for active states, data highlights, and primary call-to-actions. It provides a warm, readable contrast against the violet background.
- **Secondary (Muted Rose):** Used for system status updates, secondary navigation, and categorized telemetry tags.
- **Tertiary (Off-White):** Reserved for high-priority labels, textual content, and essential iconography to ensure maximum legibility.
- **Neutral (Deep Violet):** The foundation of the system. This color serves as the global background and the base for all surface containers.
- **Surface Strategy:** Surfaces utilize the `dusk-violet` with varying opacities and backdrop blurs. The 3D Digital Twin or primary visualization layer sits behind these semi-transparent panels.

## Typography
The typographic system creates a hierarchy between "Executive Summary" (Sans-Serif) and "Engineering Detail" (Monospace).

- **Hanken Grotesk** conveys professional authority for headlines and high-level system status.
- **Inter** handles all narrative text, ensuring that explainable AI summaries and decision-making logic remain highly legible.
- **JetBrains Mono** is mandatory for all live sensor feeds, BACnet tags, and Rupee (₹) currency values. The monospaced nature ensures that fluctuating energy costs do not cause layout shifts.
- **Localization:** Ensure the Rupee (₹) symbol is rendered using the monospaced font to align with numeric values in data tables.

## Layout & Spacing
The layout follows a **Fluid Dashboard** model, utilizing a 12-column grid system optimized for information density.

- **The Grid:** A 4px base unit governs all dimensions. The primary layout typically features a 60/40 split on desktop: a central "Stage" for 3D visualization and a right-aligned "Analytics Rail" for telemetry.
- **Responsive Behavior:** 
  - **Desktop (>1440px):** Permanent sidebars and multi-pane telemetry views.
  - **Tablet (768px - 1439px):** Collapsible rails; the 3D twin expands to fill the viewport width.
  - **Mobile (<767px):** Single column stack. The 3D view is fixed to the top 40% of the viewport, with scrollable analytical cards below.
- **Z-Axis Layers:** Base (3D environment), HUD (floating labels/vectors), and Control (fixed glassmorphic panels).

## Elevation & Depth
In this design system, depth is achieved through **Optical Transparency** rather than traditional drop shadows.

- **Glassmorphism:** All control panels use `surface-overlay` with a `16px` to `24px` backdrop blur. This allows the user to maintain visual context of the 3D facility model behind the interface.
- **Internal Illumination:** Interactive components use a subtle inner glow (1px, Soft Peach, 15% opacity) to signify they are "powered on."
- **Technical Outlines:** Panels are defined by 1px solid borders in `grid-line` color. This creates a crisp, industrial separation between elements without adding visual weight.
- **Alert Tiers:** High-priority alerts do not use shadows but instead use a pulsing outer glow in `muted-rose` to draw attention to the specific sensor or panel.

## Shapes
The shape language is **Industrial and Machined**.

Avoid large, soft curves. The system uses a strict `0.25rem` (4px) radius for standard containers to evoke the feeling of hardware components. 

For high-importance system triggers (e.g., "Manual Override" or "Load Shedding"), components should use a **45-degree chamfered corner** (cut corner) instead of a radius. This visual cue reinforces the "Safety-Critical" nature of those specific actions.

## Components
- **Buttons:** Primary buttons use a solid `soft-peach` background with `dusk-violet` text. Secondary buttons use a `muted-rose` outline with `off-white` text. All buttons feature the `label-caps` font style.
- **Telemetry Chips:** Small labels used for sensor IDs (e.g., `TRN-09`). They feature a semi-transparent `dusk-violet` background and a 2px left-accent border in `soft-peach`.
- **Data Cards:** Glassmorphic containers with a `grid-line` border. Header text uses `label-caps`. Financial metrics must always include the `₹` symbol prefix in `data-telemetry` style.
- **Input Fields:** Recessed, dark violet fields with a 1px `soft-peach` border on focus. All numerical inputs use `jetbrainsMono`.
- **Currency Displays:** Live energy cost counters should be rendered in `soft-peach` using `data-telemetry`, ensuring the Rupee symbol and decimals are perfectly aligned.
- **Status Indicators:** Use `muted-rose` for warnings/maintenance and `off-white` for standard operational data.
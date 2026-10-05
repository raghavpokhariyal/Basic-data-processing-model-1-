/**
 * Shared SVG clip paths in objectBoundingBox units, so any element can be
 * masked into an arch with `clip-path: url(#arch-mughal)`.
 */
export function ArchDefs() {
  return (
    <svg width="0" height="0" aria-hidden="true" focusable="false" style={{ position: 'absolute' }}>
      <defs>
        {/* Pointed Mughal arch with a gentle ogee shoulder */}
        <clipPath id="arch-mughal" clipPathUnits="objectBoundingBox">
          <path d="M0 1V0.36C0 0.2 0.2 0.1 0.38 0.055C0.45 0.037 0.5 0 0.5 0C0.5 0 0.55 0.037 0.62 0.055C0.8 0.1 1 0.2 1 0.36V1Z" />
        </clipPath>
        {/* Rajasthani cusped (multifoil) arch */}
        <clipPath id="arch-cusped" clipPathUnits="objectBoundingBox">
          <path d="M0 1V0.42C0 0.38 0.02 0.35 0.05 0.34C0.03 0.29 0.05 0.24 0.1 0.22C0.1 0.16 0.15 0.12 0.21 0.12C0.24 0.07 0.3 0.05 0.36 0.06C0.4 0.02 0.46 0 0.5 0C0.54 0 0.6 0.02 0.64 0.06C0.7 0.05 0.76 0.07 0.79 0.12C0.85 0.12 0.9 0.16 0.9 0.22C0.95 0.24 0.97 0.29 0.95 0.34C0.98 0.35 1 0.38 1 0.42V1Z" />
        </clipPath>
      </defs>
    </svg>
  );
}

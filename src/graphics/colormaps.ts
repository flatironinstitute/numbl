export type RGB = [number, number, number];

/** An axes colormap: a named map, or custom N×3 `data` (which wins). */
export interface ColormapSpec {
  name?: string;
  data?: number[][];
}

// Anchor samples of MATLAB's parula colormap (blue → teal → green → yellow),
// linearly interpolated to the requested length. Also used by the `parula`
// builtin.
export const PARULA_ANCHORS: RGB[] = [
  [0.2422, 0.1504, 0.6603],
  [0.278, 0.3556, 0.9777],
  [0.1085, 0.5267, 0.8943],
  [0.0469, 0.6353, 0.7861],
  [0.2161, 0.7269, 0.6499],
  [0.5044, 0.7993, 0.4519],
  [0.8328, 0.8056, 0.2008],
  [0.9871, 0.8092, 0.1432],
  [0.9763, 0.9831, 0.0538],
];

/** Linearly interpolate an N×3 table at t ∈ [0, 1]. */
function interpTable(data: number[][], t: number): RGB {
  const n = data.length;
  if (n === 1) return data[0] as RGB;
  const idx = t * (n - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, n - 1);
  const frac = idx - lo;
  return [
    data[lo][0] + frac * (data[hi][0] - data[lo][0]),
    data[lo][1] + frac * (data[hi][1] - data[lo][1]),
    data[lo][2] + frac * (data[hi][2] - data[lo][2]),
  ];
}

function jetColormap(t: number): RGB {
  let r: number, g: number, b: number;
  if (t < 0.125) {
    r = 0;
    g = 0;
    b = 0.5 + t * 4;
  } else if (t < 0.375) {
    r = 0;
    g = (t - 0.125) * 4;
    b = 1;
  } else if (t < 0.625) {
    r = (t - 0.375) * 4;
    g = 1;
    b = 1 - (t - 0.375) * 4;
  } else if (t < 0.875) {
    r = 1;
    g = 1 - (t - 0.625) * 4;
    b = 0;
  } else {
    r = 1 - (t - 0.875) * 4;
    g = 0;
    b = 0;
  }
  return [
    Math.max(0, Math.min(1, r)),
    Math.max(0, Math.min(1, g)),
    Math.max(0, Math.min(1, b)),
  ];
}

function redblueColormap(t: number): RGB {
  // Blue (0) → White (0.5) → Red (1)
  if (t < 0.5) {
    const s = t * 2;
    return [s, s, 1];
  }
  const s = (t - 0.5) * 2;
  return [1, 1 - s, 1 - s];
}

/** Color at t ∈ [0, 1] (clamped; non-finite → 0) in the given colormap.
 *  Unknown names fall back to parula. */
export function colormapLookup(t: number, cm: ColormapSpec = {}): RGB {
  // Non-finite t would give NaN components, which canvas rejects as a
  // fillStyle (silently keeping the previous color); use 0 instead.
  const clamped = Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0;
  if (cm.data && cm.data.length > 0) return interpTable(cm.data, clamped);
  if (cm.name === "jet") return jetColormap(clamped);
  if (cm.name === "redblue") return redblueColormap(clamped);
  return interpTable(PARULA_ANCHORS, clamped);
}

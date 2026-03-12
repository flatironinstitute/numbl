/** Parula-inspired colormap for surface plots */

const COLORMAP: [number, number, number][] = [
  [0.2422, 0.1504, 0.6603],
  [0.281, 0.3228, 0.9579],
  [0.1786, 0.5289, 0.9682],
  [0.0689, 0.6948, 0.8394],
  [0.128, 0.789, 0.5927],
  [0.3391, 0.849, 0.3798],
  [0.633, 0.8518, 0.2091],
  [0.8902, 0.8044, 0.1137],
  [0.9905, 0.6816, 0.0235],
  [0.9763, 0.517, 0.034],
];

export function colormapLookup(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = clamped * (COLORMAP.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, COLORMAP.length - 1);
  const frac = idx - lo;
  return [
    COLORMAP[lo][0] + frac * (COLORMAP[hi][0] - COLORMAP[lo][0]),
    COLORMAP[lo][1] + frac * (COLORMAP[hi][1] - COLORMAP[lo][1]),
    COLORMAP[lo][2] + frac * (COLORMAP[hi][2] - COLORMAP[lo][2]),
  ];
}

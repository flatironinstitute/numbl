/**
 * Selection + ordering logic for `eigs` (a subset of eigenvalues).
 *
 * `eigs` in numbl piggybacks on the dense `eig`: the runtime computes every
 * eigenvalue, then this module picks the requested `k` of them according to
 * `sigma` and returns their original indices in the order `eigs` reports
 * them. It is pure (operates on plain number arrays) so it can be unit
 * tested without the runtime.
 */

export type SigmaKind =
  | "largestabs"
  | "smallestabs"
  | "largestreal"
  | "smallestreal"
  | "bothendsreal"
  | "largestimag"
  | "smallestimag"
  | "bothendsimag";

export type SigmaSpec =
  | { kind: SigmaKind }
  | { kind: "scalar"; re: number; im: number };

/** Text sigma values and their short aliases. */
const SIGMA_ALIASES: Record<string, SigmaKind> = {
  lm: "largestabs",
  largestabs: "largestabs",
  sm: "smallestabs",
  smallestabs: "smallestabs",
  lr: "largestreal",
  la: "largestreal",
  largestreal: "largestreal",
  sr: "smallestreal",
  sa: "smallestreal",
  smallestreal: "smallestreal",
  be: "bothendsreal",
  bothendsreal: "bothendsreal",
  li: "largestimag",
  largestimag: "largestimag",
  si: "smallestimag",
  smallestimag: "smallestimag",
  bothendsimag: "bothendsimag",
};

/** Map a text sigma value to a canonical kind, or `null` if unrecognized. */
export function normalizeSigmaString(s: string): SigmaKind | null {
  return SIGMA_ALIASES[s.trim().toLowerCase()] ?? null;
}

const mag = (re: number, im: number) => Math.hypot(re, im);

/** Indices `0..n-1` sorted by `key`, descending or ascending, with the
 *  original index as a stable tiebreaker. */
function sortedBy(
  n: number,
  key: (i: number) => number,
  descending: boolean
): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => {
    const ka = key(a);
    const kb = key(b);
    if (ka !== kb) return descending ? kb - ka : ka - kb;
    return a - b;
  });
  return idx;
}

/** Combine the high/low ends used by the `bothends*` sigma options: take the
 *  top `nHigh` by `key`, then the bottom `nLow` (excluding the high picks). */
function bothEnds(
  n: number,
  key: (i: number) => number,
  k: number
): { high: number[]; low: number[] } {
  const nHigh = Math.ceil(k / 2);
  const nLow = Math.floor(k / 2);
  const desc = sortedBy(n, key, true);
  const high = desc.slice(0, nHigh);
  const highSet = new Set(high);
  const low: number[] = [];
  for (let i = desc.length - 1; i >= 0 && low.length < nLow; i--) {
    if (!highSet.has(desc[i])) low.push(desc[i]);
  }
  return { high, low };
}

/**
 * Pick `k` eigenvalues (by original index) and return them in the order
 * `eigs` reports for the given `sigma`. `re`/`im` are the real/imaginary
 * parts of all `n` eigenvalues.
 */
export function selectEigsIndices(
  re: number[],
  im: number[],
  k: number,
  sigma: SigmaSpec
): number[] {
  const n = re.length;
  const kk = Math.max(0, Math.min(k, n));
  switch (sigma.kind) {
    case "largestabs":
      return sortedBy(n, i => mag(re[i], im[i]), true).slice(0, kk);
    case "smallestabs":
      return sortedBy(n, i => mag(re[i], im[i]), false).slice(0, kk);
    case "largestreal":
      return sortedBy(n, i => re[i], true).slice(0, kk);
    case "smallestreal":
      return sortedBy(n, i => re[i], false).slice(0, kk);
    case "largestimag":
      return sortedBy(n, i => im[i], true).slice(0, kk);
    case "smallestimag":
      return sortedBy(n, i => im[i], false).slice(0, kk);
    case "scalar": {
      const { re: sr, im: si } = sigma;
      return sortedBy(n, i => mag(re[i] - sr, im[i] - si), false).slice(0, kk);
    }
    case "bothendsreal": {
      const { high, low } = bothEnds(n, i => re[i], kk);
      // Output ascending by real part.
      return [...high, ...low].sort((a, b) => re[a] - re[b] || a - b);
    }
    case "bothendsimag": {
      const { high, low } = bothEnds(n, i => im[i], kk);
      // Output descending by magnitude of the imaginary part.
      return [...high, ...low].sort(
        (a, b) => Math.abs(im[b]) - Math.abs(im[a]) || a - b
      );
    }
  }
}

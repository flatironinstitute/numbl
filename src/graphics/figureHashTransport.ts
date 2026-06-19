/**
 * Encode/decode a figure into a URL hash fragment, for handing a figure from
 * numbl to the figure viewer in a new tab.
 *
 * Why the URL and not postMessage: numbl sets COOP `same-origin` (for
 * SharedArrayBuffer / crossOriginIsolated), which severs the opener⇄popup
 * relationship for a cross-origin viewer tab — `window.opener` is null and the
 * opener's window handle reports `closed`, so postMessage can't reach it. The
 * navigation URL is the one channel that survives, so the figure rides in the
 * hash.
 *
 * Payload = gzip(JSON(figure)) → base64url. JSON is made NaN/Infinity-safe with
 * sentinels (JSON.stringify would otherwise turn them into null). Encoding is
 * synchronous so the caller can `window.open` within the click gesture.
 */
import type { FigureState } from "./figuresReducer.js";
import { gzipSync, gunzipSync, strToU8, strFromU8 } from "fflate";

export const FIGURE_HASH_KEY = "fig";
const NAN = "@@NaN@@";
const INF = "@@Inf@@";
const NINF = "@@-Inf@@";

function replacer(_k: string, v: unknown): unknown {
  if (typeof v === "number" && !Number.isFinite(v))
    return Number.isNaN(v) ? NAN : v > 0 ? INF : NINF;
  return v;
}
function reviver(_k: string, v: unknown): unknown {
  if (v === NAN) return NaN;
  if (v === INF) return Infinity;
  if (v === NINF) return -Infinity;
  return v;
}

export function bytesToBase64url(bytes: Uint8Array): string {
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk)
    bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
export function base64urlToBytes(s: string): Uint8Array {
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/"));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/** `fig=<base64url(gzip(json))>` — append to a viewer URL after `#`. */
export function encodeFigureToHash(figure: FigureState): string {
  const gz = gzipSync(strToU8(JSON.stringify(figure, replacer)));
  return `${FIGURE_HASH_KEY}=${bytesToBase64url(gz)}`;
}

/** Parse a figure out of a `location.hash` (or any `fig=…` string); null if
 *  absent or malformed. */
export function decodeFigureFromHash(hash: string): FigureState | null {
  const h = hash.replace(/^#/, "");
  const prefix = `${FIGURE_HASH_KEY}=`;
  if (!h.startsWith(prefix)) return null;
  try {
    const json = strFromU8(
      gunzipSync(base64urlToBytes(h.slice(prefix.length)))
    );
    const fig = JSON.parse(json, reviver) as FigureState;
    if (fig && typeof fig === "object" && (fig.axes || fig.uihtml)) return fig;
  } catch {
    /* malformed payload */
  }
  return null;
}

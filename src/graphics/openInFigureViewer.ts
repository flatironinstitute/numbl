import type { FigureState } from "./figuresReducer.js";
import { encodeFigureToHash } from "./figureHashTransport.js";

/** Deployed numbl figure viewer (concept-collection). */
export const DEFAULT_FIGURE_VIEWER_URL =
  "https://concept-collection.github.io/numbl-figure-viewer/";

/** Conservative ceiling for a window.open URL (browsers cap these at ~2 MB). */
const MAX_URL_LEN = 1_500_000;

/** Resolve the viewer URL: explicit arg › localStorage dev override › default.
 *  For local development, set in the numbl tab's console:
 *    localStorage.setItem("numblFigureViewerUrl", "http://localhost:5173/") */
export function resolveFigureViewerUrl(explicit?: string): string {
  if (explicit) return explicit;
  try {
    const override = localStorage.getItem("numblFigureViewerUrl");
    if (override) return override;
  } catch {
    /* localStorage unavailable */
  }
  return DEFAULT_FIGURE_VIEWER_URL;
}

export interface FigureViewerLink {
  /** Viewer URL with the figure embedded in the hash, or null if too large. */
  url: string | null;
  /** Viewer base URL (open empty, for loading a downloaded .h5 manually). */
  baseUrl: string;
  /** True when the figure won't fit in a URL and must be handed over as a file. */
  tooLarge: boolean;
}

/**
 * Build the link that opens the figure viewer with `figure` embedded in the URL
 * hash (gzip+base64url, NaN-safe — see figureHashTransport). The hash is the one
 * channel that survives numbl's COOP isolation (postMessage to a cross-origin
 * popup is blocked there).
 *
 * Pure — it opens nothing. The caller decides: open `url` in response to the
 * user's click when it fits, or, when `tooLarge`, show an in-page message with a
 * link to `baseUrl` so the user can open the viewer and load a downloaded .h5.
 */
export function buildFigureViewerLink(
  figure: FigureState,
  viewerUrl?: string
): FigureViewerLink {
  const baseUrl = resolveFigureViewerUrl(viewerUrl).replace(/#.*$/, "");
  const url = `${baseUrl}#${encodeFigureToHash(figure)}`;
  const tooLarge = url.length > MAX_URL_LEN;
  return { url: tooLarge ? null : url, baseUrl, tooLarge };
}

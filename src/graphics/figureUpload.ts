/**
 * Fallback figure hand-off for figures too large to fit in a URL hash: upload
 * the figure (as an encrypted `.h5`) to a temporary file host and pass only the
 * download URL + decryption key to the viewer.
 *
 * Privacy: the figure is AES-GCM encrypted in the browser before upload, and the
 * key travels only in the viewer URL's fragment (never sent to any server), so
 * the host stores opaque bytes. See loadFigureFromHash for the viewer side.
 *
 * The uploader is pluggable (FigureUploader) — numbl's own figure-store relay
 * (a Cloudflare Worker on a numbl.org subdomain) is the default; swap it via
 * setFigureUploader for any service that returns its own download URL. The
 * viewer just fetches whatever URL it's handed.
 */
import type { FigureState } from "./figuresReducer.js";
import { exportFigureHdf5 } from "./exportFigureHdf5.js";
import { importFigureHdf5 } from "./importFigureHdf5.js";
import {
  decodeFigureFromHash,
  bytesToBase64url,
  base64urlToBytes,
} from "./figureHashTransport.js";
import { resolveFigureViewerUrl } from "./openInFigureViewer.js";

/** Uploads bytes to a temporary host and resolves to a URL the viewer can GET
 *  (cross-origin) to retrieve them. Throws on failure. */
export type FigureUploader = (
  data: Uint8Array,
  filename: string
) => Promise<string>;

/** numbl's figure-store relay (Cloudflare Worker). See figure-store-spec.md.
 *  Override for local/staging via: localStorage["numblFigureStoreUrl"]. */
export const DEFAULT_FIGURE_STORE_URL = "https://figures.numbl.org/";

function resolveFigureStoreUrl(): string {
  try {
    const override = localStorage.getItem("numblFigureStoreUrl");
    if (override) return override;
  } catch {
    /* localStorage unavailable */
  }
  return DEFAULT_FIGURE_STORE_URL;
}

/** Default uploader: numbl's figure-store relay. Uploads the opaque (encrypted)
 *  bytes as a raw octet-stream body and returns the download URL from the
 *  response `{ url }`. Until the service is deployed, this rejects and the
 *  caller falls back to the manual download/upload message. */
export const numblStoreUploader: FigureUploader = async data => {
  const resp = await fetch(resolveFigureStoreUrl(), {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: data as BodyInit,
  });
  if (!resp.ok) throw new Error(`upload failed (HTTP ${resp.status})`);
  const json = (await resp.json()) as { url?: string };
  if (!json.url) throw new Error("upload rejected by store");
  return json.url;
};

let activeUploader: FigureUploader = numblStoreUploader;

/** Swap the temporary-host uploader (e.g. a self-hosted service). */
export function setFigureUploader(uploader: FigureUploader): void {
  activeUploader = uploader;
}

// ── encryption (AES-GCM; key + iv live only in the viewer URL fragment) ──────

async function encrypt(
  data: Uint8Array
): Promise<{ ciphertext: Uint8Array; key: string; iv: string }> {
  const rawKey = crypto.getRandomValues(new Uint8Array(32));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const cryptoKey = await crypto.subtle.importKey(
    "raw",
    rawKey,
    "AES-GCM",
    false,
    ["encrypt"]
  );
  const ct = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv },
    cryptoKey,
    data as BufferSource
  );
  return {
    ciphertext: new Uint8Array(ct),
    key: bytesToBase64url(rawKey),
    iv: bytesToBase64url(iv),
  };
}

async function decrypt(
  ciphertext: Uint8Array,
  key: string,
  iv: string
): Promise<Uint8Array> {
  const cryptoKey = await crypto.subtle.importKey(
    "raw",
    base64urlToBytes(key) as BufferSource,
    "AES-GCM",
    false,
    ["decrypt"]
  );
  const pt = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: base64urlToBytes(iv) as BufferSource },
    cryptoKey,
    ciphertext as BufferSource
  );
  return new Uint8Array(pt);
}

/**
 * Export the figure to an encrypted `.h5`, upload it, and return a viewer URL
 * referencing it: `<base>#u=<download-url>&k=<key>&iv=<iv>`. Throws if the
 * upload fails (callers fall back to manual download). Asynchronous, so the
 * caller can't `window.open` the result directly (popup blocking) — surface the
 * URL as a link for the user to click.
 */
export async function uploadFigureForViewer(
  figure: FigureState,
  opts: { viewerUrl?: string; uploader?: FigureUploader } = {}
): Promise<string> {
  const base = resolveFigureViewerUrl(opts.viewerUrl).replace(/#.*$/, "");
  const h5 = await exportFigureHdf5(figure);
  const { ciphertext, key, iv } = await encrypt(h5);
  const downloadUrl = await (opts.uploader ?? activeUploader)(
    ciphertext,
    "figure.h5.enc"
  );
  const params = new URLSearchParams({ u: downloadUrl, k: key, iv });
  return `${base}#${params.toString()}`;
}

/**
 * Viewer side: load a figure from a viewer URL hash. Handles both the direct
 * (`fig=…`) and uploaded (`u=…&k=…&iv=…`) forms. Returns null if the hash
 * carries no figure; throws if a referenced upload can't be fetched/decrypted.
 */
export async function loadFigureFromHash(
  hash: string
): Promise<FigureState | null> {
  const direct = decodeFigureFromHash(hash);
  if (direct) return direct;

  const params = new URLSearchParams(hash.replace(/^#/, ""));
  const u = params.get("u");
  const k = params.get("k");
  const iv = params.get("iv");
  if (!u || !k || !iv) return null;

  const resp = await fetch(u);
  if (!resp.ok) throw new Error(`could not fetch figure (HTTP ${resp.status})`);
  const ciphertext = new Uint8Array(await resp.arrayBuffer());
  const h5 = await decrypt(ciphertext, k, iv);
  return importFigureHdf5(h5);
}

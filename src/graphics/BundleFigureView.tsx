import { useEffect, useRef, useState } from "react";

/**
 * Renders a "directory figure": a self-contained set of static files (an
 * `index.html` plus its assets/data) shown in an iframe.
 *
 * The files never touch disk and are never uploaded. They are written to the
 * `numbl-figures` Cache and served to a same-origin iframe by a service worker
 * (`public/figure-sw.js`, scope `/figs/`) that also honors byte-range
 * requests. This keeps the renderer a black box: any static site — a
 * hand-written canvas demo, a figpack bundle, etc. — drops in unchanged.
 */

const CACHE_NAME = "numbl-figures";
const SW_URL = "/figure-sw.js";
const SW_SCOPE = "/figs/";

const MIME: Record<string, string> = {
  html: "text/html; charset=utf-8",
  htm: "text/html; charset=utf-8",
  js: "text/javascript; charset=utf-8",
  mjs: "text/javascript; charset=utf-8",
  css: "text/css; charset=utf-8",
  json: "application/json; charset=utf-8",
  svg: "image/svg+xml",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  wasm: "application/wasm",
  txt: "text/plain; charset=utf-8",
  csv: "text/csv; charset=utf-8",
};

function mimeOf(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  return MIME[ext] ?? "application/octet-stream";
}

// Register the figure service worker at most once per session.
let swRegistration: Promise<void> | null = null;

function ensureFigureSW(): Promise<void> {
  if (!("serviceWorker" in navigator)) {
    return Promise.reject(
      new Error("Service workers are not available in this context.")
    );
  }
  if (!swRegistration) {
    swRegistration = navigator.serviceWorker
      .register(SW_URL, { scope: SW_SCOPE })
      .then(waitForActive)
      .catch(err => {
        swRegistration = null; // allow a later retry
        throw err;
      });
  }
  return swRegistration;
}

function waitForActive(reg: ServiceWorkerRegistration): Promise<void> {
  if (reg.active) return Promise.resolve();
  const sw = reg.installing || reg.waiting;
  if (!sw) return Promise.resolve();
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(
      () =>
        reject(new Error("Timed out activating the figure service worker.")),
      10000
    );
    sw.addEventListener("statechange", () => {
      if (sw.state === "activated") {
        clearTimeout(timer);
        resolve();
      }
    });
  });
}

async function publishBundle(
  id: string,
  files: Map<string, string | Uint8Array>
): Promise<void> {
  const cache = await caches.open(CACHE_NAME);
  const prefix = `${SW_SCOPE}${id}/`;
  // Clear any stale files cached under this id (e.g. from a previous run).
  const existing = await cache.keys();
  await Promise.all(
    existing
      .filter(req => new URL(req.url).pathname.startsWith(prefix))
      .map(req => cache.delete(req))
  );
  // Write the new files.
  await Promise.all(
    [...files].map(([path, content]) => {
      // Uint8Array is a valid BodyInit at runtime; the cast satisfies the
      // stricter typed-array generics in newer TS lib definitions.
      const body: BodyInit =
        typeof content === "string"
          ? content
          : (content as unknown as BodyInit);
      return cache.put(
        prefix + path.replace(/^\/+/, ""),
        new Response(body, {
          headers: {
            "Content-Type": mimeOf(path),
            // Lets the iframe be embedded by the cross-origin-isolated parent.
            "Cross-Origin-Resource-Policy": "same-origin",
          },
        })
      );
    })
  );
}

interface BundleFigureViewProps {
  id: string;
  files: Map<string, string | Uint8Array>;
}

export function BundleFigureView({ id, files }: BundleFigureViewProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // Two delivery modes for the same iframe URL (/figs/<id>/index.html):
        //  - files carried in the instruction (browser IDE): publish them to
        //    the Cache and let the figure service worker serve them.
        //  - files absent (CLI): the plot server already serves them from
        //    memory, so just point the iframe at the URL.
        if (files.size > 0) {
          await ensureFigureSW();
          if (cancelled) return;
          await publishBundle(id, files);
          if (cancelled) return;
        }
        const iframe = iframeRef.current;
        if (iframe) iframe.src = `${SW_SCOPE}${id}/index.html`;
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id, files]);

  if (error) {
    return (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 16,
          boxSizing: "border-box",
          color: "#a00",
          fontFamily: "sans-serif",
          fontSize: 13,
          textAlign: "center",
        }}
      >
        <div>
          <strong>Could not display this figure.</strong>
          <div style={{ marginTop: 6, fontSize: 12 }}>{error}</div>
        </div>
      </div>
    );
  }

  return (
    <iframe
      ref={iframeRef}
      title={`figure-${id}`}
      style={{
        width: "100%",
        height: "100%",
        border: "none",
        display: "block",
      }}
    />
  );
}

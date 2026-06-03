/*
 * figure-sw.js — serves in-memory "directory figures" to same-origin iframes.
 *
 * Registered by src/graphics/BundleFigureView.tsx with scope /figs/. The page
 * places a figure's files in the `numbl-figures` Cache under /figs/<id>/...;
 * this worker reads them and, for byte-range requests, synthesizes 206 Partial
 * Content responses (needed by viewers like figpack that read chunked data via
 * Range). No files touch disk and nothing is uploaded.
 *
 * Scoped to /figs/ so it coexists with coi-serviceworker.js (scope /): the
 * iframe document and its subresources are served here; everything else stays
 * under the cross-origin-isolation worker.
 */

const CACHE_NAME = "numbl-figures";

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", event =>
  event.waitUntil(self.clients.claim())
);

self.addEventListener("fetch", event => {
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if (!url.pathname.startsWith("/figs/")) return;
  event.respondWith(serve(event.request, url));
});

async function serve(request, url) {
  const cache = await caches.open(CACHE_NAME);
  let key = url.pathname;
  if (key.endsWith("/")) key += "index.html";

  const cached = await cache.match(key);
  if (!cached) {
    return new Response("Not found: " + key, {
      status: 404,
      headers: { "Content-Type": "text/plain" },
    });
  }

  const ct = cached.headers.get("Content-Type") || "application/octet-stream";
  const buf = await cached.arrayBuffer();
  const total = buf.byteLength;

  // The embedding IDE is cross-origin isolated (COEP: require-corp via
  // coi-serviceworker), so a nested document must itself assert COEP — CORP
  // alone is not enough for an iframe. Stamp both on every response.
  const base = {
    "Content-Type": ct,
    "Cross-Origin-Resource-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Accept-Ranges": "bytes",
  };

  const range = request.headers.get("range");
  if (!range) {
    return new Response(buf, {
      status: 200,
      headers: { ...base, "Content-Length": String(total) },
    });
  }

  // Honor "Range: bytes=start-end" with a 206 response.
  const m = /bytes=(\d*)-(\d*)/.exec(range);
  let start = m && m[1] ? parseInt(m[1], 10) : 0;
  let end = m && m[2] ? parseInt(m[2], 10) : total - 1;
  if (isNaN(start)) start = 0;
  if (isNaN(end) || end >= total) end = total - 1;
  if (start > end || start >= total) {
    return new Response(null, {
      status: 416,
      headers: { ...base, "Content-Range": "bytes */" + total },
    });
  }

  const slice = buf.slice(start, end + 1);
  return new Response(slice, {
    status: 206,
    headers: {
      ...base,
      "Content-Range": "bytes " + start + "-" + end + "/" + total,
      "Content-Length": String(slice.byteLength),
    },
  });
}

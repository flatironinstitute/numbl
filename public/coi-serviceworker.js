/*
 * Cross-Origin Isolation Service Worker
 *
 * Adds COOP/COEP headers to responses so that SharedArrayBuffer is available
 * on hosts that don't allow custom response headers (e.g. GitHub Pages).
 *
 * Based on https://github.com/niccokunzmann/coi-serviceworker (MIT).
 */

/* eslint-env serviceworker */

if (typeof window === "undefined") {
  // --- Service Worker scope ---
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", event =>
    event.waitUntil(self.clients.claim())
  );

  self.addEventListener("fetch", event => {
    const request = event.request;
    if (request.cache === "only-if-cached" && request.mode !== "same-origin") {
      return; // Chrome bug workaround
    }
    event.respondWith(
      fetch(request).then(response => {
        if (response.status === 0) return response; // opaque response

        const headers = new Headers(response.headers);
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        headers.set("Cross-Origin-Opener-Policy", "same-origin");

        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers,
        });
      })
    );
  });
} else {
  // --- Window scope (registration) ---

  // Capture currentScript synchronously — it becomes null after script runs.
  const scriptUrl = document.currentScript && document.currentScript.src;

  if (!window.crossOriginIsolated && navigator.serviceWorker) {
    navigator.serviceWorker.register(scriptUrl || "/coi-serviceworker.js").then(
      reg => {
        if (reg.installing || reg.waiting) {
          const sw = reg.installing || reg.waiting;
          sw.addEventListener("statechange", () => {
            if (sw.state === "activated") window.location.reload();
          });
        } else if (reg.active && !navigator.serviceWorker.controller) {
          // Active but not yet controlling — reload to let it intercept.
          window.location.reload();
        }
      },
      err => console.error("COI service worker registration failed:", err)
    );
  }
}

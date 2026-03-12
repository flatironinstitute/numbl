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
  (() => {
    // Already cross-origin isolated — nothing to do.
    if (window.crossOriginIsolated) return;

    const register = () =>
      navigator.serviceWorker
        .register(window.document.currentScript.src)
        .then(reg => {
          // If a new service worker installed and is waiting, reload once
          // so that it can intercept requests with the right headers.
          if (reg.installing || reg.waiting) {
            const sw = reg.installing || reg.waiting;
            sw.addEventListener("statechange", () => {
              if (sw.state === "activated") window.location.reload();
            });
          } else if (reg.active && !navigator.serviceWorker.controller) {
            // Service worker is active but not controlling the page (first load).
            window.location.reload();
          }
        });

    if (navigator.serviceWorker.controller) {
      // Already controlled — SW is active but isolation might not be in effect
      // yet (e.g. after a deploy with new headers). Check & reload if needed.
      // crossOriginIsolated was already checked above, so if we're here
      // the SW isn't adding headers. Re-register in case the SW code changed.
      register();
    } else {
      register();
    }
  })();
}

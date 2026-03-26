/**
 * Synchronous sleep using Atomics.wait with a timeout.
 * Works in Web Workers and Node.js (including main thread).
 * No SharedArrayBuffer communication needed — just a local buffer.
 */
let syncSleepWarned = false;

export function syncSleep(ms: number): void {
  if (ms <= 0) return;
  if (typeof SharedArrayBuffer !== "undefined") {
    const sab = new SharedArrayBuffer(4);
    const int32 = new Int32Array(sab);
    Atomics.wait(int32, 0, 0, ms);
  } else {
    // Fallback: busy-wait when SharedArrayBuffer is unavailable
    // (e.g., browser without Cross-Origin Isolation headers)
    if (!syncSleepWarned) {
      syncSleepWarned = true;
      console.warn(
        "SharedArrayBuffer is not available — pause() will busy-wait. " +
          "Enable Cross-Origin Isolation headers (COOP/COEP) for efficient blocking."
      );
    }
    const end = Date.now() + ms;
    while (Date.now() < end) {
      /* spin */
    }
  }
}

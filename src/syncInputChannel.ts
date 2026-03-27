/**
 * Synchronous input channel between a web worker and the main thread.
 *
 * Protocol uses a SharedArrayBuffer:
 *   Int32[0] = signal: 0 = idle, 1 = response ready
 *   Int32[1] = response byte length
 *   Uint8[8..] = response as UTF-8 bytes
 *
 * Worker side: posts { type: "request-input", prompt } then Atomics.wait().
 * Main side: receives the message, collects input, writes to SAB, Atomics.notify().
 */

const SAB_SIZE = 8 + 8192; // 8 bytes header + 8KB for response text

/** Create a SharedArrayBuffer for the input channel. */
export function createInputSAB(): SharedArrayBuffer {
  return new SharedArrayBuffer(SAB_SIZE);
}

/**
 * Worker-side: create an onInput callback that blocks until the main thread responds.
 */
export function workerOnInput(
  sab: SharedArrayBuffer
): (prompt: string) => string {
  const int32 = new Int32Array(sab);
  const uint8 = new Uint8Array(sab);

  return (prompt: string): string => {
    // Reset signal
    Atomics.store(int32, 0, 0);

    // Ask main thread for input
    self.postMessage({ type: "request-input", prompt });

    // Block until main thread writes response
    Atomics.wait(int32, 0, 0);

    // Read response
    const byteLen = Atomics.load(int32, 1);
    const bytes = uint8.slice(8, 8 + byteLen);
    return new TextDecoder().decode(bytes);
  };
}

/**
 * Main-thread side: write a response string into the SAB and wake the worker.
 */
export function mainThreadRespond(
  sab: SharedArrayBuffer,
  response: string
): void {
  const int32 = new Int32Array(sab);
  const uint8 = new Uint8Array(sab);

  const encoded = new TextEncoder().encode(response);
  const byteLen = Math.min(encoded.length, SAB_SIZE - 8);

  // Write response bytes
  uint8.set(encoded.subarray(0, byteLen), 8);
  // Write length
  Atomics.store(int32, 1, byteLen);
  // Signal ready and wake worker
  Atomics.store(int32, 0, 1);
  Atomics.notify(int32, 0);
}

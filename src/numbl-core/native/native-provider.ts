/**
 * The native addon as a layer of the LAPACK provider stack (see
 * lapack-bridge.ts). Size thresholds below which the addon is slower than
 * the TypeScript path belong in `accepts` here, not in callers.
 */

import {
  PROVIDER_PRIORITY,
  type LapackBridge,
  type LapackProvider,
} from "./lapack-bridge.js";

/** At or below this length the JS FFT beats an FFTW call. */
const NATIVE_FFT1D_MAX_JS_LENGTH = 128;

export function nativeAddonProvider(
  addon: Partial<LapackBridge>
): LapackProvider {
  return {
    name: "native",
    priority: PROVIDER_PRIORITY.native,
    entries: addon,
    accepts: {
      fft1dComplex: (_re, _im, n) => n > NATIVE_FFT1D_MAX_JS_LENGTH,
    },
  };
}

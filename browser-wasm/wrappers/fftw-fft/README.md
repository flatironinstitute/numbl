# fftw FFT wrapper

This directory contains a minimal browser-Wasm wrapper around the Emscripten
FFTW port.

The exported C ABI is:

- `numbl_fft1d_f64`

The wrapper is intentionally 1D-only. It is meant to be a low-risk proof that
FFTW-backed browser Wasm is viable without a separate upstream FFTW source
tree. The browser bridge can still discover and use it as a fallback FFT
kernel.

Build it through the browser Wasm target:

```bash
node scripts/build-browser-wasm.mjs fftw-fft
```

The default release profile is `-O3 -flto -msimd128 -DNDEBUG` plus the safe
floating-point flags `-fno-fast-math -fno-math-errno -ffp-contract=on`. Set
`NUMBL_BROWSER_WASM_FAST_MATH=1` if you explicitly want `-ffast-math`.

The build requires `emcc` and the Emscripten FFTW port.

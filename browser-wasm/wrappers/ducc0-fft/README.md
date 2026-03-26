# ducc0 FFT wrapper

This directory contains a minimal browser-Wasm wrapper around `ducc0` FFT.

The wrapper exports a flat C ABI:

- `numbl_fft1d_f64`
- `numbl_fft_along_dim_f64`

Build it by pointing `DUCC0_SRC_ROOT` at a ducc checkout:

```bash
cmake -S browser-wasm/wrappers/ducc0-fft -B build/ducc0-fft -DDUCC0_SRC_ROOT=/abs/path/to/ducc
cmake --build build/ducc0-fft
```

Numbl's browser build stages this wrapper into a temporary source tree and
copies the upstream `src/ducc0` subtree beside it before invoking Emscripten.

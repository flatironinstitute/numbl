# Browser Wasm Artifacts

Generated browser-side Wasm binaries are written here by:

```bash
npm run build:browser-wasm
```

The build command reads external source roots from per-target environment
variables or `browser-wasm/local-sources.json`.

Expected generated files:

- `public/wasm-kernels/ducc0-fft.wasm`
- `public/wasm-kernels/blas-lapack.wasm`
- `public/wasm-kernels/manifest.json`

This directory is part of the web app's static asset tree, so anything built
here can be fetched by the browser at runtime.

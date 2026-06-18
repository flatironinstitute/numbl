// Ambient declaration for the Vite `?url` asset import of the qhull `.wasm`
// binary. Importing with the `?url` suffix yields the emitted asset's URL,
// which we fetch and hand to the emscripten module as `wasmBinary` (see
// qhull-browser.ts) so the binary need not be located next to the glue at
// runtime.
declare module "qhull-wasm/dist/qhull.wasm?url" {
  const url: string;
  export default url;
}

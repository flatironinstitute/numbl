/**
 * Browser entry point for the `numbl/graphics` subpath export.
 *
 * Exposes numbl's React figure renderer and the HDF5 figure import/export so
 * external apps can display numbl figures (e.g. open a downloaded `.h5`) without
 * the rest of numbl-core. Built for the browser (ESM); `react`/`react-dom` are
 * peer dependencies and `h5wasm` is loaded lazily inside the import/export
 * helpers. See docs/figure-hdf5-format.md for the file format.
 */

export { FigureView } from "./graphics/FigureView.js";
export {
  figuresReducer,
  initialFiguresState,
} from "./graphics/figuresReducer.js";

export type {
  FigureState,
  AxesState,
  FiguresState,
  FiguresAction,
} from "./graphics/figuresReducer.js";

export type * from "./graphics/types.js";

export {
  exportFigureHdf5,
  downloadFigureHdf5,
} from "./graphics/exportFigureHdf5.js";
export { importFigureHdf5 } from "./graphics/importFigureHdf5.js";
export { FIGURE_HDF5_VERSION } from "./graphics/figureHdf5Schema.js";
export {
  buildFigureViewerLink,
  DEFAULT_FIGURE_VIEWER_URL,
} from "./graphics/openInFigureViewer.js";
export type { FigureViewerLink } from "./graphics/openInFigureViewer.js";
export {
  uploadFigureForViewer,
  loadFigureFromHash,
  numblStoreUploader,
  setFigureUploader,
  DEFAULT_FIGURE_STORE_URL,
} from "./graphics/figureUpload.js";
export type { FigureUploader } from "./graphics/figureUpload.js";
export {
  encodeFigureToHash,
  decodeFigureFromHash,
} from "./graphics/figureHashTransport.js";

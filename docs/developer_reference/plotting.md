# Plotting

## Instructions, not pixels

The interpreter's plotting commands (`plot`, `surf`, `imagesc`, `figure`, `hold`, etc.) do not render anything themselves. They emit serializable `PlotInstruction` objects — a structured description of the figure state — into an accumulator on the runtime. At flush points (`drawnow`, end of script, explicit refresh), the runtime hands the accumulated instructions to an `onDrawnow` callback provided by the platform.

This keeps numbl-core free of any rendering code and lets every platform surface render however it likes:

- **CLI** — with `--plot`, serves the prebuilt plot viewer bundle (`dist-plot-viewer/`) from a local HTTP server and pushes instructions to a connected browser.
- **Web app** — renders in-page: the IDE, gallery and embed pages (and the static site viewer) import `FigureView` from `src/graphics/` and compile it from source, so they do not use the viewer bundle.
- **Server** — normally forwards instructions back to the calling client.

## Code layout

The core half lives in `src/numbl-core/plotting/`:

| File                                                | Role                                                                                      |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `types.ts`                                          | Trace interfaces and the `PlotInstruction` union (the contract with the renderer)         |
| `parseArgsCommon.ts`                                | Shared argument helpers: colors, LineSpec, Name-Value keys, numeric conversion            |
| `parseArgs{Line,Patch,Surface,Chart,Quiver}.ts`     | Argument parsers, one file per plot family                                                |
| `instructions.ts`                                   | One `*Call` per plot builtin: parse the arguments, push the instructions                  |
| `streamline.ts`, `axisCommand.ts`                   | `streamline`/`stream2` and the `axis(...)` forms                                          |
| `dispatch.ts`                                       | `dispatchPlotBuiltin`, the `(name, args)` → instructions mapping, and the plot name lists |
| `builtins.ts`                                       | Builtin registrations; adds handle return values and the runtime's figure/hold state      |
| `axesTraces.ts`                                     | The per-kind trace fields of an axes, declared once (see below)                           |
| `figuresReducer.ts`                                 | Pure reducer from instructions to `FiguresState`                                          |
| `axisLimits.ts`, `imagescExtent.ts`, `colormaps.ts` | Pure helpers used by both the runtime (`lim = axis`, `parula`) and the renderer           |

The renderer half lives in `src/graphics/`. `FigureView.tsx` lays out a figure's axes and routes each one either to `SurfView.tsx` (Three.js, for axes with 3-D content) or to `drawPlot.ts` (Canvas 2D). `drawPlot` draws what all 2-D kinds share (data limits, axis modes and scales, grid, ticks, labels, legend, colorbar) and delegates each kind's drawing to a renderer in `src/graphics/renderers/`, looked up in the `CANVAS_RENDERERS` table in `renderers/index.ts`. A renderer either draws inside the shared data axes or, like the pie chart and heatmap, takes over the whole canvas.

### Trace fields

`AxesState` holds each plot kind's traces in its own field (`barTraces`, `imagescTrace`, ...). These fields are declared once, in `axesTraces.ts`: the `AxesTraces` type and the `TRACE_FIELDS` table give, per field, its HDF5 `kind` names, whether it holds one trace or many, whether it makes the axes 3-D, and its contribution to the data extents. The reducer's empty axes and hold logic, `dataExtents`/`axesIs3D` in `axisLimits.ts`, and the HDF5 writer and reader all iterate this table, and the renderer table is typed over its 2-D fields, so a field missing from any of them is a compile error or a failing unit test (`src/__tests__/plot-trace-fields.test.ts`).

The dependency runs one way. The renderer in `src/graphics/` imports the types, the reducer, and the pure helpers from `src/numbl-core/plotting/`; numbl-core never imports from `src/graphics/`, and an ESLint `no-restricted-imports` rule enforces this.

## The plot viewer

The CLI's plot viewer (`src/plot-viewer/`) is a separate Vite-built bundle in `dist-plot-viewer/`. It receives `PlotInstruction` batches from the CLI over server-sent events, folds them into a `FiguresState` with `figuresReducer`, and renders the current figure with `FigureView`, the same component the web app uses. 2-D axes are static canvases with no interaction; 3-D axes can be rotated and zoomed with the mouse (Three.js `OrbitControls`, panning disabled).

Because the viewer is its own bundle, it does not rebuild with the rest of numbl-core. **After changing anything under `src/graphics/`, or a core plotting module it imports (`types.ts`, `axesTraces.ts`, `figuresReducer.ts`, `axisLimits.ts`, `imagescExtent.ts`, `colormaps.ts`), rebuild the viewer explicitly** — unit tests will not catch a stale bundle. The web app does not need this step, since it compiles the renderer from source.

## Adding a plot function

The common shape is:

1. Add the trace type and `PlotInstruction` variant in `plotting/types.ts`.
2. Parse the arguments and emit the instruction (`parseArgs*.ts`, `instructions.ts`), then add the name to `dispatch.ts` so it is registered. Keep the instruction serializable (no functions, no class instances).
3. Declare the trace field in `axesTraces.ts`, handle the new instruction in `figuresReducer.ts`, and add a renderer to `src/graphics/renderers/` (2-D) or `SurfView.tsx` (3-D).
4. Rebuild the viewer bundle.
5. Add tests — preferably integration scripts that exercise the command end-to-end — and an example in `examples/plots/`, which the `/gallery` page and its Playwright smoke test (`tests-browser/plot-gallery.spec.ts`) pick up.

See `docs/adding-plot-functions.md` for the specific checklist.

## Interactive HTML figures

`uihtml` is a different kind of figure: instead of axes/traces it renders a
self-contained HTML document in an iframe and exposes a two-way data/event
bridge to the interpreter (the standard MATLAB `uihtml` API). See
[uihtml.md](uihtml.md) for the architecture and
[../uihtml-figures.md](../uihtml-figures.md) for authoring guidance.

## Downloading figure data (HDF5)

The plot viewer and the IDE figure panel offer a "download data" control that
serializes a figure's `FigureState` to a self-describing HDF5 file (numeric data
as gzip-compressed datasets, styling/layout as attributes). The writer is
[../../src/graphics/exportFigureHdf5.ts](../../src/graphics/exportFigureHdf5.ts);
the on-disk schema is documented in
[../figure-hdf5-format.md](../figure-hdf5-format.md).

[h5wasm](https://github.com/usnistgov/h5wasm) is the in-browser HDF5 writer. It
is a single-file build (the wasm is embedded in the JS, so nothing extra needs
to be served) and is loaded **lazily** via dynamic `import()` — its ~1 MB
(gzipped) chunk is fetched only when a user actually downloads, never in the main
bundle.

The inverse, [../../src/graphics/importFigureHdf5.ts](../../src/graphics/importFigureHdf5.ts),
reads such a file back into a `FigureState`. The shared schema tables live in
[../../src/graphics/figureHdf5Schema.ts](../../src/graphics/figureHdf5Schema.ts)
so the writer and reader stay in sync; the kind-to-field mapping there is
derived from `TRACE_FIELDS` in `axesTraces.ts`.

### `numbl/graphics` package export

The figure renderer and the HDF5 import/export are published from a browser
subpath export so external apps can display numbl figures without numbl-core.
The entry is [../../src/graphics-lib.ts](../../src/graphics-lib.ts) (built by
`npm run build:graphics` to `dist-graphics/`), exposing `FigureView`,
`FigureState`/`AxesState`, `figuresReducer`, and `importFigureHdf5`/
`exportFigureHdf5`. The plot types and `figuresReducer` are re-exported from
`src/numbl-core/plotting/`; the bundle pulls in only those pure modules, not
the interpreter. `react`/`react-dom` are optional peer deps; `three` is
bundled; `h5wasm` stays lazy. Example consumer: the `numbl-figure-viewer` app
(`import { importFigureHdf5, FigureView } from "numbl/graphics"`).

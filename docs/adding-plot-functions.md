# Adding a New Plot Function

This guide walks through every file you need to touch when adding a new plot type (e.g., `bar`, `stem`, `area`). The `bar` implementation is used as a concrete example throughout.

## Architecture overview

The plotting system has two halves joined by a shared type:

```
Core (numbl-core/plotting/)             Renderer (graphics/)
───────────────────────────             ────────────────────
.m script calls bar(x,y)
  → builtins.ts registration            figuresReducer.ts (core)
      routes name "bar"                   AxesState holds barTraces[]
  → dispatch.ts                           (declared in axesTraces.ts)
      dispatchPlotBuiltin()               ↓
  → instructions.ts                     FigureView.tsx
      barCall()                           passes the AxesState through
  → parseArgsChart.ts                     ↓
      parseBarArgs()                    drawPlot.ts + renderers/bar.ts
  → pushes PlotInstruction ──────────→    canvas 2D rendering
```

The **`PlotInstruction`** type (defined in `src/numbl-core/plotting/types.ts`) is the contract between the two halves. The core produces instructions; the renderer consumes them. The dependency runs one way: `src/graphics/` imports the plot types and the pure helpers it shares with the runtime (`figuresReducer.ts`, `axesTraces.ts`, `axisLimits.ts`, `imagescExtent.ts`, `colormaps.ts`) from `src/numbl-core/plotting/`, and numbl-core never imports from `src/graphics/` (an ESLint rule enforces this).

### 2D vs 3D

- **2D plot types** (plot, bar, scatter, imagesc, contour, ...) render via `drawPlot.ts` using an HTML Canvas 2D context. `drawPlot` owns what all 2-D kinds share (data limits, axis modes and scales, grid, ticks, labels, legend, colorbar); each kind's drawing is a renderer in `src/graphics/renderers/`, registered in the table in `renderers/index.ts`.
- **3D plot types** (surf, mesh, plot3, bar3, quiver3) render via `SurfView.tsx` using Three.js.

An axes is drawn by `SurfView` when it holds any 3-D content (`axesIs3D` in `axisLimits.ts`), and by `drawPlot` otherwise. This guide focuses on 2D since it's more common.

### Key data conventions

- All colors are RGB triples with values in `[0, 1]`, e.g., `[1, 0, 0]` for red.
- Tensors use **column-major** (Fortran) storage: element `(i, j)` of an `[m, n]` matrix is at `data[j * m + i]`.
- The `RuntimeValue` type represents any value in the numbl runtime (numbers, tensors, strings, etc.). Plot argument parsers receive `RuntimeValue[]` and must convert to plain JS arrays.

## Files to edit (in order)

| Step | File                                        | What to add                                                     |
| ---- | ------------------------------------------- | --------------------------------------------------------------- |
| 1    | `src/numbl-core/plotting/types.ts`          | Trace interface + `PlotInstruction` variant                     |
| 2    | `src/numbl-core/plotting/parseArgs*.ts`     | Argument parser (`parseXxxArgs`)                                |
| 3    | `src/numbl-core/plotting/instructions.ts`   | Call function (`xxxCall`)                                       |
| 4    | `src/numbl-core/plotting/dispatch.ts`       | Import `*Call`, add `switch` case, add to `PLOT_DISPATCH_NAMES` |
| 5    | `src/numbl-core/plotting/axesTraces.ts`     | Field in `AxesTraces` + entry in `TRACE_FIELDS`                 |
| 6    | `src/numbl-core/plotting/figuresReducer.ts` | Reducer case                                                    |
| 7    | `src/graphics/renderers/*.ts`               | 2-D renderer, registered in `renderers/index.ts`                |

The entry in step 5 is the only place the new field is declared. The empty axes, the hold logic, the axis limits (`lim = axis` and the drawn view), the 2-D/3-D routing and the figure HDF5 export and import are all derived from it, and TypeScript rejects a field without an entry, or a 2-D field without a renderer. `FigureView.tsx` passes the whole `AxesState` to `drawPlot`, so it needs no change. For a 3-D kind, step 7 is instead adding the field to `TraceField3D` in `axesTraces.ts` and the drawing code to `SurfView.tsx` (plus its props in `FigureView.tsx`).

## Step-by-step

### 1. Define the trace type — `src/numbl-core/plotting/types.ts`

This file is the single source of truth for all trace interfaces and the `PlotInstruction` union. Add your trace interface before the `PlotInstruction` definition, following the existing pattern (each trace type has its own section with a comment header).

**Add the trace interface** (between the last trace interface and `// ── Plot Instructions`):

```ts
// ── BarTrace type ───────────────────────────────────────────────────────

export interface BarTrace {
  /** X positions for each bar */
  x: number[];
  /** Y values (bar heights) for each bar */
  y: number[];
  /** Relative bar width (0–1, default 0.8) */
  width: number;
  /** Bar color as RGB triple [0–1] */
  color?: [number, number, number];
}
```

Design your interface to contain only the data needed for rendering — no runtime types, no optional parse state. Keep it minimal; you can always add fields later.

**Add a variant to the `PlotInstruction` union** (among the other trace-producing instructions):

```ts
export type PlotInstruction =
  | { type: "set_figure_handle"; handle: number }
  | { type: "plot"; traces: PlotTrace[] }
  | { type: "plot3"; traces: Plot3Trace[] }
  | { type: "surf"; trace: SurfTrace }
  | { type: "imagesc"; trace: ImagescTrace }
  | { type: "contour"; trace: ContourTrace }
  | { type: "mesh"; trace: SurfTrace }
  | { type: "bar"; traces: BarTrace[] } // ← add here
  | { type: "set_hold"; value: boolean };
// ... rest of union
```

Note: some instructions use `trace` (singular) and some use `traces` (array). Use `traces` if the function could produce multiple series in a single call (like `plot(x1,y1,x2,y2)`). Use `trace` if a single call always produces one trace (like `surf`).

### 2. Argument parser — `src/numbl-core/plotting/parseArgs*.ts`

Parsers are grouped by plot family:

| File                  | Parsers                                                                |
| --------------------- | ---------------------------------------------------------------------- |
| `parseArgsLine.ts`    | plot, plot3, line, scatter, scatter3, stairs, errorbar, area           |
| `parseArgsPatch.ts`   | patch, fill, trimesh, trisurf                                          |
| `parseArgsSurface.ts` | surf, mesh, pcolor, imagesc, contour, heatmap                          |
| `parseArgsChart.ts`   | bar, barh, bar3, histogram, histogram2, boxchart, swarmchart, piechart |
| `parseArgsQuiver.ts`  | quiver, quiver3                                                        |
| `parseArgsCommon.ts`  | Shared helpers (colors, LineSpec, Name-Value keys, numeric conversion) |

Each parser converts `RuntimeValue[]` into the trace type(s) defined in step 1. Add yours to the file for its family (here `parseArgsChart.ts`), importing the trace type from `./types.js` and any helpers from `./parseArgsCommon.js`:

```ts
import type { BarTrace } from "./types.js";
import {
  getStringValue,
  isNumericArg,
  isStringArg,
  oneBasedIndices,
  resolveColor,
  toNumberArray,
} from "./parseArgsCommon.js";
```

**Write the parser function** at the end of the file. The parser's job is to interpret the various argument patterns the function supports. For `bar`:

```ts
/**
 * Parse bar() arguments.
 *
 * Supported forms:
 *   bar(Y)             — Y values with x = 1:length(Y)
 *   bar(X, Y)          — explicit X positions
 *   bar(..., width)     — relative bar width (scalar 0–1)
 *   bar(..., color)     — single-char color spec
 */
export function parseBarArgs(args: RuntimeValue[]): BarTrace[] {
  let pos = 0;
  let xData: number[] | undefined;
  let yData: number[];
  let width = 0.8;
  let color: [number, number, number] | undefined;

  if (args.length === 0) throw new Error("bar requires at least 1 argument");

  const first = args[pos++];

  if (pos < args.length && isNumericArg(args[pos])) {
    // bar(X, Y, ...)
    xData = toNumberArray(first);
    yData = toNumberArray(args[pos++]);
  } else {
    // bar(Y)
    yData = toNumberArray(first);
  }

  // Check for optional width (scalar) or color (string)
  while (pos < args.length) {
    if (isNumericArg(args[pos])) {
      const v = args[pos];
      if (isRuntimeNumber(v)) {
        width = v as number;
        pos++;
      } else {
        break;
      }
    } else if (isStringArg(args[pos])) {
      const s = getStringValue(args[pos]);
      const c = resolveColor(s);
      if (c) {
        color = c;
        pos++;
      } else {
        break;
      }
    } else {
      break;
    }
  }

  if (!xData) {
    xData = oneBasedIndices(yData.length);
  }

  const trace: BarTrace = { x: xData, y: yData, width };
  if (color) trace.color = color;
  return [trace];
}
```

**Available helper functions** (exported from `parseArgsCommon.ts`):

| Helper                                                                         | Purpose                                                                            |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `toNumberArray(v: RuntimeValue): number[]`                                     | Convert any runtime value to a flat number array                                   |
| `isNumericArg(v: unknown): boolean`                                            | Check if a value is numeric (number, tensor, or logical)                           |
| `isStringArg(v: unknown): boolean`                                             | Check if a value is a string or char                                               |
| `getStringValue(v: RuntimeValue): string`                                      | Extract the JS string from a string/char runtime value                             |
| `resolveColor(v: RuntimeValue \| string): [number,number,number] \| undefined` | Parse a color spec (short name like `'r'`, color name like `'red'`, or RGB tensor) |
| `oneBasedIndices(n: number): number[]`                                         | Generate `[1, 2, ..., n]` for implicit x-axis                                      |
| `isRuntimeNumber(v)`, `isRuntimeTensor(v)`, etc.                               | Type guards for runtime values                                                     |
| `toNumber(v: RuntimeValue): number`                                            | Extract a scalar number                                                            |
| `expandColumns(v: RuntimeValue): number[][]`                                   | Split a matrix into column arrays (for multi-series plots)                         |
| `getMatrixInfo(v: RuntimeValue): { data, rows, cols }`                         | Get flat data + dimensions from a matrix                                           |
| `expandXY(xVal, yVal, rows, cols): { x, y }`                                   | Meshgrid-expand X/Y vectors to match a grid                                        |
| `parseLineSpec(s: string): ParsedLineSpec \| null`                             | Parse a LineSpec string like `'r--o'` into color/lineStyle/marker                  |
| `isNameValueKey(v: RuntimeValue): string \| null`                              | Check if a value is a known Name-Value key like `'Color'`                          |

### 3. Call function — `src/numbl-core/plotting/instructions.ts`

This file bridges the argument parsers and the `PlotInstruction` array. Each plot function has a corresponding exported function here.

**Add import** of your parser (in the import block for its `parseArgs*.js` file):

```ts
import {
  parseBarArgs, // ← add
  parseBarHArgs,
  parseBar3Args,
  // ...
} from "./parseArgsChart.js";
```

**Add the call function** (near the other `*Call` functions, before `legendCall`):

```ts
export function barCall(
  plotInstructions: PlotInstruction[],
  args: RuntimeValue[]
): void {
  const traces = parseBarArgs(args);
  if (traces.length > 0) {
    plotInstructions.push({ type: "bar", traces });
  }
}
```

This pattern is the same for every plot function — parse args, push instruction. The first parameter is always the `plotInstructions` array (accumulated during script execution), and the second is the raw runtime arguments.

### 4. Dispatch — `src/numbl-core/plotting/dispatch.ts`

`dispatchPlotBuiltin` is the single source of truth for the `(name, args) → PlotInstruction[]` mapping. Every data-bearing plot builtin is a `case` in its `switch (name)` that calls the matching `*Call` function and returns `true`.

**4a. Import** your call function (in the import block from `./instructions.js`):

```ts
import {
  plotCall,
  // ... existing ...
  barCall, // ← add
} from "./instructions.js";
```

**4b. Add a `case`** to the `switch (name)` inside `dispatchPlotBuiltin`:

```ts
case "bar":
  barCall(instructions, args);
  return true;
```

`instructions` is the `PlotInstruction[]` passed in by the caller (it is `rt.plotInstructions` in the live runtime); `args` is already `RuntimeValue[]`, so no `ensureRuntimeValue` mapping is needed here.

**4c. Register the name** in the `PLOT_DISPATCH_NAMES` array (lower in the same file):

```ts
export const PLOT_DISPATCH_NAMES: ReadonlyArray<string> = [
  // ... existing ...
  "bar", // ← add
  // ...
];
```

Adding the name to `PLOT_DISPATCH_NAMES` is enough to register it: `src/numbl-core/plotting/builtins.ts` registers every name in that list as a builtin that calls `dispatchPlotBuiltin`, which is what makes `bar(...)` resolve from interpreted `.m` code. Only functions that need a custom return value or `nargout` handle (e.g. `pcolor`, `fplot`) get an explicit registration in that file instead.

### 5. Declare the trace field — `src/numbl-core/plotting/axesTraces.ts`

`AxesState` (in `figuresReducer.ts`) is `AxesTraces` plus labels and appearance settings. Add the field that holds your traces to `AxesTraces`, importing the trace type from `./types.js`:

```ts
export type AxesTraces = {
  traces: PlotTrace[];
  // ...
  barTraces: BarTrace[]; // ← add
  // ...
};
```

Then add its entry to `TRACE_FIELDS` (the compiler reports it as missing). The entry's position is the order traces are written to the HDF5 file:

```ts
barTraces: {
  // `kind` names in the figure HDF5 layout: the first is written on export,
  // all are accepted on import.
  kinds: ["bar"],
  // "many" for an array field, "one" for a single trace per axes.
  multiplicity: "many",
  // Whether non-empty content makes the axes 3-D (drawn by SurfView).
  is3D: () => false,
  // Grow the data extents (used for `lim = axis` and the drawn limits).
  extents(bars, { x, y }) {
    for (const bt of bars) {
      const hw = bt.width / 2;
      for (let i = 0; i < bt.x.length; i++) {
        include(x, bt.x[i] - hw);
        include(x, bt.x[i] + hw);
        include(y, bt.y[i]);
      }
      include(y, 0); // bars always extend to zero
    }
  },
},
```

`include`/`includeAll` ignore non-finite values. If the field is drawn only in 3-D, also add it to the `TraceField3D` union. If its traces have flat column-major grid fields (`rows × cols`), list them under the kind in `GRID_FIELDS` in `src/graphics/figureHdf5Schema.ts` so they are stored as 2-D datasets, and document the kind in `docs/figure-hdf5-format.md`.

### 6. Figures reducer — `src/numbl-core/plotting/figuresReducer.ts`

The reducer turns each `PlotInstruction` into a new immutable `FiguresState`. It lives in the core because the runtime also uses it (to answer `lim = axis`); the renderer imports it from there. Add a case for the new instruction. `addTraces` applies the hold logic: with hold off, every trace field it is not given is cleared.

```ts
case "bar": {
  const axes = getAxes(ensureFig(state));
  return addTraces(state, {
    barTraces: axes.holdOn
      ? [...axes.barTraces, ...action.traces]
      : [...action.traces],
  });
}
```

### 7. Canvas renderer — `src/graphics/renderers/`

Write the drawing code in a new file (or next to a related kind, as `bar.ts` holds both `bar` and `barh`). A renderer receives the field's non-empty value; `drawPlot` has already computed the limits, drawn the axes, and clipped the canvas to the plot area:

```ts
import type { BarTrace } from "../../numbl-core/plotting/types.js";
import type { DataRenderer } from "./types.js";

export const barRenderer: DataRenderer<"barTraces"> = {
  layer: "data",
  draw({ ctx }, bars, { toCanvasX, toCanvasY }) {
    // ctx: CanvasRenderingContext2D (CSS-pixel coordinates)
    // toCanvasX/toCanvasY: data → canvas coordinates (handle log axes, axis ij)
    // The first argument also carries `axes` (the whole AxesState), `cmap`
    // (the colormap) and `dpr`.
  },
};
```

Register it in `CANVAS_RENDERERS` in `renderers/index.ts`. The table's order is the drawing order, so later entries paint over earlier ones (images and filled regions first, lines last). Two optional parts:

- `colorValues(value, consume)`: for kinds colored through the colormap, feed the scalar values that should set the automatic colorbar range.
- `layer: "standalone"` (with `draw(rc, value, width, height)`) instead of `"data"`: the kind takes over the whole canvas with no shared axes, as the pie chart and heatmap do.

## Testing

After implementation:

1. **Type check**: `npm run typecheck` (runs `tsc -b`) — must have zero errors
2. **Unit + integration tests**: `npm test` and `npm run test:scripts`; prefer integration scripts in `numbl_test_scripts/` that exercise the command end-to-end
3. **Rebuild the viewer**: `npm run build:plot-viewer` (the renderer bundle also includes the core modules it imports)
4. **Visual test**: create an example script in `examples/plots/` and run with:
   ```bash
   npx tsx src/cli.ts run examples/plots/bar_example.m --plot
   ```
   This opens a browser window showing the plot. Kill the process with Ctrl+C when done. The example also appears on the web app's `/gallery` page.
5. **Browser smoke test**: `npm run build`, then `npx playwright test tests-browser/plot-gallery.spec.ts`. It renders every gallery example and fails on script errors, console errors, or blank canvases. With `PLOT_GALLERY_HASHES=<file>` it also writes a hash of each 2-D canvas's pixels; comparing the files from before and after a change (on the same machine) shows whether any rendered output changed.

## Existing plot functions (for reference)

| Function    | Trace type     | Instruction type | `AxesState` field | Renderer                               |
| ----------- | -------------- | ---------------- | ----------------- | -------------------------------------- |
| `plot`      | `PlotTrace[]`  | `"plot"`         | `traces`          | `renderers/lines.ts`                   |
| `scatter`   | `PlotTrace[]`  | `"plot"`         | `traces`          | `renderers/lines.ts` (marker-only)     |
| `plot3`     | `Plot3Trace[]` | `"plot3"`        | `plot3Traces`     | `SurfView.tsx` (3D lines)              |
| `surf`      | `SurfTrace`    | `"surf"`         | `surfTraces`      | `SurfView.tsx` (3D surface)            |
| `mesh`      | `SurfTrace`    | `"mesh"`         | `surfTraces`      | `SurfView.tsx` (3D wireframe)          |
| `waterfall` | `SurfTrace`    | `"mesh"`         | `surfTraces`      | `SurfView.tsx` (aliased to mesh)       |
| `imagesc`   | `ImagescTrace` | `"imagesc"`      | `imagescTrace`    | `renderers/imagesc.ts` (pixel grid)    |
| `contour`   | `ContourTrace` | `"contour"`      | `contourTraces`   | `renderers/contour.ts` (line contours) |
| `contourf`  | `ContourTrace` | `"contour"`      | `contourTraces`   | `renderers/contour.ts` (filled cells)  |
| `bar`       | `BarTrace[]`   | `"bar"`          | `barTraces`       | `renderers/bar.ts` (filled rectangles) |

Note: `scatter` reuses `PlotTrace` with `lineStyle: "none"` and a marker set. `waterfall` is aliased to `mesh`. When possible, reuse an existing trace type rather than creating a new one.

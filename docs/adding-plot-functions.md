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
  → dispatch.ts                           ↓
      dispatchPlotBuiltin()             FigureView.tsx
  → instructions.ts                       passes traces to renderer
      barCall()                           ↓
  → parseArgsChart.ts                   drawPlot.ts
      parseBarArgs()                      canvas 2D rendering
  → pushes PlotInstruction ──────────→
```

The **`PlotInstruction`** type (defined in `src/numbl-core/plotting/types.ts`) is the contract between the two halves. The core produces instructions; the renderer consumes them. The dependency runs one way: `src/graphics/` imports the plot types and the pure helpers it shares with the runtime (`figuresReducer.ts`, `axisLimits.ts`, `imagescExtent.ts`, `colormaps.ts`) from `src/numbl-core/plotting/`, and numbl-core never imports from `src/graphics/` (an ESLint rule enforces this).

### 2D vs 3D

- **2D plot types** (plot, bar, scatter, imagesc, contour) render via `drawPlot.ts` using an HTML Canvas 2D context.
- **3D plot types** (surf, mesh, plot3) render via `SurfView.tsx` using Three.js.

If you're adding a 2D plot type, you'll work with `drawPlot.ts`. If 3D, you'll work with `SurfView.tsx`. This guide focuses on 2D since it's more common.

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
| 5    | `src/numbl-core/plotting/figuresReducer.ts` | `AxesState`, `defaultAxes`, `addTraces`, reducer case           |
| 6    | `src/graphics/FigureView.tsx`               | Pass new traces through to the drawing function                 |
| 7    | `src/graphics/drawPlot.ts`                  | Rendering logic (data bounds + canvas drawing)                  |

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

### 5. Figures reducer — `src/numbl-core/plotting/figuresReducer.ts`

This file manages the state of all figures. It's a React-style reducer: each `PlotInstruction` produces a new immutable state. It lives in the core because the runtime also uses it (to answer `lim = axis`); the renderer imports it from there. There are four places to edit.

**5a. Import** the trace type:

```ts
import type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
  ImagescTrace,
  ContourTrace,
  BarTrace, // ← add
  PlotInstruction,
} from "./types.js";
```

**5b. AxesState** — add the field to hold your traces:

```ts
export type AxesState = {
  holdOn: boolean;
  traces: PlotTrace[];
  plot3Traces: Plot3Trace[];
  surfTraces: SurfTrace[];
  imagescTrace?: ImagescTrace;
  contourTraces: ContourTrace[];
  barTraces: BarTrace[]; // ← add
  title?: string;
  // ...
};
```

**5c. defaultAxes** — initialize to empty:

```ts
const defaultAxes: AxesState = {
  holdOn: false,
  traces: [],
  plot3Traces: [],
  surfTraces: [],
  contourTraces: [],
  barTraces: [], // ← add
};
```

**5d. addTraces** — this function handles hold-state logic (clearing other trace types when hold is off). Add your field to both the `Pick` type parameter and the object spread:

```ts
function addTraces(
  state: FiguresState,
  update: Partial<
    Pick<
      AxesState,
      | "traces"
      | "plot3Traces"
      | "surfTraces"
      | "imagescTrace"
      | "contourTraces"
      | "barTraces" // ← add to Pick
    >
  >
): FiguresState {
  // ...
  return {
    // ...
    contourTraces: update.contourTraces ?? (hold ? axes.contourTraces : []),
    barTraces: update.barTraces ?? (hold ? axes.barTraces : []), // ← add to spread
    // ...
  };
}
```

**5e. Reducer switch** — add a case to `figuresReducer`:

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

The hold-on logic: when `hold on` is active, new traces are appended to existing ones. When off (default), they replace.

### 6. FigureView — `src/graphics/FigureView.tsx`

This React component connects the state to the renderer. The `SingleAxesView` component decides whether to use 2D or 3D rendering, then `PlotCanvas` calls `drawPlot()`.

There are four changes — all are about threading the new data through:

**6a. PlotCanvas JSX props** (in `SingleAxesView`, add the prop):

```tsx
<PlotCanvas
  traces={axes.traces}
  // ... existing props ...
  axisMode={axes.axisMode}
  barTraces={axes.barTraces} // ← add
/>
```

**6b. PlotCanvas component props type** (add to the destructured props and type annotation):

```tsx
function PlotCanvas({
  traces,
  // ... existing ...
  axisMode,
  barTraces,                         // ← add
}: {
  traces: PlotTrace[];
  // ... existing ...
  axisMode?: string;
  barTraces?: AxesState["barTraces"]; // ← add
}) {
```

**6c. drawPlot() call** (inside the `useCallback`):

```ts
drawPlot(
  canvas,
  traces,
  title,
  xlabel,
  ylabel,
  legend,
  gridOn,
  imagescTrace,
  contourTraces,
  colormap,
  axisMode,
  barTraces // ← add
);
```

**6d. useCallback dependency array**:

```ts
}, [traces, title, xlabel, ylabel, legend, gridOn,
    imagescTrace, contourTraces, colormap, axisMode,
    barTraces]);      // ← add
```

### 7. Canvas rendering — `src/graphics/drawPlot.ts`

This is where you write the actual drawing code. The `drawPlot` function receives all trace data and renders to a `<canvas>` element using the Canvas 2D API.

**7a. Import** the trace type:

```ts
import type {
  PlotTrace,
  ImagescTrace,
  ContourTrace,
  BarTrace,
} from "../numbl-core/plotting/types.js";
```

**7b. Function signature** — add the parameter at the end:

```ts
export function drawPlot(
  canvas: HTMLCanvasElement,
  traces: PlotTrace[],
  title?: string,
  xlabel?: string,
  ylabel?: string,
  legend?: string[],
  gridOn?: boolean,
  imagescTrace?: ImagescTrace,
  contourTraces?: ContourTrace[],
  colormap?: string,
  axisMode?: string,
  barTraces?: BarTrace[]            // ← add
) {
```

**7c. hasContent check** — the function exits early if there's nothing to draw:

```ts
const hasContent =
  traces.length > 0 ||
  imagescTrace !== undefined ||
  (contourTraces && contourTraces.length > 0) ||
  (barTraces && barTraces.length > 0); // ← add
```

**7d. Data bounds** — the function computes `xMin/xMax/yMin/yMax` across all traces to set up the coordinate system. Add a block for your trace type. This goes after the existing contour bounds block and before the `if (!isFinite(xMin)) return;` check:

```ts
// Include bar bounds
if (barTraces) {
  for (const bt of barTraces) {
    const halfW = bt.width / 2;
    for (let i = 0; i < bt.x.length; i++) {
      const bx = bt.x[i];
      if (isFinite(bx)) {
        if (bx - halfW < xMin) xMin = bx - halfW;
        if (bx + halfW > xMax) xMax = bx + halfW;
      }
      const by = bt.y[i];
      if (isFinite(by)) {
        if (by < yMin) yMin = by;
        if (by > yMax) yMax = by;
      }
    }
    // Bars always extend to zero on the y-axis
    if (0 < yMin) yMin = 0;
    if (0 > yMax) yMax = 0;
  }
}
```

**7e. Drawing** — add rendering code inside the clipped region. The clip region is set up earlier with `ctx.clip()`. Your drawing code goes after contour rendering and before the line-trace loop. You have access to:

- `ctx` — the `CanvasRenderingContext2D`
- `toCanvasX(v)` / `toCanvasY(v)` — convert data coordinates to canvas pixel coordinates
- `effMarginLeft`, `effMarginTop`, `effPlotW`, `effPlotH` — the plot area bounds in pixels

```ts
// Bar rendering
if (barTraces) {
  const defaultColors: [number, number, number][] = [
    [0.0, 0.447, 0.741],
    [0.85, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
  ];
  for (let bi = 0; bi < barTraces.length; bi++) {
    const bt = barTraces[bi];
    const [r, g, b] = bt.color ?? defaultColors[bi % defaultColors.length];
    ctx.fillStyle = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    ctx.strokeStyle = `rgb(${Math.round(r * 180)},${Math.round(g * 180)},${Math.round(b * 180)})`;
    ctx.lineWidth = 1;
    const y0 = toCanvasY(0);
    for (let i = 0; i < bt.x.length; i++) {
      const cx = toCanvasX(bt.x[i]);
      const cy = toCanvasY(bt.y[i]);
      const halfW = (toCanvasX(bt.x[i] + bt.width) - toCanvasX(bt.x[i])) / 2;
      ctx.fillRect(cx - halfW, Math.min(cy, y0), halfW * 2, Math.abs(cy - y0));
      ctx.strokeRect(
        cx - halfW,
        Math.min(cy, y0),
        halfW * 2,
        Math.abs(cy - y0)
      );
    }
  }
}
```

The default color palette matches MATLAB's default bar colors. Bars extend from the baseline (y=0) to the data value, correctly handling negative values with `Math.min`/`Math.abs`.

## Testing

After implementation:

1. **Type check**: `npx tsc --noEmit` — must have zero errors
2. **Unit + integration tests**: `npm test` — all existing tests must still pass
3. **Rebuild the viewer**: `npm run build:plot-viewer` (the renderer bundle also includes the core modules it imports)
4. **Visual test**: create an example script in `examples/plots/` and run with:
   ```bash
   npx tsx src/cli.ts run examples/plots/bar_example.m --plot
   ```
   This opens a browser window showing the plot. Kill the process with Ctrl+C when done.

## Existing plot functions (for reference)

| Function    | Trace type     | Instruction type | Renderer                           |
| ----------- | -------------- | ---------------- | ---------------------------------- |
| `plot`      | `PlotTrace[]`  | `"plot"`         | `drawPlot.ts` (line traces)        |
| `scatter`   | `PlotTrace[]`  | `"plot"`         | `drawPlot.ts` (marker-only traces) |
| `plot3`     | `Plot3Trace[]` | `"plot3"`        | `SurfView.tsx` (3D lines)          |
| `surf`      | `SurfTrace`    | `"surf"`         | `SurfView.tsx` (3D surface)        |
| `mesh`      | `SurfTrace`    | `"mesh"`         | `SurfView.tsx` (3D wireframe)      |
| `waterfall` | `SurfTrace`    | `"mesh"`         | `SurfView.tsx` (aliased to mesh)   |
| `imagesc`   | `ImagescTrace` | `"imagesc"`      | `drawPlot.ts` (pixel grid)         |
| `contour`   | `ContourTrace` | `"contour"`      | `drawPlot.ts` (marching squares)   |
| `contourf`  | `ContourTrace` | `"contour"`      | `drawPlot.ts` (filled cells)       |
| `bar`       | `BarTrace[]`   | `"bar"`          | `drawPlot.ts` (filled rectangles)  |

Note: `scatter` reuses `PlotTrace` with `lineStyle: "none"` and a marker set. `waterfall` is aliased to `mesh`. When possible, reuse an existing trace type rather than creating a new one.

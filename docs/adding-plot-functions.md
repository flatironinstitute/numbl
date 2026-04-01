# Adding a New Plot Function

This guide walks through every file you need to touch when adding a new plot type (e.g., `bar`, `stem`, `area`). The `bar` implementation is used as a concrete example throughout.

## Architecture overview

The plotting system has two halves joined by a shared type:

```
Runtime (numbl-core/)                   Graphics (graphics/)
─────────────────────                   ─────────────────────
.m script calls bar(x,y)
  → runtimeDispatch.ts                  figuresReducer.ts
      routes name "bar"                   AxesState holds barTraces[]
  → runtime.ts                            ↓
      bar_call() method                 FigureView.tsx
  → runtimePlot.ts                        passes traces to renderer
      barCall() parses args               ↓
  → pushes PlotInstruction ──────────→  drawPlot.ts
                                          canvas 2D rendering
```

The **`PlotInstruction`** type (defined in `src/graphics/types.ts`) is the contract between the two halves. The runtime produces instructions; the graphics layer consumes them. This is the only shared dependency.

### 2D vs 3D

- **2D plot types** (plot, bar, scatter, imagesc, contour) render via `drawPlot.ts` using an HTML Canvas 2D context.
- **3D plot types** (surf, mesh, plot3) render via `SurfView.tsx` using Three.js.

If you're adding a 2D plot type, you'll work with `drawPlot.ts`. If 3D, you'll work with `SurfView.tsx`. This guide focuses on 2D since it's more common.

### Key data conventions

- All colors are RGB triples with values in `[0, 1]`, e.g., `[1, 0, 0]` for red.
- Tensors use **column-major** (Fortran) storage: element `(i, j)` of an `[m, n]` matrix is at `data[j * m + i]`.
- The `RuntimeValue` type represents any value in the numbl runtime (numbers, tensors, strings, etc.). Plot argument parsers receive `RuntimeValue[]` and must convert to plain JS arrays.

## Files to edit (in order)

| Step | File                                            | What to add                                           |
| ---- | ----------------------------------------------- | ----------------------------------------------------- |
| 1    | `src/graphics/types.ts`                         | Trace interface + `PlotInstruction` variant           |
| 2    | `src/numbl-core/runtime/plotUtils.ts`           | Argument parser (`parseXxxArgs`)                      |
| 3    | `src/numbl-core/runtime/runtimePlot.ts`         | Call function (`xxxCall`)                             |
| 4    | `src/numbl-core/runtime/runtimeDispatch.ts`     | Dispatch case in `dispatchPlotCall`                   |
| 5    | `src/numbl-core/runtime/runtime.ts`             | Import, public method, builtin registration           |
| 6    | `src/numbl-core/runtime/specialBuiltinNames.ts` | Add function name to the array                        |
| 7    | `src/graphics/figuresReducer.ts`                | `AxesState`, `defaultAxes`, `addTraces`, reducer case |
| 8    | `src/graphics/FigureView.tsx`                   | Pass new traces through to the drawing function       |
| 9    | `src/graphics/drawPlot.ts`                      | Rendering logic (data bounds + canvas drawing)        |

## Step-by-step

### 1. Define the trace type — `src/graphics/types.ts`

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

### 2. Argument parser — `src/numbl-core/runtime/plotUtils.ts`

This file contains parsers for all plot functions. Each parser converts `RuntimeValue[]` into the trace type(s) defined in step 1.

**Add to the re-export and import blocks** at the top of the file. There are two blocks that must stay in sync:

```ts
// Re-export block (~line 20)
export type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
  ImagescTrace,
  ContourTrace,
  BarTrace, // ← add
} from "../../graphics/types.js";

// Import block (~line 28)
import type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
  ImagescTrace,
  ContourTrace,
  BarTrace, // ← add
} from "../../graphics/types.js";
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

**Available helper functions** (all defined earlier in plotUtils.ts):

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

### 3. Call function — `src/numbl-core/runtime/runtimePlot.ts`

This file bridges the argument parsers and the `PlotInstruction` array. Each plot function has a corresponding exported function here.

**Add import** of your parser (in the import block from `plotUtils.js`):

```ts
import {
  parsePlotArgs,
  parsePlot3Args,
  parseSurfArgs,
  parseScatterArgs,
  parseImagescArgs,
  parseContourArgs,
  parseMeshArgs,
  parseBarArgs, // ← add
} from "../runtime/plotUtils.js";
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

### 4. Dispatch — `src/numbl-core/runtime/runtimeDispatch.ts`

This file routes function names to the appropriate `*_call` method on the Runtime object. There is a `dispatchPlotCall` function with a `switch` statement.

**Add a case** in the `switch (name)` block inside `dispatchPlotCall`:

```ts
function dispatchPlotCall(
  rt: Runtime,
  name: string,
  args: unknown[]
): unknown | undefined {
  switch (name) {
    case "plot":
      return rt.plot_call(args.map(a => ensureRuntimeValue(a)));
    // ... existing cases ...
    case "mesh":
    case "waterfall":
      return rt.mesh_call(args.map(a => ensureRuntimeValue(a)));
    case "bar": // ← add
      return rt.bar_call(args.map(a => ensureRuntimeValue(a))); // ← add
    default:
      return undefined;
  }
}
```

Note: `args.map(a => ensureRuntimeValue(a))` converts raw `unknown[]` to `RuntimeValue[]`. This is required because the dispatch layer receives untyped args.

### 5. Runtime registration — `src/numbl-core/runtime/runtime.ts`

Three changes in this file. The `Runtime` class is large; search for the existing plot-related code to find insertion points.

**5a. Import** — add to the import block from `"./runtimePlot.js"` (~line 97):

```ts
import {
  plotInstr as _plotInstr,
  plotCall as _plotCall,
  plot3Call as _plot3Call,
  surfCall as _surfCall,
  scatterCall as _scatterCall,
  imagescCall as _imagescCall,
  contourCall as _contourCall,
  meshCall as _meshCall,
  barCall as _barCall, // ← add
  viewCall as _viewCall,
  legendCall as _legendCall,
  drawnow as _drawnow,
  pause as _pause,
} from "./runtimePlot.js";
```

**5b. Public method** — add alongside the other `*_call` methods (search for `public mesh_call`):

```ts
public bar_call(args: RuntimeValue[]): void {
  _barCall(this.plotInstructions, args);
}
```

**5c. Builtin registration** — add in the `initBuiltins()` method, after the other plot builtin registrations (search for `this.builtins["waterfall"]`):

```ts
this.builtins["bar"] = (_nargout: number, args: unknown[]) => {
  this.bar_call(args.map(a => ensureRuntimeValue(a)));
};
```

The builtin registration is what makes the function callable from .m scripts. Without this, `bar(...)` would produce an "undefined function" error.

### 6. Special builtin name — `src/numbl-core/runtime/specialBuiltinNames.ts`

This file lists function names that bypass normal workspace/function resolution and go directly to the runtime. All plot functions must be listed here.

**Add the name** to the `SPECIAL_BUILTIN_NAMES` array, near the other plot names:

```ts
export const SPECIAL_BUILTIN_NAMES: readonly string[] = [
  // ... other names ...
  "mesh",
  "waterfall",
  "bar", // ← add
  "colormap",
  // ... rest ...
];
```

If you forget this step, the function will only work when called directly; it will fail inside user-defined functions or when the JIT compiler is active.

### 7. Figures reducer — `src/graphics/figuresReducer.ts`

This file manages the state of all figures. It's a React-style reducer: each `PlotInstruction` produces a new immutable state. There are four places to edit.

**7a. Import** the trace type:

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

**7b. AxesState** — add the field to hold your traces:

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

**7c. defaultAxes** — initialize to empty:

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

**7d. addTraces** — this function handles hold-state logic (clearing other trace types when hold is off). Add your field to both the `Pick` type parameter and the object spread:

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

**7e. Reducer switch** — add a case to `figuresReducer`:

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

### 8. FigureView — `src/graphics/FigureView.tsx`

This React component connects the state to the renderer. The `SingleAxesView` component decides whether to use 2D or 3D rendering, then `PlotCanvas` calls `drawPlot()`.

There are four changes — all are about threading the new data through:

**8a. PlotCanvas JSX props** (in `SingleAxesView`, add the prop):

```tsx
<PlotCanvas
  traces={axes.traces}
  // ... existing props ...
  axisMode={axes.axisMode}
  barTraces={axes.barTraces} // ← add
/>
```

**8b. PlotCanvas component props type** (add to the destructured props and type annotation):

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

**8c. drawPlot() call** (inside the `useCallback`):

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

**8d. useCallback dependency array**:

```ts
}, [traces, title, xlabel, ylabel, legend, gridOn,
    imagescTrace, contourTraces, colormap, axisMode,
    barTraces]);      // ← add
```

### 9. Canvas rendering — `src/graphics/drawPlot.ts`

This is where you write the actual drawing code. The `drawPlot` function receives all trace data and renders to a `<canvas>` element using the Canvas 2D API.

**9a. Import** the trace type:

```ts
import type {
  PlotTrace,
  ImagescTrace,
  ContourTrace,
  BarTrace,
} from "./types.js";
```

**9b. Function signature** — add the parameter at the end:

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

**9c. hasContent check** — the function exits early if there's nothing to draw:

```ts
const hasContent =
  traces.length > 0 ||
  imagescTrace !== undefined ||
  (contourTraces && contourTraces.length > 0) ||
  (barTraces && barTraces.length > 0); // ← add
```

**9d. Data bounds** — the function computes `xMin/xMax/yMin/yMax` across all traces to set up the coordinate system. Add a block for your trace type. This goes after the existing contour bounds block and before the `if (!isFinite(xMin)) return;` check:

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

**9e. Drawing** — add rendering code inside the clipped region. The clip region is set up earlier with `ctx.clip()`. Your drawing code goes after contour rendering and before the line-trace loop. You have access to:

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
3. **Visual test**: create an example script in `examples/plots/` and run with:
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

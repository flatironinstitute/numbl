/**
 * Plot builtin dispatch — the single source of truth for the
 * `(name, args)` → `PlotInstruction[]` mapping that every plotting
 * builtin performs.
 *
 * Historically this logic was scattered across two places:
 *   - `runtime.ts` registered the data-bearing plot names
 *     (`plot`, `surf`, `imagesc`, …) inside `initBuiltins`.
 *   - `specialBuiltins.ts` registered the graphics ops (`figure`,
 *     `title`, `xlabel`, `hold`, `subplot`, `clf`, …) inside
 *     `registerSpecialBuiltins`.
 *
 * Both sites pushed onto `rt.plotInstructions` and (for `hold` /
 * `tiledlayout` / `nexttile`) updated small flags on the Runtime.
 * `dispatchPlotBuiltin` is the consolidation. The Runtime registers
 * each name with a 1-line wrapper that calls this function and then
 * returns the right `nargout`-handle value; the side-effect (the
 * push) lives here.
 *
 * Out-of-scope names — kept registered separately because their
 * semantics need the full Runtime context:
 *   - `fplot` / `fplot3` — evaluate a user function-handle, need
 *     the Runtime's executor.
 *   - `drawnow` / `pause` — call the `ExecOptions` callbacks
 *     (`onDrawnow`, `onPause`).
 *   - `streamline` / `stream2` — return non-trivial values.
 *   - `ishold` — reads `state.holdState`, but is a pure query (no
 *     instructions pushed); the Runtime can call `state.holdState`
 *     directly.
 *
 * Reusable from outside numbl: mtoc2 (the static-translation
 * sibling project) imports this function to render plots in its
 * browser IDE without reimplementing per-builtin logic.
 */

import type { RuntimeValue } from "./types.js";
import { isRuntimeNumber, isRuntimeTensor } from "./types.js";
import type { PlotInstruction } from "../../graphics/types.js";
import { toNumber, toString } from "./convert.js";
import { RuntimeError } from "./error.js";
import {
  plotInstr,
  plotCall,
  plot3Call,
  lineCall,
  patchCall,
  fillCall,
  surfCall,
  surfaceCall,
  scatterCall,
  imagescCall,
  pcolorCall,
  contourCall,
  meshCall,
  barCall,
  barhCall,
  bar3Call,
  bar3hCall,
  stairsCall,
  errorbarCall,
  semilogxCall,
  semilogyCall,
  loglogCall,
  areaCall,
  scatter3Call,
  histogramCall,
  histogram2Call,
  boxchartCall,
  swarmchartCall,
  swarmchart3Call,
  piechartCall,
  donutchartCall,
  heatmapCall,
  quiverCall,
  quiver3Call,
  viewCall,
  legendCall,
} from "./runtimePlot.js";
import { applyAxisCommand } from "./axisCommand.js";

/** State carried by graphics ops that need to remember things between
 *  calls. `holdState` is read by `ishold()` so `hold on; if ishold ...`
 *  works; `tiledLayoutState` lets `nexttile` know where to place the
 *  next axes. The `Runtime` class implements this interface; mtoc2's
 *  browser viewer keeps its own instance. */
export interface PlotDispatchState {
  holdState: boolean;
  tiledLayoutState: TiledLayoutState | null;
  /** Current figure handle (0 = none created yet). */
  currentFigureHandle?: number;
  /** Highest figure handle allocated so far. `figure` with no argument
   *  creates a NEW figure with handle `maxFigureHandle + 1` (MATLAB
   *  semantics), rather than always reusing handle 1. */
  maxFigureHandle?: number;
}

/** Active tiled-layout grid. `mode` controls how the grid grows: in
 *  `flow` (default), nexttile expands rows/cols to fit; `vertical` and
 *  `horizontal` extend one axis; `fixed` uses the M/N from the
 *  tiledlayout(M,N) call verbatim. */
export interface TiledLayoutState {
  rows: number;
  cols: number;
  mode: "fixed" | "flow" | "vertical" | "horizontal";
  count: number;
}

/** Run a single plot-builtin call. Pushes onto `instructions` and
 *  updates `state` to mirror what numbl's runtime did before this
 *  refactor. Returns `true` iff `name` was a recognized plot builtin;
 *  callers fall through to their own dispatch on `false`. */
export function dispatchPlotBuiltin(
  name: string,
  args: RuntimeValue[],
  instructions: PlotInstruction[],
  state: PlotDispatchState
): boolean {
  switch (name) {
    // ── Data-bearing builtins: delegate to runtimePlot.ts ──────────
    case "plot":
      plotCall(instructions, args);
      return true;
    case "plot3":
      plot3Call(instructions, args);
      return true;
    case "line":
      lineCall(instructions, args);
      return true;
    case "patch":
      patchCall(instructions, args);
      return true;
    case "fill":
      fillCall(instructions, args);
      return true;
    case "surf":
      surfCall(instructions, args);
      return true;
    case "surface":
      surfaceCall(instructions, args);
      return true;
    case "scatter":
      scatterCall(instructions, args);
      return true;
    case "imagesc":
      imagescCall(instructions, args);
      return true;
    case "pcolor":
      pcolorCall(instructions, args);
      return true;
    case "contour":
      contourCall(instructions, args, false);
      return true;
    case "contourf":
      contourCall(instructions, args, true);
      return true;
    case "mesh":
    case "waterfall":
      meshCall(instructions, args);
      return true;
    case "bar":
      barCall(instructions, args);
      return true;
    case "barh":
      barhCall(instructions, args);
      return true;
    case "bar3":
      bar3Call(instructions, args);
      return true;
    case "bar3h":
      bar3hCall(instructions, args);
      return true;
    case "stairs":
      stairsCall(instructions, args);
      return true;
    case "errorbar":
      errorbarCall(instructions, args);
      return true;
    case "semilogx":
      semilogxCall(instructions, args);
      return true;
    case "semilogy":
      semilogyCall(instructions, args);
      return true;
    case "loglog":
      loglogCall(instructions, args);
      return true;
    case "area":
      areaCall(instructions, args);
      return true;
    case "scatter3":
      scatter3Call(instructions, args);
      return true;
    case "histogram":
      histogramCall(instructions, args);
      return true;
    case "histogram2":
      histogram2Call(instructions, args);
      return true;
    case "boxchart":
      boxchartCall(instructions, args);
      return true;
    case "swarmchart":
      swarmchartCall(instructions, args);
      return true;
    case "swarmchart3":
      swarmchart3Call(instructions, args);
      return true;
    case "piechart":
      piechartCall(instructions, args);
      return true;
    case "donutchart":
      donutchartCall(instructions, args);
      return true;
    case "heatmap":
      heatmapCall(instructions, args);
      return true;
    case "quiver":
      quiverCall(instructions, args);
      return true;
    case "quiver3":
      quiver3Call(instructions, args);
      return true;
    case "view":
      viewCall(instructions, args);
      return true;
    case "legend":
      legendCall(instructions, args);
      return true;

    // ── Graphics ops: figure / labels / hold / layout ──────────────
    case "figure": {
      // Forms (MATLAB):
      //   figure                 — create a NEW figure (next handle)
      //   figure(n) / figure(f)  — select/create figure n (numeric handle)
      //   figure(Name,Value,...) — create a NEW figure, set properties
      // A numeric first argument selects/creates that figure number; anything
      // else (a string property name) starts name-value pairs and creates a
      // new figure. The properties (Visible, Color, Name, Position, ...) have
      // no effect on numbl's headless rendering, so they are accepted and
      // ignored rather than coerced to a number (which used to crash).
      let handle: number;
      if (
        args.length > 0 &&
        (isRuntimeNumber(args[0]) || isRuntimeTensor(args[0]))
      ) {
        handle = toNumber(args[0]);
        state.maxFigureHandle = Math.max(state.maxFigureHandle ?? 0, handle);
      } else {
        handle = (state.maxFigureHandle ?? 0) + 1;
        state.maxFigureHandle = handle;
      }
      state.currentFigureHandle = handle;
      plotInstr(instructions, { type: "set_figure_handle", handle });
      return true;
    }
    case "subplot": {
      if (args.length >= 3) {
        plotInstr(instructions, {
          type: "set_subplot",
          rows: args[0],
          cols: args[1],
          index: args[2],
        });
      } else if (args.length === 1) {
        // Single three-digit form: subplot(mnp), e.g. subplot(131) is
        // shorthand for subplot(1,3,1). Each digit must be 1-9.
        const mnp = toNumber(args[0]);
        const rows = Math.floor(mnp / 100);
        const cols = Math.floor((mnp % 100) / 10);
        const index = mnp % 10;
        if (rows >= 1 && cols >= 1 && index >= 1 && index <= rows * cols) {
          plotInstr(instructions, { type: "set_subplot", rows, cols, index });
        }
      }
      return true;
    }
    case "tiledlayout":
      dispatchTiledLayout(args, instructions, state);
      return true;
    case "nexttile":
      dispatchNextTile(args, instructions, state);
      return true;
    case "title":
    case "xlabel":
    case "ylabel":
    case "zlabel":
    case "sgtitle": {
      if (args.length > 0) {
        const type =
          name === "title"
            ? "set_title"
            : name === "xlabel"
              ? "set_xlabel"
              : name === "ylabel"
                ? "set_ylabel"
                : name === "zlabel"
                  ? "set_zlabel"
                  : "set_sgtitle";
        plotInstr(instructions, { type, text: args[0] });
      }
      return true;
    }
    case "hold": {
      if (args.length > 0) {
        plotInstr(instructions, { type: "set_hold", value: args[0] });
        // runtimePlot.plotInstr already parsed args[0] into a boolean
        // when pushing; mirror that to the dispatch state so an
        // adjacent ishold() returns the right thing.
        const last = instructions[instructions.length - 1];
        if (last && last.type === "set_hold") state.holdState = last.value;
      }
      return true;
    }
    case "grid": {
      if (args.length > 0) {
        plotInstr(instructions, { type: "set_grid", value: args[0] });
      }
      return true;
    }
    case "box": {
      if (args.length > 0) {
        plotInstr(instructions, { type: "set_box", value: args[0] });
      }
      return true;
    }
    case "close": {
      if (args.length > 0 && toString(args[0]) === "all") {
        plotInstr(instructions, { type: "close_all" });
      } else {
        plotInstr(instructions, { type: "close" });
      }
      return true;
    }
    case "clf":
      plotInstr(instructions, { type: "clf" });
      return true;
    case "cla": {
      // cla / cla(ax) / cla reset / cla(ax,'reset'). numbl has no real
      // axes handles, so an `ax` argument is accepted and ignored — we
      // always act on the current axes. Any 'reset' argument also resets
      // the axes properties to their defaults.
      let reset = false;
      for (const a of args) {
        try {
          if (
            toString(a)
              .toLowerCase()
              .replace(/^["']|["']$/g, "") === "reset"
          )
            reset = true;
        } catch {
          // Non-string arg (the ax handle) — ignore.
        }
      }
      plotInstr(instructions, { type: "cla", reset });
      return true;
    }
    case "shading": {
      if (args.length > 0) {
        plotInstr(instructions, { type: "set_shading", shading: args[0] });
      }
      return true;
    }
    case "colorbar":
      dispatchColorbar(args, instructions);
      return true;
    case "colormap":
      dispatchColormap(args, instructions);
      return true;
    case "axis": {
      // The query form (`lim = axis`) needs the runtime's accumulated state
      // and is handled by the Runtime's own `axis` override; here we only
      // apply the setting forms. `manual`-freeze is likewise a Runtime-only
      // concern (no current-limits provider in this generic path).
      applyAxisCommand(args, instructions);
      return true;
    }
    case "caxis":
    case "clim": {
      if (args.length > 0) {
        const rv = args[0];
        if (isRuntimeTensor(rv) && rv.data.length >= 2) {
          plotInstr(instructions, {
            type: "set_caxis",
            limits: [rv.data[0], rv.data[1]],
          });
        }
      }
      return true;
    }

    default:
      return false;
  }
}

/** `tiledlayout([M,N | 'flow'|'vertical'|'horizontal'], NV...)` — replace
 *  any existing layout with a fresh axes grid. NV-pairs are accepted
 *  for source compatibility but otherwise ignored. */
function dispatchTiledLayout(
  args: RuntimeValue[],
  instructions: PlotInstruction[],
  state: PlotDispatchState
): void {
  let mode: "fixed" | "flow" | "vertical" | "horizontal" = "flow";
  let rows = 1;
  let cols = 1;
  let pos = 0;
  if (args.length === 0) {
    mode = "flow";
  } else {
    const first = args[0];
    if (isRuntimeNumber(first)) {
      if (args.length < 2 || !isRuntimeNumber(args[1])) {
        throw new RuntimeError(
          "tiledlayout: numeric form requires both M and N"
        );
      }
      rows = Math.max(1, Math.floor(toNumber(first)));
      cols = Math.max(1, Math.floor(toNumber(args[1])));
      mode = "fixed";
      pos = 2;
    } else {
      const s = toString(first).toLowerCase();
      if (s === "flow" || s === "vertical" || s === "horizontal") {
        mode = s;
        pos = 1;
      } else {
        throw new RuntimeError(
          `tiledlayout: unknown arrangement '${toString(first)}'`
        );
      }
    }
  }
  // Silently consume remaining name-value pairs (TileSpacing, Padding, …).
  while (pos + 1 < args.length) pos += 2;
  plotInstr(instructions, { type: "clf" });
  state.tiledLayoutState = { rows, cols, mode, count: 0 };
}

/** `nexttile([index | [r c]])` — advance the active tile in the current
 *  tiled layout. If no layout is active, create a 1×1 flow layout
 *  first. The span argument is accepted for compatibility but not
 *  rendered as a true span. */
function dispatchNextTile(
  args: RuntimeValue[],
  instructions: PlotInstruction[],
  state: PlotDispatchState
): void {
  if (!state.tiledLayoutState) {
    plotInstr(instructions, { type: "clf" });
    state.tiledLayoutState = { rows: 1, cols: 1, mode: "flow", count: 0 };
  }
  const st = state.tiledLayoutState;
  let targetIndex: number | null = null;
  if (args.length >= 1 && isRuntimeNumber(args[0])) {
    targetIndex = Math.max(1, Math.floor(toNumber(args[0])));
  }
  if (targetIndex === null) {
    st.count += 1;
    targetIndex = st.count;
  } else if (targetIndex > st.count) {
    st.count = targetIndex;
  }
  let { rows, cols } = st;
  if (st.mode === "flow") {
    const g = flowGrid(st.count);
    rows = g.rows;
    cols = g.cols;
    st.rows = rows;
    st.cols = cols;
  } else if (st.mode === "vertical") {
    rows = Math.max(1, st.count);
    cols = 1;
    st.rows = rows;
    st.cols = cols;
  } else if (st.mode === "horizontal") {
    rows = 1;
    cols = Math.max(1, st.count);
    st.rows = rows;
    st.cols = cols;
  }
  plotInstr(instructions, {
    type: "set_subplot",
    rows,
    cols,
    index: targetIndex,
  });
}

/** Compute a flow grid (rows × cols) that fits n tiles roughly
 *  squarely. Mirrors MATLAB's tiledlayout('flow') growth rule. */
function flowGrid(n: number): { rows: number; cols: number } {
  const k = Math.max(1, n);
  const cols = Math.max(1, Math.ceil(Math.sqrt(k)));
  const rows = Math.max(1, Math.ceil(k / cols));
  return { rows, cols };
}

/** `colorbar(['off' | <location> | NV-pairs ...])` — toggle the colorbar
 *  or pin its position. NV-pairs (Direction, Ticks, …) are accepted and
 *  silently skipped. */
function dispatchColorbar(
  args: RuntimeValue[],
  instructions: PlotInstruction[]
): void {
  const LOCATIONS = new Set([
    "east",
    "west",
    "north",
    "south",
    "eastoutside",
    "westoutside",
    "northoutside",
    "southoutside",
  ]);
  let value = "on";
  let location: string | undefined;
  let i = 0;
  while (i < args.length) {
    let s: string;
    try {
      s = toString(args[i]);
    } catch {
      break;
    }
    const lower = s.toLowerCase();
    if (lower === "off") {
      value = "off";
      i++;
      continue;
    }
    if (LOCATIONS.has(lower)) {
      location = lower;
      i++;
      continue;
    }
    // Name-value pairs we don't model — accept and skip.
    i += 2;
  }
  plotInstr(instructions, { type: "set_colorbar", value, location });
}

/** `colormap(<name> | <N×3 matrix>)` — switch the current colormap.
 *  A matrix arg becomes a custom colormap; anything else is treated as
 *  a name string. */
function dispatchColormap(
  args: RuntimeValue[],
  instructions: PlotInstruction[]
): void {
  if (args.length === 0) return;
  const rv = args[0];
  if (isRuntimeTensor(rv) && rv.shape.length === 2 && rv.shape[1] === 3) {
    const rows = rv.shape[0];
    const data: number[][] = [];
    for (let i = 0; i < rows; i++) {
      data.push([rv.data[i], rv.data[rows + i], rv.data[2 * rows + i]]);
    }
    plotInstr(instructions, {
      type: "set_colormap",
      name: "__custom__",
      data,
    });
  } else {
    const name = toString(rv).replace(/^"|"$/g, "");
    plotInstr(instructions, { type: "set_colormap", name });
  }
}

/** Names handled by `dispatchPlotBuiltin`. Useful for registering all
 *  of them in a loop (numbl's `initBuiltins` / `registerSpecialBuiltins`
 *  do that for the side-effect-only subset). Stays in sync with the
 *  cases above by construction. */
export const PLOT_DISPATCH_NAMES: ReadonlyArray<string> = [
  "plot",
  "plot3",
  "line",
  "patch",
  "fill",
  "surf",
  "surface",
  "scatter",
  "imagesc",
  "pcolor",
  "contour",
  "contourf",
  "mesh",
  "waterfall",
  "bar",
  "barh",
  "bar3",
  "bar3h",
  "stairs",
  "errorbar",
  "semilogx",
  "semilogy",
  "loglog",
  "area",
  "scatter3",
  "histogram",
  "histogram2",
  "boxchart",
  "swarmchart",
  "swarmchart3",
  "piechart",
  "donutchart",
  "heatmap",
  "quiver",
  "quiver3",
  "view",
  "legend",
  "figure",
  "subplot",
  "tiledlayout",
  "nexttile",
  "title",
  "xlabel",
  "ylabel",
  "zlabel",
  "sgtitle",
  "hold",
  "grid",
  "box",
  "close",
  "clf",
  "cla",
  "shading",
  "colorbar",
  "colormap",
  "axis",
  "caxis",
  "clim",
];

/** Graphics-related names numbl exposes as no-op (or near-no-op)
 *  stubs — they don't push plot instructions and don't render, but
 *  numbl accepts them so MATLAB scripts compile and run. Sources:
 *    - `interpreter/builtins/misc.ts` (`groot`, `gcf`, `gca`, `shg`,
 *      `newplot`, `xlim`, `ylim`, `set`, `get`, `ishold`)
 *    - `runtime/specialBuiltins.ts` (`drawnow`, `pause`)
 *
 *  These names belong here, alongside `PLOT_DISPATCH_NAMES`, because
 *  downstream consumers (mtoc2's translator, syntax highlighters,
 *  doc generators) want to know the full "plot-script surface area"
 *  numbl tolerates — not just the renderable subset. Keeping the
 *  union in one place is the whole point of consolidating dispatch
 *  here. */
export const PLOT_STUB_NAMES: ReadonlyArray<string> = [
  "groot",
  "gcf",
  "gca",
  "shg",
  "newplot",
  "xlim",
  "ylim",
  "drawnow",
  "pause",
  "ishold",
  "set",
  "get",
];

/** The union of `PLOT_DISPATCH_NAMES` (renderable) and
 *  `PLOT_STUB_NAMES` (recognized-but-no-render). Every name a plotting
 *  script might invoke that numbl tolerates. Mtoc2 imports this to
 *  drive its own builtin registration so the two repos stay in
 *  lockstep — when numbl gains a new plot builtin, mtoc2 picks it up
 *  at the next `tsc` / `vite build` with no mtoc2-side edit. */
export const PLOT_ALL_NAMES: ReadonlyArray<string> = [
  ...PLOT_DISPATCH_NAMES,
  ...PLOT_STUB_NAMES,
];

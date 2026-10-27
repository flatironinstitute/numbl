/**
 * Plot builtins: data-bearing plot calls (plot, surf, contour, ...) and the
 * graphics ops that return handles or need the runtime's plot state. Graphics
 * ops with figure/axes state (figure, hold, subplot, ...) live in
 * specialBuiltins.ts.
 */

// Registers figure, hold, subplot, ... first, so the generic loop at the
// end of this file skips them.
import "../runtime/specialBuiltins.js";
import type { Runtime } from "../runtime/runtime.js";
import type { RuntimeValue } from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { allocFloat64Array } from "../runtime/alloc.js";
import { registerSpecial } from "../runtime/registerSpecial.js";
import { getIBuiltin } from "../interpreter/builtins/types.js";
import {
  pcolorCall as _pcolorCall,
  lineCall as _lineCall,
  patchCall as _patchCall,
  trimeshCall as _trimeshCall,
  trisurfCall as _trisurfCall,
  fillCall as _fillCall,
  contourCall as _contourCall,
  quiver3Call as _quiver3Call,
  fplotCall as _fplotCall,
  fplot3Call as _fplot3Call,
  streamlineCall as _streamlineCall,
  stream2Call as _stream2Call,
} from "./instructions.js";
import { computeContourMatrix } from "./parseArgs.js";
import { dispatchPlotBuiltin, PLOT_DISPATCH_NAMES } from "./dispatch.js";
import { applyAxisCommand } from "./axisCommand.js";
import { isosurfaceFromArgs } from "../helpers/marching-cubes.js";

/** Graphics ops that do not create an implicit figure (the `axis` query form
 *  must not, and the others manage figures themselves). */
const NO_AUTO_FIGURE = new Set([
  "figure",
  "close",
  "clf",
  "cla",
  "axis",
  "isosurface",
  "cameratoolbar",
]);

/** Register a plot builtin. Any graphics op other than those in
 *  NO_AUTO_FIGURE implies a current figure: register the implicit figure 1
 *  on first use so a later no-arg `figure` allocates a new handle. */
function registerPlot(
  name: string,
  fn: (nargout: number, args: RuntimeValue[], rt: Runtime) => unknown
): void {
  registerSpecial(name, (nargout, args, rt) => {
    if (!NO_AUTO_FIGURE.has(name)) rt.ensureCurrentFigure();
    return fn(nargout, args, rt);
  });
}

// `pcolor` returns a graphics handle when an output is requested —
// the dispatch table pushes the instruction, and we read back the
// freshly-pushed trace for the handle. Override the loop entry.
registerPlot("pcolor", (_nargout, args, rt) => {
  const margs = args.map(a => ensureRuntimeValue(a));
  _pcolorCall(rt.plotInstructions, margs);
  if (_nargout >= 1) {
    const last = rt.plotInstructions[rt.plotInstructions.length - 1];
    if (last && last.type === "pcolor") {
      return RTV.graphicsHandle(
        last.trace as unknown as Record<string, unknown>,
        "pcolor"
      );
    }
    return RTV.dummyHandle();
  }
});
// `line` returns a primitive Line handle when an output is requested
// (`pl = line(...)`). Like pcolor, the dispatch table pushes the
// instruction and we wrap the freshly-pushed trace so later property
// assignments (e.g. `pl.Color = 'green'`) update the rendered line.
registerPlot("line", (_nargout, args, rt) => {
  const margs = args.map(a => ensureRuntimeValue(a));
  _lineCall(rt.plotInstructions, margs);
  if (_nargout >= 1) {
    const last = rt.plotInstructions[rt.plotInstructions.length - 1];
    if (last && (last.type === "line" || last.type === "line3")) {
      const traces = last.traces;
      if (traces.length > 0) {
        // Tag the returned trace so later `set(pl,'XData',...)` calls can
        // find and live-update it via an `update_trace` instruction.
        traces[0].id = rt.graphicsIdCounter++;
        return RTV.graphicsHandle(
          traces[0] as unknown as Record<string, unknown>,
          "line"
        );
      }
    }
    return RTV.dummyHandle();
  }
});
// `patch` returns a Patch handle when an output is requested
// (`p = patch(...)`). Wrap the freshly-pushed trace and tag it with an id
// so `set(p,'FaceColor',...)` can live-update the rendered patch.
registerPlot("patch", (_nargout, args, rt) => {
  const margs = args.map(a => ensureRuntimeValue(a));
  _patchCall(rt.plotInstructions, margs);
  if (_nargout >= 1) {
    const last = rt.plotInstructions[rt.plotInstructions.length - 1];
    if (last && last.type === "patch") {
      last.trace.id = rt.graphicsIdCounter++;
      return RTV.graphicsHandle(
        last.trace as unknown as Record<string, unknown>,
        "patch"
      );
    }
    return RTV.dummyHandle();
  }
});
// `trimesh` / `trisurf` draw a triangular mesh as a patch; `h = trimesh(...)`
// returns the patch handle so later property assignments update the mesh.
for (const [name, call] of [
  ["trimesh", _trimeshCall],
  ["trisurf", _trisurfCall],
] as const) {
  registerPlot(name, (_nargout, args, rt) => {
    const margs = args.map(a => ensureRuntimeValue(a));
    call(rt.plotInstructions, margs);
    if (_nargout >= 1) {
      const last = rt.plotInstructions[rt.plotInstructions.length - 1];
      if (last && last.type === "patch") {
        last.trace.id = rt.graphicsIdCounter++;
        return RTV.graphicsHandle(
          last.trace as unknown as Record<string, unknown>,
          "patch"
        );
      }
      return RTV.dummyHandle();
    }
  });
}
// `isosurface` extracts a triangulated surface from a 3-D volume via
// marching cubes. With no output it draws a 3-D patch (reusing the patch
// renderer); otherwise it returns the geometry as a struct `{vertices,
// faces}` (nargout 1), `[faces, verts]` (2), or `[faces, verts, colors]`
// (3). Faces are 1-based in the returned data (MATLAB convention) but
// 0-based in the patch trace.
registerPlot("isosurface", (_nargout, args, rt) => {
  const margs = args.map(a => ensureRuntimeValue(a));
  const mesh = isosurfaceFromArgs(margs);
  const nv = mesh.vertices.length;
  const nf = mesh.faces.length;
  const vertsTensor = (): RuntimeValue => {
    const d = allocFloat64Array(nv * 3);
    for (let c = 0; c < 3; c++)
      for (let r = 0; r < nv; r++) d[c * nv + r] = mesh.vertices[r][c];
    return RTV.tensor(d, [nv, 3]);
  };
  const facesTensor = (): RuntimeValue => {
    const d = allocFloat64Array(nf * 3);
    for (let c = 0; c < 3; c++)
      for (let r = 0; r < nf; r++) d[c * nf + r] = mesh.faces[r][c] + 1;
    return RTV.tensor(d, [nf, 3]);
  };
  const colorsTensor = (): RuntimeValue =>
    RTV.tensor(allocFloat64Array(mesh.colors ?? []), [
      mesh.colors?.length ?? 0,
      mesh.colors && mesh.colors.length > 0 ? 1 : 0,
    ]);

  if (_nargout === 0) {
    // Draw: push a 3-D patch trace (0-based faces).
    const trace: Record<string, unknown> = {
      vertices: mesh.vertices,
      faces: mesh.faces,
      is3D: true,
      faceColor: [0.8, 0.9, 1] as [number, number, number],
      edgeColor: "none",
    };
    if (mesh.colors) {
      trace.faceVertexCData = mesh.colors;
      trace.faceColor = "interp";
    }
    rt.plotInstructions.push({
      type: "patch",
      trace: trace as never,
    });
    return undefined;
  }
  if (_nargout === 1) {
    const fields: Record<string, RuntimeValue> = {
      vertices: vertsTensor(),
      faces: facesTensor(),
    };
    if (mesh.colors) fields.facevertexcdata = colorsTensor();
    return RTV.struct(fields);
  }
  if (_nargout === 2) return [facesTensor(), vertsTensor()];
  return [facesTensor(), vertsTensor(), colorsTensor()];
});
// `fill` creates filled 2-D patches (one per X,Y,C group). It reuses the
// patch trace/handle machinery; `p = fill(...)` returns the handle of the
// first patch created (MATLAB returns a vector when there are several).
registerPlot("fill", (_nargout, args, rt) => {
  const margs = args.map(a => ensureRuntimeValue(a));
  const start = rt.plotInstructions.length;
  _fillCall(rt.plotInstructions, margs);
  if (_nargout >= 1) {
    let firstHandle: RuntimeValue | undefined;
    for (let i = start; i < rt.plotInstructions.length; i++) {
      const instr = rt.plotInstructions[i];
      if (instr.type === "patch") {
        instr.trace.id = rt.graphicsIdCounter++;
        if (!firstHandle) {
          firstHandle = RTV.graphicsHandle(
            instr.trace as unknown as Record<string, unknown>,
            "patch"
          );
        }
      }
    }
    return firstHandle ?? RTV.dummyHandle();
  }
});
// `contour` / `contourf` push the plot instruction and, when an output
// is requested, also return the MATLAB contour matrix `C` (computed via
// marching squares) and a contour handle `H` exposing the line
// properties / level list. Mirrors `[C,H] = contour(...)`.
const contourOverride =
  (filled: boolean) =>
  (_nargout: number, args: RuntimeValue[], rt: Runtime) => {
    const margs = args.map(a => ensureRuntimeValue(a));
    _contourCall(rt.plotInstructions, margs, filled);
    if (_nargout < 1) return undefined;
    const last = rt.plotInstructions[rt.plotInstructions.length - 1];
    if (!last || last.type !== "contour") return RTV.dummyHandle();
    const trace = last.trace;
    const cm = computeContourMatrix(trace);
    const C = RTV.tensor(allocFloat64Array(cm.data), [2, cm.n] as number[]);
    if (_nargout < 2) return C;
    const H = RTV.graphicsHandle(
      {
        LineWidth: trace.lineWidth ?? 0.5,
        LineStyle: trace.lineStyle ?? "-",
        LineColor: trace.lineColor ?? "flat",
        LevelList: cm.levelList,
        __instruction: last,
      },
      "contour"
    );
    return [C, H];
  };
registerPlot("contour", contourOverride(false));
registerPlot("contourf", contourOverride(true));
// `quiver3` returns a Quiver handle when an output is requested. The
// handle wraps the pushed trace object directly, so later property
// assignments (e.g. `q.ShowArrowHead = 'off'`) update the rendered plot.
registerPlot("quiver3", (_nargout, args, rt) => {
  const margs = args.map(a => ensureRuntimeValue(a));
  _quiver3Call(rt.plotInstructions, margs);
  if (_nargout < 1) return undefined;
  const last = rt.plotInstructions[rt.plotInstructions.length - 1];
  if (last && last.type === "quiver3") {
    return RTV.graphicsHandle(
      last.trace as unknown as Record<string, unknown>,
      "quiver3"
    );
  }
  return RTV.dummyHandle();
});
// `axis` has a query form (`lim = axis` / `lim = axis(ax)`) that returns
// the current limits, and an `axis manual` form that freezes the current
// limits — both need the runtime's accumulated plot state, so they are
// handled here rather than in the generic dispatch.
registerPlot("axis", (nargout, args, rt) => {
  const margs = args.map(a => ensureRuntimeValue(a));
  const applied = applyAxisCommand(margs, rt.plotInstructions, () =>
    rt.currentAxisLimits()
  );
  if (!applied && nargout >= 1) {
    return RTV.row(rt.currentAxisLimits());
  }
  return undefined;
});
// `fplot` / `fplot3` evaluate a user function-handle and so need
// the full Runtime context — kept as explicit registrations.
registerPlot("fplot", (_nargout, args, rt) => {
  _fplotCall(rt, rt.plotInstructions, args.map(ensureRuntimeValue));
});
registerPlot("fplot3", (_nargout, args, rt) => {
  _fplot3Call(rt, rt.plotInstructions, args.map(ensureRuntimeValue));
});
// `streamline` / `stream2` return non-trivial values (handle / matrix).
registerPlot("streamline", (_nargout, args, rt) => {
  const h = _streamlineCall(rt.plotInstructions, args.map(ensureRuntimeValue));
  if (_nargout >= 1) return h;
});
registerPlot("stream2", (_nargout, args) => {
  return _stream2Call(args.map(ensureRuntimeValue));
});
// `cameratoolbar` — accepted as a no-op (numbl has no interactive camera
// toolbar). State/SetMode/Get… arguments are ignored. Returns a dummy
// handle only when an output is requested, so the bare-statement form
// (used by simpplot/distmeshsurface) does not print an `ans`.
registerPlot("cameratoolbar", _nargout => {
  if (_nargout >= 1) return RTV.dummyHandle();
});

// Every other name handled by `dispatchPlotBuiltin` only pushes its plot
// instruction. Names registered above, or by specialBuiltins.ts (figure,
// hold, subplot, ...), keep those registrations.
for (const name of PLOT_DISPATCH_NAMES) {
  if (getIBuiltin(name)) continue;
  registerPlot(name, (_nargout, args, rt) => {
    dispatchPlotBuiltin(name, args, rt.plotInstructions, rt);
  });
}

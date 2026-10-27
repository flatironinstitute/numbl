/**
 * Plot builtin registrations: data-bearing plot calls (plot, surf, contour,
 * ...), graphics ops with figure/axes state (figure, hold, subplot, ...), and
 * the ops that return handles or need the runtime's plot state. Names with no
 * custom registration fall through to the generic loop at the end, which
 * routes them to `dispatchPlotBuiltin`.
 */

import type { Runtime } from "../runtime/runtime.js";
import {
  type RuntimeValue,
  isRuntimeChar,
  isRuntimeFunction,
  isRuntimeGraphicsHandle,
  isRuntimeString,
  isRuntimeStruct,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { toString } from "../runtime/convert.js";
import { RuntimeError } from "../runtime/error.js";
import { ensureRuntimeValue } from "../runtime/runtimeHelpers.js";
import { allocFloat64Array } from "../runtime/alloc.js";
import { incref, decref } from "../runtime/refcount.js";
import { applyHandleProperty } from "../runtime/struct-access.js";
import {
  registerSpecial,
  registerSpecialVoid,
} from "../runtime/registerSpecial.js";
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
} from "./instructions.js";
import {
  streamlineCall as _streamlineCall,
  stream2Call as _stream2Call,
} from "./streamline.js";
import { computeContourMatrix } from "./parseArgsSurface.js";
import { dispatchPlotBuiltin, PLOT_DISPATCH_NAMES } from "./dispatch.js";
import { applyAxisCommand } from "./axisCommand.js";
import type { PlotInstruction } from "./types.js";
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

// ── drawnow / pause ──────────────────────────────────────────────────

registerSpecialVoid("drawnow", (_args, rt) => {
  rt.drawnow();
});

registerSpecial("pause", (nargout, args, rt) => {
  rt.pause(args[0] ?? 0);
  // MATLAB returns the current pause state when assigned.
  return nargout >= 1 ? RTV.char("on") : undefined;
});

// ── Graphics ops ─────────────────────────────────────────────────────
//
// In MATLAB most plotting functions have an *optional* output (a graphics
// handle) and do not set `ans` when called as statements.  The numbl
// stubs don't track real handles, so they hand back placeholder values
// only when an output is explicitly requested.

registerSpecial("ishold", (_nargout, _args, rt) => {
  return rt.ishold();
});

registerSpecial("figure", (nargout, args, rt) => {
  dispatchPlotBuiltin(
    "figure",
    args.map(ensureRuntimeValue),
    rt.plotInstructions,
    rt
  );
  // dispatchPlotBuiltin set rt.currentFigureHandle to the (possibly newly
  // allocated) handle; return it so `h = figure` gets the right value.
  return nargout >= 1 ? RTV.num(rt.currentFigureHandle) : undefined;
});

// `drawuihtml(html[, dataJson])` — native emit primitive behind the `uihtml`
// component. Pushes a `uihtml` PlotInstruction (rendered as an iframe by
// FigureView) onto the current figure. A fresh id per call makes the viewer
// remount the iframe when HTMLSource or Data changes. `dataJson`, when given,
// is the `Data` property already JSON-encoded (the shim calls `jsonencode`);
// the renderer parses it and feeds it to the page's `htmlComponent` so the
// `setup`/`"DataChanged"` bridge fires.
registerSpecial("drawuihtml", (nargout, args, rt) => {
  const html = args.length > 0 ? toString(ensureRuntimeValue(args[0])) : "";
  const data =
    args.length > 1 ? toString(ensureRuntimeValue(args[1])) : undefined;
  rt.uihtmlSeq += 1;
  const id = "uh" + rt.uihtmlSeq;
  rt.plotInstructions.push({
    type: "uihtml",
    id,
    html,
    ...(data !== undefined ? { data } : {}),
  });
  // Return the component id so the shim can register reverse-channel
  // callbacks (HTMLEventReceivedFcn / DataChangedFcn) against it.
  return nargout >= 1 ? RTV.char(id) : undefined;
});

// `registeruihtmlcallback(compId, eventType, fcnHandle)` — record a callback
// for a uihtml component's reverse channel (HTML → MATLAB). `eventType` is
// "HTMLEventReceived" (from JS `sendEventToMATLAB`) or "DataChanged" (from JS
// setting `htmlComponent.Data`). The handle is incref'd so it survives after
// the run while the host keeps this runtime alive to dispatch events. A new
// registration for the same component replaces (and decrefs) the old one.
registerSpecial("registeruihtmlcallback", (_nargout, args, rt) => {
  if (args.length < 3) return undefined;
  const compId = toString(ensureRuntimeValue(args[0]));
  const eventType = toString(ensureRuntimeValue(args[1]));
  const fcn = ensureRuntimeValue(args[2]);
  if (eventType !== "HTMLEventReceived" && eventType !== "DataChanged") {
    return undefined;
  }
  if (!isRuntimeFunction(fcn)) return undefined;
  const entry = rt.uihtmlCallbacks.get(compId) ?? {};
  const prev = entry[eventType];
  if (prev) decref(rt, prev);
  incref(fcn);
  entry[eventType] = fcn;
  rt.uihtmlCallbacks.set(compId, entry);
  return undefined;
});

// `sendEventToHTMLSource(src, name, data)` — push an event from MATLAB back to
// a uihtml component's page (MATLAB → JS). `src` is the struct passed to the
// callback (carrying ComponentId); `data` is jsonencode'd and the host
// forwards it to the iframe, where it fires the page's addEventListener(name).
registerSpecial("sendEventToHTMLSource", (_nargout, args, rt) => {
  if (args.length < 2) {
    throw new RuntimeError("sendEventToHTMLSource requires src and name");
  }
  const src = ensureRuntimeValue(args[0]);
  if (!isRuntimeStruct(src) || !src.fields.has("ComponentId")) {
    throw new RuntimeError(
      "sendEventToHTMLSource: src must be the component passed to the callback"
    );
  }
  const compId = toString(src.fields.get("ComponentId")!);
  const name = toString(ensureRuntimeValue(args[1]));
  const dataArg = args.length > 2 ? ensureRuntimeValue(args[2]) : RTV.num(0);
  const dataJson = toString(
    ensureRuntimeValue(rt.dispatch("jsonencode", 1, [dataArg]))
  );
  rt.onHtmlSourceEvent?.(compId, name, dataJson);
  return undefined;
});

// MATLAB `uigridlayout` shim so standard layout-wrapped uihtml code runs
// unchanged in numbl. numbl fills the figure pane regardless of layout, so
// this is a no-op container handle (parent/sizing args are ignored). Use a
// plain `figure` as the parent.
registerSpecial("uigridlayout", nargout =>
  nargout >= 1 ? RTV.num(0) : undefined
);

// Bulk register graphics ops that share the simple
//   "push instruction, return RTV.num(<placeholder>) when nargout>=1"
// pattern. The push side lives in plotting/dispatch.ts — one source
// of truth for the (name → set_xxx) mapping.
const PLOT_RETURNS_ZERO = ["subplot", "tiledlayout", "nexttile", "legend"];
for (const name of PLOT_RETURNS_ZERO) {
  registerSpecial(name, (nargout, args, rt) => {
    dispatchPlotBuiltin(
      name,
      args.map(ensureRuntimeValue),
      rt.plotInstructions,
      rt
    );
    return nargout >= 1 ? RTV.num(0) : undefined;
  });
}

// Text-creating ops (`title`, `xlabel`, …) push their instruction and, when
// an output is requested (`tt = title(...)`), return a text handle. The
// handle remembers which instruction sets its string so `set(tt,'String',s)`
// can re-emit it — letting titles/labels animate across `drawnow`.
const TEXT_LABELS: Record<
  string,
  "set_title" | "set_xlabel" | "set_ylabel" | "set_zlabel" | "set_sgtitle"
> = {
  title: "set_title",
  xlabel: "set_xlabel",
  ylabel: "set_ylabel",
  zlabel: "set_zlabel",
  sgtitle: "set_sgtitle",
};
for (const [name, instrType] of Object.entries(TEXT_LABELS)) {
  registerSpecial(name, (nargout, args, rt) => {
    const margs = args.map(ensureRuntimeValue);
    dispatchPlotBuiltin(name, margs, rt.plotInstructions, rt);
    if (nargout >= 1) {
      const text = margs.length > 0 ? toString(margs[0]) : "";
      return RTV.graphicsHandle({ string: text, __target: instrType }, "text");
    }
    return undefined;
  });
}

// `set(handle, 'Prop', value, ...)` updates graphics-handle properties. For
// a live-updatable trace (e.g. a line returned by `line`) it mutates the
// wrapped trace AND re-emits an `update_trace` instruction, since plot
// batches are serialized to the viewer and an already-flushed instruction
// can't be reached by mutation alone. `set` on a dummy handle (gca/gcf) or a
// non-handle is accepted as a no-op.
registerSpecial("set", (_nargout, args, rt) => {
  const margs = args.map(ensureRuntimeValue);
  const handle = margs[0];
  if (!isRuntimeGraphicsHandle(handle)) return undefined;
  const props: Record<string, unknown> = {};
  for (let i = 1; i + 1 < margs.length; i += 2) {
    const keyArg = margs[i];
    const key = isRuntimeString(keyArg)
      ? keyArg
      : isRuntimeChar(keyArg)
        ? keyArg.value
        : null;
    if (!key) continue;
    const val = margs[i + 1];
    if (handle._traceType === "text") {
      // Title/label handle: 'String' updates the text; re-emit the matching
      // label instruction. Other text props (FontSize, …) aren't modeled.
      if (key.toLowerCase() === "string") {
        const text = toString(val);
        handle._trace.string = text;
        const target = handle._trace.__target as
          | PlotInstruction["type"]
          | undefined;
        if (target) {
          rt.plotInstructions.push({ type: target, text } as PlotInstruction);
        }
      }
      continue;
    }
    const { key: tkey, value: stored } = applyHandleProperty(
      handle._trace,
      handle._traceType,
      key,
      val
    );
    props[tkey] = stored;
  }
  const id = handle._trace.id;
  if (typeof id === "number" && Object.keys(props).length > 0) {
    rt.plotInstructions.push({ type: "update_trace", id, props });
  }
  return undefined;
});

const PLOT_RETURNS_ONE = ["close", "clf"];
for (const name of PLOT_RETURNS_ONE) {
  registerSpecial(name, (nargout, args, rt) => {
    dispatchPlotBuiltin(
      name,
      args.map(ensureRuntimeValue),
      rt.plotInstructions,
      rt
    );
    return nargout >= 1 ? RTV.num(1) : undefined;
  });
}

// `hold`, `grid`, `shading`, `cla` are truly void in MATLAB (they error
// on `r = hold`). registerSpecialVoid surfaces that as
// "Too many output arguments." when nargout > 0.
const PLOT_VOID = ["hold", "grid", "shading", "cla"];
for (const name of PLOT_VOID) {
  registerSpecialVoid(name, (args, rt) => {
    dispatchPlotBuiltin(
      name,
      args.map(ensureRuntimeValue),
      rt.plotInstructions,
      rt
    );
  });
}

// `colorbar` returns a handle-shaped struct when an output is
// requested — kept as a custom registration so we can build the
// struct from the args we just parsed. The push side still goes
// through dispatchPlotBuiltin (which re-parses the args to produce
// the same set_colorbar instruction).
registerSpecial("colorbar", (nargout, args, rt) => {
  const margs = args.map(ensureRuntimeValue);
  dispatchPlotBuiltin("colorbar", margs, rt.plotInstructions, rt);
  if (nargout >= 1) {
    const last = rt.plotInstructions[rt.plotInstructions.length - 1];
    if (last && last.type === "set_colorbar") {
      return RTV.struct({
        Location: RTV.char(last.location ?? "eastoutside"),
        Visible: RTV.char(last.value === "off" ? "off" : "on"),
      });
    }
  }
  return undefined;
});

// Every other name handled by `dispatchPlotBuiltin` only pushes its plot
// instruction. Names registered above keep those registrations.
for (const name of PLOT_DISPATCH_NAMES) {
  if (getIBuiltin(name)) continue;
  registerPlot(name, (_nargout, args, rt) => {
    dispatchPlotBuiltin(name, args, rt.plotInstructions, rt);
  });
}

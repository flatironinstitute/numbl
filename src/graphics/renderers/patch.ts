import type { PatchTrace, PlotTrace } from "../../numbl-core/plotting/types.js";
import {
  colormapLookup,
  type ColormapSpec,
} from "../../numbl-core/plotting/colormaps.js";
import { getLineDash } from "../plotHelpers.js";
import { drawMarkers } from "../plotMarkers.js";
import type { DataRenderer } from "./types.js";

/** Render a patch trace: filled polygons with edges and optional markers.
 *  Supports single-color / 'flat' / 'interp' / 'none' for both face and edge,
 *  scalar (colormap-mapped) or RGB color data, faceAlpha, and line styling.
 *  A polygon containing a non-finite vertex is drawn as an open polyline
 *  (matching MATLAB's "NaN makes patch draw a line"). */
function drawPatch(
  ctx: CanvasRenderingContext2D,
  trace: PatchTrace,
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number,
  colormap: ColormapSpec,
  caxis?: [number, number]
) {
  const { vertices, faces } = trace;
  if (!faces || faces.length === 0 || vertices.length === 0) return;
  const cdata = trace.faceVertexCData;

  // Color range for scalar color data.
  let cMin = Infinity;
  let cMax = -Infinity;
  if (caxis) {
    [cMin, cMax] = caxis;
  } else if (cdata) {
    for (const cv of cdata) {
      if (typeof cv === "number" && isFinite(cv)) {
        if (cv < cMin) cMin = cv;
        if (cv > cMax) cMax = cv;
      }
    }
  }
  const cRange = isFinite(cMin) && cMax > cMin ? cMax - cMin : 1;
  const mapColor = (
    entry: number | [number, number, number] | undefined
  ): [number, number, number] => {
    if (Array.isArray(entry)) return entry;
    if (typeof entry !== "number") return [0, 0, 0];
    const t = isFinite(cMin) ? (entry - cMin) / cRange : 0.5;
    return colormapLookup(t, colormap);
  };
  const cdataAt = (idx: number) =>
    cdata
      ? mapColor(cdata[Math.min(Math.max(idx, 0), cdata.length - 1)])
      : undefined;
  const css = (c: [number, number, number]) =>
    `rgb(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)})`;

  const faceColor = trace.faceColor ?? "none";
  const edgeColor = trace.edgeColor ?? [0, 0, 0];
  const alpha = trace.faceAlpha ?? 1;
  const lineWidth = trace.lineWidth ?? 0.5;
  const lineStyle = trace.lineStyle ?? "-";

  ctx.save();
  for (let f = 0; f < faces.length; f++) {
    const face = faces[f];
    if (face.length === 0) continue;
    const pts = face.map(vi => vertices[vi]).filter(Boolean);
    const isClosed = pts.every(p => isFinite(p[0]) && isFinite(p[1]));

    // Resolve this face's fill color.
    let fill: [number, number, number] | null = null;
    if (faceColor === "none") fill = null;
    else if (Array.isArray(faceColor)) fill = faceColor;
    else if (faceColor === "flat") {
      const idx = cdata && cdata.length === faces.length ? f : face[0];
      fill = cdataAt(idx) ?? [0, 0, 0];
    } else if (faceColor === "interp") {
      let r = 0,
        g = 0,
        b = 0,
        n = 0;
      for (const vi of face) {
        const col = cdataAt(vi);
        if (col) {
          r += col[0];
          g += col[1];
          b += col[2];
          n++;
        }
      }
      fill = n ? [r / n, g / n, b / n] : [0, 0, 0];
    }

    // Fill (closed polygons only).
    if (fill && isClosed) {
      ctx.globalAlpha = alpha;
      ctx.fillStyle = css(fill);
      ctx.beginPath();
      pts.forEach((p, i) => {
        const cx = toCanvasX(p[0]);
        const cy = toCanvasY(p[1]);
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      });
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    // Edges.
    if (edgeColor !== "none" && lineStyle !== "none") {
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(getLineDash(lineStyle));
      if (Array.isArray(edgeColor)) {
        ctx.strokeStyle = css(edgeColor);
        ctx.beginPath();
        let pen = false;
        for (const vi of face) {
          const p = vertices[vi];
          if (!p || !isFinite(p[0]) || !isFinite(p[1])) {
            pen = false;
            continue;
          }
          const cx = toCanvasX(p[0]);
          const cy = toCanvasY(p[1]);
          if (!pen) {
            ctx.moveTo(cx, cy);
            pen = true;
          } else ctx.lineTo(cx, cy);
        }
        if (isClosed) ctx.closePath();
        ctx.stroke();
      } else {
        // 'flat' / 'interp': color each edge segment from per-vertex data.
        const nSeg = isClosed ? face.length : face.length - 1;
        for (let i = 0; i < nSeg; i++) {
          const a = vertices[face[i]];
          const b = vertices[face[(i + 1) % face.length]];
          if (!a || !b) continue;
          if (!isFinite(a[0]) || !isFinite(a[1])) continue;
          if (!isFinite(b[0]) || !isFinite(b[1])) continue;
          const ca = cdataAt(face[i]) ?? [0, 0, 0];
          const ax = toCanvasX(a[0]);
          const ay = toCanvasY(a[1]);
          const bx = toCanvasX(b[0]);
          const by = toCanvasY(b[1]);
          if (edgeColor === "interp") {
            const cb = cdataAt(face[(i + 1) % face.length]) ?? ca;
            const grad = ctx.createLinearGradient(ax, ay, bx, by);
            grad.addColorStop(0, css(ca));
            grad.addColorStop(1, css(cb));
            ctx.strokeStyle = grad;
          } else {
            ctx.strokeStyle = css(ca);
          }
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    }

    // Markers at each finite vertex.
    if (trace.marker && trace.marker !== "none") {
      const mfc =
        trace.markerFaceColor === "flat"
          ? (fill ?? undefined)
          : Array.isArray(trace.markerFaceColor)
            ? trace.markerFaceColor
            : undefined;
      const markerTrace: PlotTrace = {
        x: pts.map(p => p[0]),
        y: pts.map(p => p[1]),
        marker: trace.marker,
        ...(mfc ? { markerFaceColor: mfc } : {}),
      };
      const dflt = Array.isArray(edgeColor) ? css(edgeColor) : "rgb(0,0,0)";
      drawMarkers(ctx, markerTrace, toCanvasX, toCanvasY, dflt);
    }
  }
  ctx.restore();
}

export const patchRenderer: DataRenderer<"patchTraces"> = {
  layer: "data",
  draw: ({ ctx, cmap, axes }, traces, { toCanvasX, toCanvasY }) => {
    for (const pt of traces)
      drawPatch(ctx, pt, toCanvasX, toCanvasY, cmap, axes.caxis);
  },
  colorValues: (traces, consume) => {
    for (const pt of traces) {
      if (pt.faceVertexCData) {
        for (const cv of pt.faceVertexCData) {
          if (typeof cv === "number") consume(cv);
        }
      }
    }
  },
};

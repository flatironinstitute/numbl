import { Component, useRef, useEffect, useCallback } from "react";
import type { ErrorInfo, ReactNode } from "react";
import type { PlotTrace, PatchTrace } from "./types.js";
import type { AxesState, FigureState } from "./figuresReducer.js";
import { SurfView } from "./SurfView.js";
import { drawPlot } from "./drawPlot.js";
import { buildUihtmlSrcDoc } from "./uihtmlSrcDoc.js";

class AxesErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Axes render error:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 8,
            color: "#a00",
            fontFamily: "sans-serif",
            fontSize: 12,
            textAlign: "center",
            boxSizing: "border-box",
          }}
        >
          Plot render error: {this.state.error.message}
        </div>
      );
    }
    return this.props.children;
  }
}

interface FigureViewProps {
  figure: FigureState;
}

export function FigureView({ figure }: FigureViewProps) {
  const { subplotGrid, sgtitle, axes } = figure;

  // An HTML UI component (MATLAB `uihtml`) renders its self-contained HTML in
  // an iframe, taking precedence over the axes/trace canvas. A srcdoc document
  // needs no server; key by id to remount when HTMLSource/Data changes. The
  // srcdoc embeds the `htmlComponent` data bridge (see buildUihtmlSrcDoc).
  if (figure.uihtml) {
    return (
      <iframe
        key={figure.uihtml.id}
        title={`uihtml-${figure.uihtml.id}`}
        srcDoc={buildUihtmlSrcDoc(
          figure.uihtml.html,
          figure.uihtml.data,
          figure.uihtml.id
        )}
        style={{
          width: "100%",
          height: "100%",
          border: "none",
          display: "block",
        }}
      />
    );
  }

  const axesIndices = Object.keys(axes)
    .map(Number)
    .sort((a, b) => a - b);

  if (axesIndices.length === 0) return null;

  // No subplots: render single axes
  if (!subplotGrid) {
    const ax = axes[axesIndices[0]];
    if (!ax) return null;
    return (
      <AxesErrorBoundary>
        <SingleAxesView axes={ax} />
      </AxesErrorBoundary>
    );
  }

  // Subplot grid layout
  const { rows, cols } = subplotGrid;

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {sgtitle && (
        <div
          style={{
            textAlign: "center",
            fontWeight: "bold",
            fontSize: 16,
            fontFamily: "sans-serif",
            padding: "8px 0 4px 0",
            flexShrink: 0,
          }}
        >
          {sgtitle}
        </div>
      )}
      <div
        style={{
          flex: 1,
          display: "grid",
          gridTemplateRows: `repeat(${rows}, 1fr)`,
          gridTemplateColumns: `repeat(${cols}, 1fr)`,
          gap: 4,
          padding: 4,
          minHeight: 0,
        }}
      >
        {Array.from({ length: rows * cols }, (_, i) => {
          const idx = i + 1; // 1-based
          const ax = axes[idx];
          const row = Math.floor(i / cols);
          const col = i % cols;
          return (
            <div
              key={idx}
              style={{
                gridRow: row + 1,
                gridColumn: col + 1,
                minWidth: 0,
                minHeight: 0,
                overflow: "hidden",
              }}
            >
              {ax ? (
                <AxesErrorBoundary>
                  <SingleAxesView axes={ax} />
                </AxesErrorBoundary>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** True when a 3-D patch has a non-trivial z extent. A flat patch (all z
 *  equal, including the 2-D case where z defaults to 0) renders correctly in
 *  the 2-D canvas, so it should not force the 3-D viewer. */
function hasZVariation(patch: PatchTrace): boolean {
  if (!patch.is3D) return false;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of patch.vertices) {
    const z = v[2] ?? 0;
    if (z < lo) lo = z;
    if (z > hi) hi = z;
  }
  return hi - lo > 1e-9;
}

function SingleAxesView({ axes }: { axes: AxesState }) {
  // A patch with varying z-coordinates (e.g. trimesh(T,x,y,z)) makes the axes
  // 3-D, the same as a surf/plot3/bar3 trace — otherwise it renders flattened
  // in the 2-D canvas, dropping z. A flat patch (z all equal, e.g. simpplot's
  // trimesh(...,0*x,...) under view(2)) stays in the 2-D canvas, which the
  // 3-D viewer's free-orbit camera cannot reproduce.
  const patch3D = (axes.patchTraces ?? []).filter(hasZVariation);
  const has3D =
    (axes.surfTraces && axes.surfTraces.length > 0) ||
    (axes.plot3Traces && axes.plot3Traces.length > 0) ||
    (axes.bar3Traces && axes.bar3Traces.length > 0) ||
    (axes.bar3hTraces && axes.bar3hTraces.length > 0) ||
    (axes.quiver3Traces && axes.quiver3Traces.length > 0) ||
    patch3D.length > 0;

  if (has3D) {
    return (
      <SurfView
        surfTraces={axes.surfTraces ?? []}
        plot3Traces={axes.plot3Traces ?? []}
        bar3Traces={axes.bar3Traces ?? []}
        bar3hTraces={axes.bar3hTraces ?? []}
        quiver3Traces={axes.quiver3Traces ?? []}
        patchTraces={axes.patchTraces ?? []}
        shading={axes.shading}
        colorbar={axes.colorbar}
        colorbarLocation={axes.colorbarLocation}
        colormap={axes.colormap}
        axisVisible={axes.axisVisible}
      />
    );
  }

  return (
    <PlotCanvas
      traces={axes.traces}
      title={axes.title}
      xlabel={axes.xlabel}
      ylabel={axes.ylabel}
      legend={axes.legend}
      gridOn={axes.gridOn}
      boxOn={axes.boxOn}
      imagescTrace={axes.imagescTrace}
      pcolorTraces={axes.pcolorTraces}
      contourTraces={axes.contourTraces}
      colormap={axes.colormap}
      colormapData={axes.colormapData}
      axisMode={axes.axisMode}
      axisScale={axes.axisScale}
      barTraces={axes.barTraces}
      barhTraces={axes.barhTraces}
      errorBarTraces={axes.errorBarTraces}
      boxTraces={axes.boxTraces}
      pieTrace={axes.pieTrace}
      heatmapTrace={axes.heatmapTrace}
      quiverTraces={axes.quiverTraces}
      areaTraces={axes.areaTraces}
      areaBaseValue={axes.areaBaseValue}
      patchTraces={axes.patchTraces}
      shading={axes.shading}
      colorbar={axes.colorbar}
      colorbarLocation={axes.colorbarLocation}
      caxis={axes.caxis}
      xlim={axes.xlim}
      ylim={axes.ylim}
      yDir={axes.yDir}
      axisVisible={axes.axisVisible}
    />
  );
}

function PlotCanvas({
  traces,
  title,
  xlabel,
  ylabel,
  legend,
  gridOn,
  boxOn,
  imagescTrace,
  pcolorTraces,
  contourTraces,
  colormap,
  colormapData,
  axisMode,
  axisScale,
  barTraces,
  barhTraces,
  errorBarTraces,
  boxTraces,
  pieTrace,
  heatmapTrace,
  quiverTraces,
  areaTraces,
  areaBaseValue,
  patchTraces,
  shading,
  colorbar,
  colorbarLocation,
  caxis,
  xlim,
  ylim,
  yDir,
  axisVisible,
}: {
  traces: PlotTrace[];
  title?: string;
  xlabel?: string;
  ylabel?: string;
  legend?: string[];
  gridOn?: boolean;
  boxOn?: boolean;
  imagescTrace?: AxesState["imagescTrace"];
  pcolorTraces?: AxesState["pcolorTraces"];
  contourTraces?: AxesState["contourTraces"];
  colormap?: string;
  colormapData?: number[][];
  axisMode?: string;
  axisScale?: AxesState["axisScale"];
  barTraces?: AxesState["barTraces"];
  barhTraces?: AxesState["barhTraces"];
  errorBarTraces?: AxesState["errorBarTraces"];
  boxTraces?: AxesState["boxTraces"];
  pieTrace?: AxesState["pieTrace"];
  heatmapTrace?: AxesState["heatmapTrace"];
  quiverTraces?: AxesState["quiverTraces"];
  areaTraces?: AxesState["areaTraces"];
  areaBaseValue?: number;
  patchTraces?: AxesState["patchTraces"];
  shading?: AxesState["shading"];
  colorbar?: boolean;
  colorbarLocation?: string;
  caxis?: [number, number];
  xlim?: AxesState["xlim"];
  ylim?: AxesState["ylim"];
  yDir?: AxesState["yDir"];
  axisVisible?: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
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
      axisScale,
      barTraces,
      barhTraces,
      errorBarTraces,
      boxTraces,
      pieTrace,
      heatmapTrace,
      areaTraces,
      areaBaseValue,
      pcolorTraces,
      shading,
      colorbar,
      colorbarLocation,
      caxis,
      colormapData,
      quiverTraces,
      xlim,
      ylim,
      yDir,
      axisVisible,
      boxOn,
      patchTraces
    );
  }, [
    traces,
    title,
    xlabel,
    ylabel,
    legend,
    gridOn,
    imagescTrace,
    contourTraces,
    colormap,
    colormapData,
    axisMode,
    axisScale,
    barTraces,
    barhTraces,
    errorBarTraces,
    boxTraces,
    pieTrace,
    heatmapTrace,
    areaTraces,
    areaBaseValue,
    pcolorTraces,
    shading,
    colorbar,
    colorbarLocation,
    caxis,
    quiverTraces,
    xlim,
    ylim,
    yDir,
    axisVisible,
    boxOn,
    patchTraces,
  ]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const observer = new ResizeObserver(() => {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      redraw();
    });
    observer.observe(container);

    return () => observer.disconnect();
  }, [redraw]);

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%" }}>
      <canvas ref={canvasRef} style={{ display: "block" }} />
    </div>
  );
}

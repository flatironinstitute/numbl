import { Component, useRef, useEffect, useCallback } from "react";
import type { ErrorInfo, ReactNode } from "react";
import type { PlotTrace } from "./types.js";
import type { AxesState, FigureState } from "./figuresReducer.js";
import { SurfView } from "./SurfView.js";
import { drawPlot } from "./drawPlot.js";

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

function SingleAxesView({ axes }: { axes: AxesState }) {
  const has3D =
    (axes.surfTraces && axes.surfTraces.length > 0) ||
    (axes.plot3Traces && axes.plot3Traces.length > 0) ||
    (axes.bar3Traces && axes.bar3Traces.length > 0) ||
    (axes.bar3hTraces && axes.bar3hTraces.length > 0);

  if (has3D) {
    return (
      <SurfView
        surfTraces={axes.surfTraces ?? []}
        plot3Traces={axes.plot3Traces ?? []}
        bar3Traces={axes.bar3Traces ?? []}
        bar3hTraces={axes.bar3hTraces ?? []}
        shading={axes.shading}
        colorbar={axes.colorbar}
        colorbarLocation={axes.colorbarLocation}
        colormap={axes.colormap}
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
      areaTraces={axes.areaTraces}
      areaBaseValue={axes.areaBaseValue}
      shading={axes.shading}
      colorbar={axes.colorbar}
      colorbarLocation={axes.colorbarLocation}
      caxis={axes.caxis}
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
  areaTraces,
  areaBaseValue,
  shading,
  colorbar,
  colorbarLocation,
  caxis,
}: {
  traces: PlotTrace[];
  title?: string;
  xlabel?: string;
  ylabel?: string;
  legend?: string[];
  gridOn?: boolean;
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
  areaTraces?: AxesState["areaTraces"];
  areaBaseValue?: number;
  shading?: AxesState["shading"];
  colorbar?: boolean;
  colorbarLocation?: string;
  caxis?: [number, number];
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
      colormapData
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

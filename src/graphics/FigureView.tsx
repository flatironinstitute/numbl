import { useRef, useEffect, useCallback } from "react";
import type { PlotTrace } from "./types.js";
import type { AxesState, FigureState } from "./figuresReducer.js";
import { SurfView } from "./SurfView.js";
import { drawPlot } from "./drawPlot.js";

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
    return <SingleAxesView axes={ax} />;
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
              {ax ? <SingleAxesView axes={ax} /> : null}
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
    (axes.plot3Traces && axes.plot3Traces.length > 0);

  if (has3D) {
    return (
      <SurfView
        surfTraces={axes.surfTraces ?? []}
        plot3Traces={axes.plot3Traces ?? []}
        shading={axes.shading}
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
      contourTraces={axes.contourTraces}
      colormap={axes.colormap}
      axisMode={axes.axisMode}
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
  contourTraces,
  colormap,
  axisMode,
}: {
  traces: PlotTrace[];
  title?: string;
  xlabel?: string;
  ylabel?: string;
  legend?: string[];
  gridOn?: boolean;
  imagescTrace?: AxesState["imagescTrace"];
  contourTraces?: AxesState["contourTraces"];
  colormap?: string;
  axisMode?: string;
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
      axisMode
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
    axisMode,
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

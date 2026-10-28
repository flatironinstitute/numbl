import { Component, useRef, useEffect, useCallback } from "react";
import type { ErrorInfo, ReactNode } from "react";
import type {
  AxesState,
  FigureState,
} from "../numbl-core/plotting/figuresReducer.js";
import { SurfView } from "./SurfView.js";
import { drawPlot } from "./drawPlot.js";
import { axesIs3D } from "../numbl-core/plotting/axisLimits.js";
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

function SingleAxesView({ axes }: { axes: AxesState }) {
  // Axes with 3-D content (surf, plot3, bar3, quiver3, or a patch with
  // varying z such as trimesh(T,x,y,z)) go to the 3-D viewer. A flat patch
  // (z all equal, e.g. simpplot's trimesh(...,0*x,...) under view(2)) stays in
  // the 2-D canvas, which the 3-D viewer's free-orbit camera cannot reproduce.
  if (axesIs3D(axes)) {
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
        colormapData={axes.colormapData}
        axisVisible={axes.axisVisible}
      />
    );
  }

  return <PlotCanvas axes={axes} />;
}

function PlotCanvas({ axes }: { axes: AxesState }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawPlot(canvas, axes);
  }, [axes]);
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

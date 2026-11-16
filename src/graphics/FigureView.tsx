import {
  Component,
  useRef,
  useEffect,
  useCallback,
  useMemo,
  useState,
} from "react";
import type { ErrorInfo, ReactNode } from "react";
import type {
  AxesState,
  FigureState,
} from "../numbl-core/plotting/figuresReducer.js";
import { SurfView } from "./SurfView.js";
import { drawPlot } from "./drawPlot.js";
import { axesIs3D } from "../numbl-core/plotting/axisLimits.js";
import { buildUihtmlSrcDoc } from "./uihtmlSrcDoc.js";
import {
  applyView,
  colorbarFraction,
  inRect,
  plotFractions,
  shiftRange,
  valueAt,
  wheelFactor,
  wheelPixels,
  zoomRange,
  type PlotLayout,
  type ViewOverride,
} from "./plotInteraction.js";

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
  /** Let the user pan, zoom and change color limits on 2-D axes with the
   *  mouse and keyboard. Off by default, since it captures the scroll wheel
   *  over the plot. */
  interactive?: boolean;
}

export function FigureView({ figure, interactive = false }: FigureViewProps) {
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
        <SingleAxesView axes={ax} interactive={interactive} />
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
                  <SingleAxesView axes={ax} interactive={interactive} />
                </AxesErrorBoundary>
              ) : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SingleAxesView({
  axes,
  interactive,
}: {
  axes: AxesState;
  interactive: boolean;
}) {
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

  return <PlotCanvas axes={axes} interactive={interactive} />;
}

const HELP =
  "Scroll: zoom (Shift: x only, Ctrl: y only). Drag: pan. " +
  "Scroll or drag on the colorbar: color limits. Double-click: reset.\n" +
  "Keys (click the plot first): arrows pan, + - zoom, [ ] contrast, " +
  "{ } shift colors, r reset.";

function PlotCanvas({
  axes,
  interactive,
}: {
  axes: AxesState;
  interactive: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  // The user's pan / zoom / color limits. Kept apart from `axes` so that
  // they persist while new data arrives, until reset.
  const [view, setView] = useState<ViewOverride | null>(null);
  const layoutRef = useRef<PlotLayout | undefined>(undefined);
  const drawn = useMemo(() => applyView(axes, view), [axes, view]);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    layoutRef.current = drawPlot(canvas, drawn);
  }, [drawn]);
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

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!interactive || !canvas) return;
    return attachInteraction(canvas, layoutRef, setView);
  }, [interactive]);

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", position: "relative" }}
    >
      <canvas
        ref={canvasRef}
        tabIndex={interactive ? 0 : undefined}
        title={interactive ? HELP : undefined}
        style={{ display: "block", outline: "none" }}
      />
      {interactive && view && (
        <button
          onClick={() => setView(null)}
          style={resetButtonStyle}
          title="Back to the automatic limits (or double-click the plot)"
        >
          Reset view
        </button>
      )}
    </div>
  );
}

/** Wire mouse and keyboard pan / zoom / color limits to a plot canvas.
 *  Returns a function that removes the listeners. */
function attachInteraction(
  canvas: HTMLCanvasElement,
  layoutRef: { current: PlotLayout | undefined },
  setView: (f: (v: ViewOverride | null) => ViewOverride | null) => void
): () => void {
  const update = (patch: ViewOverride) =>
    setView(v => ({ ...(v ?? {}), ...patch }));
  const pos = (e: MouseEvent): [number, number] => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  const onColorbar = (L: PlotLayout, px: number, py: number) =>
    !!L.colorbar && !!L.colorRange && inRect(L.colorbar, px, py);

  const onWheel = (e: WheelEvent) => {
    const L = layoutRef.current;
    if (!L) return;
    const [px, py] = pos(e);
    const f = wheelFactor(wheelPixels(e));
    if (onColorbar(L, px, py)) {
      e.preventDefault();
      const range = L.colorRange!;
      const c = valueAt(range, colorbarFraction(L.colorbar!, px, py));
      update({ caxis: zoomRange(range, c, f) });
      return;
    }
    if (!inRect(L.plot, px, py)) return;
    e.preventDefault();
    const { fx, fy } = plotFractions(L, px, py);
    const patch: ViewOverride = {};
    if (!e.ctrlKey)
      patch.xlim = zoomRange(L.x, valueAt(L.x, fx, L.logX), f, L.logX);
    if (!e.shiftKey)
      patch.ylim = zoomRange(L.y, valueAt(L.y, fy, L.logY), f, L.logY);
    update(patch);
  };

  let drag: {
    kind: "plot" | "colorbar";
    px: number;
    py: number;
    L: PlotLayout;
  } | null = null;
  const onDown = (e: PointerEvent) => {
    if (e.button !== 0) return;
    canvas.focus();
    const L = layoutRef.current;
    if (!L) return;
    const [px, py] = pos(e);
    const kind = onColorbar(L, px, py)
      ? "colorbar"
      : inRect(L.plot, px, py)
        ? "plot"
        : null;
    if (!kind) return;
    drag = { kind, px, py, L };
    canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  };
  const onMove = (e: PointerEvent) => {
    if (!drag) return;
    const [px, py] = pos(e);
    const { L } = drag;
    if (drag.kind === "plot") {
      // The data follows the pointer.
      const fx = -(px - drag.px) / L.plot.width;
      const fy = ((py - drag.py) / L.plot.height) * (L.yReversed ? -1 : 1);
      update({
        xlim: shiftRange(L.x, fx, L.logX),
        ylim: shiftRange(L.y, fy, L.logY),
      });
    } else {
      const cb = L.colorbar!;
      const frac = cb.vertical
        ? (py - drag.py) / cb.height
        : -(px - drag.px) / cb.width;
      update({ caxis: shiftRange(L.colorRange!, frac) });
    }
  };
  const onUp = (e: PointerEvent) => {
    if (!drag) return;
    drag = null;
    canvas.releasePointerCapture(e.pointerId);
  };
  const onDblClick = () => setView(() => null);

  const onKey = (e: KeyboardEvent) => {
    const L = layoutRef.current;
    if (!L || e.ctrlKey || e.metaKey || e.altKey) return;
    const up = L.yReversed ? -0.1 : 0.1;
    const zoomBoth = (f: number): ViewOverride => ({
      xlim: zoomRange(L.x, valueAt(L.x, 0.5, L.logX), f, L.logX),
      ylim: zoomRange(L.y, valueAt(L.y, 0.5, L.logY), f, L.logY),
    });
    const c = L.colorRange;
    let patch: ViewOverride | null = null;
    switch (e.key) {
      case "ArrowLeft":
        patch = { xlim: shiftRange(L.x, -0.1, L.logX) };
        break;
      case "ArrowRight":
        patch = { xlim: shiftRange(L.x, 0.1, L.logX) };
        break;
      case "ArrowUp":
        patch = { ylim: shiftRange(L.y, up, L.logY) };
        break;
      case "ArrowDown":
        patch = { ylim: shiftRange(L.y, -up, L.logY) };
        break;
      case "+":
      case "=":
        patch = zoomBoth(0.8);
        break;
      case "-":
      case "_":
        patch = zoomBoth(1.25);
        break;
      case "]":
      case "[":
        if (c)
          patch = {
            caxis: zoomRange(c, valueAt(c, 0.5), e.key === "]" ? 0.8 : 1.25),
          };
        break;
      case "}":
      case "{":
        if (c) patch = { caxis: shiftRange(c, e.key === "}" ? 0.1 : -0.1) };
        break;
      case "r":
      case "0":
        e.preventDefault();
        setView(() => null);
        return;
    }
    if (patch) {
      e.preventDefault();
      update(patch);
    }
  };

  canvas.addEventListener("wheel", onWheel, { passive: false });
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);
  canvas.addEventListener("dblclick", onDblClick);
  canvas.addEventListener("keydown", onKey);
  return () => {
    canvas.removeEventListener("wheel", onWheel);
    canvas.removeEventListener("pointerdown", onDown);
    canvas.removeEventListener("pointermove", onMove);
    canvas.removeEventListener("pointerup", onUp);
    canvas.removeEventListener("pointercancel", onUp);
    canvas.removeEventListener("dblclick", onDblClick);
    canvas.removeEventListener("keydown", onKey);
  };
}

const resetButtonStyle: React.CSSProperties = {
  position: "absolute",
  top: 4,
  right: 4,
  padding: "2px 8px",
  border: "1px solid #ccc",
  borderRadius: 4,
  background: "rgba(255, 255, 255, 0.9)",
  fontFamily: "sans-serif",
  fontSize: 11,
  cursor: "pointer",
};

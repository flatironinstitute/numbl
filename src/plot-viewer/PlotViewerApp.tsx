import { useReducer, useEffect, useRef, useState, useCallback } from "react";
import type { PlotInstruction } from "../numbl-core/executor/types.js";
import { FigureView } from "../components/FigureView";
import { figuresReducer, initialFiguresState } from "../shared/figuresReducer";

export function PlotViewerApp() {
  const [figures, dispatch] = useReducer(figuresReducer, initialFiguresState);
  const [scriptDone, setScriptDone] = useState(false);
  const [activeFigure, setActiveFigure] = useState(1);
  const activeFigureRef = useRef(activeFigure);

  const handlePlotInstruction = useCallback((instruction: PlotInstruction) => {
    switch (instruction.type) {
      case "set_figure_handle":
        dispatch({ type: "set_current_handle", handle: instruction.handle });
        // Auto-switch to the figure that was just selected
        activeFigureRef.current = instruction.handle;
        setActiveFigure(instruction.handle);
        break;
      case "set_hold":
        dispatch({ type: "set_hold", value: instruction.value });
        break;
      case "plot":
        dispatch({ type: "add_plot", traces: instruction.traces });
        break;
      case "plot3":
        dispatch({ type: "add_plot3", traces: instruction.traces });
        break;
      case "surf":
        dispatch({ type: "add_surf", trace: instruction.trace });
        break;
      case "close":
        dispatch({ type: "close" });
        break;
      case "close_all":
        dispatch({ type: "close_all" });
        break;
      case "clf":
        dispatch({ type: "clf" });
        break;
    }
  }, []);

  useEffect(() => {
    const evtSource = new EventSource("/events");

    evtSource.onmessage = event => {
      const instructions: PlotInstruction[] = JSON.parse(event.data);
      // JSON.stringify converts NaN/Infinity to null; restore them so the
      // renderer can skip non-finite points instead of drawing them at 0.
      for (const instr of instructions) {
        if (instr.type === "plot") {
          for (const trace of instr.traces) {
            for (let i = 0; i < trace.x.length; i++) {
              if (trace.x[i] === null) (trace.x as number[])[i] = NaN;
            }
            for (let i = 0; i < trace.y.length; i++) {
              if (trace.y[i] === null) (trace.y as number[])[i] = NaN;
            }
          }
        }
        if (instr.type === "plot3") {
          for (const trace of instr.traces) {
            for (let i = 0; i < trace.x.length; i++) {
              if (trace.x[i] === null) (trace.x as number[])[i] = NaN;
            }
            for (let i = 0; i < trace.y.length; i++) {
              if (trace.y[i] === null) (trace.y as number[])[i] = NaN;
            }
            for (let i = 0; i < trace.z.length; i++) {
              if (trace.z[i] === null) (trace.z as number[])[i] = NaN;
            }
          }
        }
        handlePlotInstruction(instr);
      }
    };

    evtSource.addEventListener("done", () => {
      setScriptDone(true);
    });

    return () => evtSource.close();
  }, [handlePlotInstruction]);

  const handles = Object.keys(figures.figs)
    .map(Number)
    .sort((a, b) => a - b);
  const currentFig = figures.figs[activeFigure];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Tab bar */}
      {handles.length > 1 && (
        <div style={tabBarStyle}>
          {handles.map(h => (
            <button
              key={h}
              onClick={() => setActiveFigure(h)}
              style={h === activeFigure ? activeTabStyle : tabStyle}
            >
              Figure {h}
            </button>
          ))}
        </div>
      )}

      {/* Canvas area */}
      <div style={{ flex: 1, position: "relative" }}>
        {currentFig ? (
          <div
            style={{
              position: "absolute",
              inset: 0,
            }}
          >
            <FigureView
              traces={currentFig.traces}
              plot3Traces={currentFig.plot3Traces}
              surfTraces={currentFig.surfTraces}
            />
          </div>
        ) : (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              color: "#999",
              fontFamily: "sans-serif",
            }}
          >
            Waiting for plot data...
          </div>
        )}
      </div>

      {/* Status bar */}
      <div style={statusBarStyle}>
        {scriptDone
          ? "Script complete. Press Ctrl+C in terminal to exit."
          : "Running..."}
      </div>
    </div>
  );
}

const tabBarStyle: React.CSSProperties = {
  display: "flex",
  gap: 2,
  padding: "4px 8px",
  background: "#f0f0f0",
  borderBottom: "1px solid #ccc",
};

const tabStyle: React.CSSProperties = {
  padding: "4px 12px",
  border: "1px solid #ccc",
  borderBottom: "none",
  background: "#e0e0e0",
  cursor: "pointer",
  fontFamily: "sans-serif",
  fontSize: 13,
  borderRadius: "4px 4px 0 0",
};

const activeTabStyle: React.CSSProperties = {
  ...tabStyle,
  background: "#fff",
  fontWeight: "bold",
};

const statusBarStyle: React.CSSProperties = {
  padding: "4px 12px",
  background: "#f5f5f5",
  borderTop: "1px solid #ddd",
  fontSize: 12,
  fontFamily: "sans-serif",
  color: "#666",
};

import { useReducer, useEffect, useRef, useState, useCallback } from "react";
import type { PlotInstruction } from "../graphics/types.js";
import { FigureView } from "../graphics/FigureView.js";
import {
  figuresReducer,
  initialFiguresState,
} from "../graphics/figuresReducer.js";
import { restoreNaNs } from "../graphics/restoreNaNs.js";

export function PlotViewerApp() {
  const [figures, dispatch] = useReducer(figuresReducer, initialFiguresState);
  const [scriptDone, setScriptDone] = useState(false);
  const [activeFigure, setActiveFigure] = useState(1);
  const activeFigureRef = useRef(activeFigure);

  const handlePlotInstruction = useCallback((instruction: PlotInstruction) => {
    dispatch(instruction);

    // Side-effects for active tab tracking
    if (instruction.type === "set_figure_handle") {
      activeFigureRef.current = instruction.handle;
      setActiveFigure(instruction.handle);
    } else if (instruction.type === "close_all") {
      activeFigureRef.current = 1;
      setActiveFigure(1);
    }
  }, []);

  // When the active figure is removed (e.g., by close), fall back to the
  // highest remaining handle so that a valid figure is always displayed.
  const effectiveActiveFigure = (() => {
    if (figures.figs[activeFigure]) return activeFigure;
    const handles = Object.keys(figures.figs).map(Number);
    if (handles.length === 0) return activeFigure;
    const next = handles.sort((a, b) => a - b)[handles.length - 1];
    return next;
  })();

  useEffect(() => {
    const evtSource = new EventSource("/events");

    evtSource.onmessage = event => {
      const instructions: PlotInstruction[] = JSON.parse(event.data);
      // JSON.stringify converts NaN/Infinity to null; restore them so the
      // renderer can skip non-finite points instead of drawing them at 0.
      for (const instr of instructions) {
        restoreNaNs(instr);
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
  const currentFig = figures.figs[effectiveActiveFigure];

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      {/* Tab bar */}
      {handles.length > 1 && (
        <div style={tabBarStyle}>
          {handles.map(h => (
            <button
              key={h}
              onClick={() => setActiveFigure(h)}
              style={h === effectiveActiveFigure ? activeTabStyle : tabStyle}
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
            <FigureView figure={currentFig} />
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

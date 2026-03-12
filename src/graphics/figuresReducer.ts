import type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
} from "../numbl-core/runtime/plotUtils.js";
import type { PlotInstruction } from "../numbl-core/executor/types.js";

export type AxesState = {
  holdOn: boolean;
  traces: PlotTrace[];
  plot3Traces: Plot3Trace[];
  surfTraces: SurfTrace[];
  title?: string;
  xlabel?: string;
  ylabel?: string;
  shading?: "faceted" | "flat" | "interp";
  legend?: string[];
  gridOn?: boolean;
};

export type FigureState = {
  subplotGrid?: { rows: number; cols: number };
  currentAxesIndex: number; // 1-based
  sgtitle?: string;
  axes: { [index: number]: AxesState };
};

export type FiguresState = {
  currentHandle: number;
  figs: {
    [handle: number]: FigureState;
  };
};

const defaultAxes: AxesState = {
  holdOn: false,
  traces: [],
  plot3Traces: [],
  surfTraces: [],
};

function getAxes(fig: FigureState): AxesState {
  return fig.axes[fig.currentAxesIndex] || { ...defaultAxes };
}

function setAxes(fig: FigureState, axes: AxesState): FigureState {
  return {
    ...fig,
    axes: { ...fig.axes, [fig.currentAxesIndex]: axes },
  };
}

export type FiguresStateAction =
  | { type: "set_current_handle"; handle: number }
  | { type: "set_hold"; value: boolean }
  | { type: "add_plot"; traces: PlotTrace[] }
  | { type: "add_plot3"; traces: Plot3Trace[] }
  | { type: "add_surf"; trace: SurfTrace }
  | { type: "close" }
  | { type: "close_all" }
  | { type: "set_title"; text: string }
  | { type: "set_xlabel"; text: string }
  | { type: "set_ylabel"; text: string }
  | { type: "set_shading"; shading: "faceted" | "flat" | "interp" }
  | { type: "clf" }
  | { type: "clear" }
  | { type: "set_subplot"; rows: number; cols: number; index: number }
  | { type: "set_legend"; labels: string[] }
  | { type: "set_sgtitle"; text: string }
  | { type: "set_grid"; value: boolean };

export const initialFiguresState: FiguresState = {
  currentHandle: 1,
  figs: {},
};

const defaultFigure: FigureState = {
  currentAxesIndex: 1,
  axes: {},
};

function ensureFig(state: FiguresState): FigureState {
  return state.figs[state.currentHandle] || { ...defaultFigure };
}

export const figuresReducer = (
  state: FiguresState,
  action: FiguresStateAction
): FiguresState => {
  switch (action.type) {
    case "set_current_handle":
      return { ...state, currentHandle: action.handle };

    case "set_hold": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            holdOn: action.value,
          }),
        },
      };
    }

    case "add_plot": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            traces: axes.holdOn
              ? [...axes.traces, ...action.traces]
              : [...action.traces],
            plot3Traces: axes.holdOn ? axes.plot3Traces : [],
            surfTraces: axes.holdOn ? axes.surfTraces : [],
          }),
        },
      };
    }

    case "add_plot3": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            traces: axes.holdOn ? axes.traces : [],
            plot3Traces: axes.holdOn
              ? [...axes.plot3Traces, ...action.traces]
              : [...action.traces],
            surfTraces: axes.holdOn ? axes.surfTraces : [],
          }),
        },
      };
    }

    case "add_surf": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            traces: axes.holdOn ? axes.traces : [],
            plot3Traces: axes.holdOn ? axes.plot3Traces : [],
            surfTraces: axes.holdOn
              ? [...axes.surfTraces, action.trace]
              : [action.trace],
          }),
        },
      };
    }

    case "close": {
      const remainingFigs = Object.fromEntries(
        Object.entries(state.figs).filter(
          ([k]) => Number(k) !== state.currentHandle
        )
      ) as typeof state.figs;
      const handles = Object.keys(remainingFigs)
        .map(Number)
        .sort((a, b) => a - b);
      return {
        ...state,
        currentHandle: handles.length > 0 ? handles[handles.length - 1] : 1,
        figs: remainingFigs,
      };
    }

    case "close_all":
      return initialFiguresState;

    case "set_title": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, { ...axes, title: action.text }),
        },
      };
    }

    case "set_xlabel": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, { ...axes, xlabel: action.text }),
        },
      };
    }

    case "set_ylabel": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, { ...axes, ylabel: action.text }),
        },
      };
    }

    case "set_shading": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            shading: action.shading,
          }),
        },
      };
    }

    case "clf": {
      const fig = state.figs[state.currentHandle];
      if (!fig) return state;
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: { ...defaultFigure },
        },
      };
    }

    case "clear":
      return initialFiguresState;

    case "set_subplot": {
      const fig = ensureFig(state);
      const newFig: FigureState = {
        ...fig,
        subplotGrid: { rows: action.rows, cols: action.cols },
        currentAxesIndex: action.index,
      };
      // Ensure the axes slot exists
      if (!newFig.axes[action.index]) {
        newFig.axes = { ...newFig.axes, [action.index]: { ...defaultAxes } };
      }
      return {
        ...state,
        figs: { ...state.figs, [state.currentHandle]: newFig },
      };
    }

    case "set_legend": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            legend: action.labels,
          }),
        },
      };
    }

    case "set_sgtitle": {
      const fig = ensureFig(state);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: { ...fig, sgtitle: action.text },
        },
      };
    }

    case "set_grid": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            gridOn: action.value,
          }),
        },
      };
    }

    default:
      return state;
  }
};

/**
 * Convert a PlotInstruction into the corresponding FiguresStateAction.
 * Returns `undefined` for instruction types that have no reducer equivalent.
 */
export function plotInstructionToAction(
  instruction: PlotInstruction
): FiguresStateAction | undefined {
  switch (instruction.type) {
    case "set_figure_handle":
      return { type: "set_current_handle", handle: instruction.handle };
    case "set_hold":
      return { type: "set_hold", value: instruction.value };
    case "plot":
      return { type: "add_plot", traces: instruction.traces };
    case "plot3":
      return { type: "add_plot3", traces: instruction.traces };
    case "surf":
      return { type: "add_surf", trace: instruction.trace };
    case "close":
      return { type: "close" };
    case "close_all":
      return { type: "close_all" };
    case "set_title":
      return { type: "set_title", text: instruction.text };
    case "set_xlabel":
      return { type: "set_xlabel", text: instruction.text };
    case "set_ylabel":
      return { type: "set_ylabel", text: instruction.text };
    case "set_shading":
      return { type: "set_shading", shading: instruction.shading };
    case "clf":
      return { type: "clf" };
    case "set_subplot":
      return {
        type: "set_subplot",
        rows: instruction.rows,
        cols: instruction.cols,
        index: instruction.index,
      };
    case "set_legend":
      return { type: "set_legend", labels: instruction.labels };
    case "set_sgtitle":
      return { type: "set_sgtitle", text: instruction.text };
    case "set_grid":
      return { type: "set_grid", value: instruction.value };
    default:
      return undefined;
  }
}

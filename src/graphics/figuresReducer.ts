import type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
  ImagescTrace,
  ContourTrace,
} from "../numbl-core/runtime/plotUtils.js";
import type { PlotInstruction } from "../numbl-core/executor/types.js";

export type AxesState = {
  holdOn: boolean;
  traces: PlotTrace[];
  plot3Traces: Plot3Trace[];
  surfTraces: SurfTrace[];
  imagescTrace?: ImagescTrace;
  contourTraces: ContourTrace[];
  title?: string;
  xlabel?: string;
  ylabel?: string;
  zlabel?: string;
  shading?: "faceted" | "flat" | "interp";
  legend?: string[];
  gridOn?: boolean;
  colorbar?: boolean;
  colormap?: string;
  view?: { az: number; el: number };
  axisMode?: string;
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
  contourTraces: [],
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
  | { type: "add_imagesc"; trace: ImagescTrace }
  | { type: "add_contour"; trace: ContourTrace }
  | { type: "add_mesh"; trace: SurfTrace }
  | { type: "close" }
  | { type: "close_all" }
  | { type: "set_title"; text: string }
  | { type: "set_xlabel"; text: string }
  | { type: "set_ylabel"; text: string }
  | { type: "set_zlabel"; text: string }
  | { type: "set_shading"; shading: "faceted" | "flat" | "interp" }
  | { type: "clf" }
  | { type: "clear" }
  | { type: "set_subplot"; rows: number; cols: number; index: number }
  | { type: "set_legend"; labels: string[] }
  | { type: "set_sgtitle"; text: string }
  | { type: "set_grid"; value: boolean }
  | { type: "set_colorbar"; value: string }
  | { type: "set_colormap"; name: string }
  | { type: "set_view"; az: number; el: number }
  | { type: "set_axis"; value: string };

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

    case "add_imagesc": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            imagescTrace: action.trace,
            traces: axes.holdOn ? axes.traces : [],
            plot3Traces: axes.holdOn ? axes.plot3Traces : [],
            surfTraces: axes.holdOn ? axes.surfTraces : [],
            contourTraces: axes.holdOn ? axes.contourTraces : [],
          }),
        },
      };
    }

    case "add_contour": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            contourTraces: axes.holdOn
              ? [...axes.contourTraces, action.trace]
              : [action.trace],
            traces: axes.holdOn ? axes.traces : [],
            plot3Traces: axes.holdOn ? axes.plot3Traces : [],
            surfTraces: axes.holdOn ? axes.surfTraces : [],
          }),
        },
      };
    }

    case "add_mesh": {
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
            contourTraces: axes.holdOn ? axes.contourTraces : [],
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

    case "set_zlabel": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            zlabel: action.text,
          }),
        },
      };
    }

    case "set_colorbar": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            colorbar: action.value !== "off",
          }),
        },
      };
    }

    case "set_colormap": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            colormap: action.name,
          }),
        },
      };
    }

    case "set_view": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            view: { az: action.az, el: action.el },
          }),
        },
      };
    }

    case "set_axis": {
      const fig = ensureFig(state);
      const axes = getAxes(fig);
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: setAxes(fig, {
            ...axes,
            axisMode: action.value,
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
    case "imagesc":
      return { type: "add_imagesc", trace: instruction.trace };
    case "contour":
      return { type: "add_contour", trace: instruction.trace };
    case "mesh":
      return { type: "add_mesh", trace: instruction.trace };
    case "set_zlabel":
      return { type: "set_zlabel", text: instruction.text };
    case "set_colorbar":
      return { type: "set_colorbar", value: instruction.value };
    case "set_colormap":
      return { type: "set_colormap", name: instruction.name };
    case "set_view":
      return { type: "set_view", az: instruction.az, el: instruction.el };
    case "set_axis":
      return { type: "set_axis", value: instruction.value };
    default:
      return undefined;
  }
}

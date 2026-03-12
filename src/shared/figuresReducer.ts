import type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
} from "../numbl-core/runtime/plotUtils.js";
import type { PlotInstruction } from "../numbl-core/executor/types.js";

export type FiguresState = {
  currentHandle: number;
  figs: {
    [handle: number]: {
      holdOn: boolean;
      traces: PlotTrace[];
      plot3Traces: Plot3Trace[];
      surfTraces: SurfTrace[];
      title?: string;
      xlabel?: string;
      ylabel?: string;
      shading?: "faceted" | "flat" | "interp";
    };
  };
};

export type FiguresStateAction =
  | {
      type: "set_current_handle";
      handle: number;
    }
  | {
      type: "set_hold";
      value: boolean;
    }
  | {
      type: "add_plot";
      traces: PlotTrace[];
    }
  | {
      type: "add_plot3";
      traces: Plot3Trace[];
    }
  | {
      type: "add_surf";
      trace: SurfTrace;
    }
  | {
      type: "close";
    }
  | {
      type: "close_all";
    }
  | {
      type: "set_title";
      text: string;
    }
  | {
      type: "set_xlabel";
      text: string;
    }
  | {
      type: "set_ylabel";
      text: string;
    }
  | {
      type: "set_shading";
      shading: "faceted" | "flat" | "interp";
    }
  | {
      type: "clf";
    }
  | {
      type: "clear";
    };

export const initialFiguresState: FiguresState = {
  currentHandle: 1,
  figs: {},
};

export const figuresReducer = (
  state: FiguresState,
  action: FiguresStateAction
): FiguresState => {
  switch (action.type) {
    case "set_current_handle":
      return { ...state, currentHandle: action.handle };
    case "set_hold": {
      const currentFig = state.figs[state.currentHandle] || {
        holdOn: false,
        traces: [],
        plot3Traces: [],
        surfTraces: [],
      };
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: {
            ...currentFig,
            holdOn: action.value,
          },
        },
      };
    }
    case "add_plot": {
      const currentFig = state.figs[state.currentHandle] || {
        holdOn: false,
        traces: [],
        plot3Traces: [],
        surfTraces: [],
      };
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: {
            ...currentFig,
            traces: currentFig.holdOn
              ? [...currentFig.traces, ...action.traces]
              : [...action.traces],
            plot3Traces: currentFig.holdOn ? currentFig.plot3Traces : [],
            surfTraces: currentFig.holdOn ? currentFig.surfTraces : [],
          },
        },
      };
    }
    case "add_plot3": {
      const currentFig = state.figs[state.currentHandle] || {
        holdOn: false,
        traces: [],
        plot3Traces: [],
        surfTraces: [],
      };
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: {
            ...currentFig,
            traces: currentFig.holdOn ? currentFig.traces : [],
            plot3Traces: currentFig.holdOn
              ? [...currentFig.plot3Traces, ...action.traces]
              : [...action.traces],
            surfTraces: currentFig.holdOn ? currentFig.surfTraces : [],
          },
        },
      };
    }
    case "add_surf": {
      const currentFig = state.figs[state.currentHandle] || {
        holdOn: false,
        traces: [],
        plot3Traces: [],
        surfTraces: [],
      };
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: {
            ...currentFig,
            traces: currentFig.holdOn ? currentFig.traces : [],
            plot3Traces: currentFig.holdOn ? currentFig.plot3Traces : [],
            surfTraces: currentFig.holdOn
              ? [...currentFig.surfTraces, action.trace]
              : [action.trace],
          },
        },
      };
    }
    case "close": {
      // Close the current figure (remove it from figs)
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
    case "close_all": {
      // Close all figures
      return initialFiguresState;
    }
    case "set_title": {
      const currentFig = state.figs[state.currentHandle] || {
        holdOn: false,
        traces: [],
        plot3Traces: [],
        surfTraces: [],
      };
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: { ...currentFig, title: action.text },
        },
      };
    }
    case "set_xlabel": {
      const currentFig = state.figs[state.currentHandle] || {
        holdOn: false,
        traces: [],
        plot3Traces: [],
        surfTraces: [],
      };
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: { ...currentFig, xlabel: action.text },
        },
      };
    }
    case "set_ylabel": {
      const currentFig = state.figs[state.currentHandle] || {
        holdOn: false,
        traces: [],
        plot3Traces: [],
        surfTraces: [],
      };
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: { ...currentFig, ylabel: action.text },
        },
      };
    }
    case "set_shading": {
      const currentFig = state.figs[state.currentHandle] || {
        holdOn: false,
        traces: [],
        plot3Traces: [],
        surfTraces: [],
      };
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: { ...currentFig, shading: action.shading },
        },
      };
    }
    case "clf": {
      // Clear the current figure (remove traces but keep the tab)
      const currentFig = state.figs[state.currentHandle];
      if (!currentFig) return state;
      return {
        ...state,
        figs: {
          ...state.figs,
          [state.currentHandle]: {
            ...currentFig,
            traces: [],
            plot3Traces: [],
            surfTraces: [],
            title: undefined,
            xlabel: undefined,
            ylabel: undefined,
            shading: undefined,
          },
        },
      };
    }
    case "clear": {
      return initialFiguresState;
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
    default:
      return undefined;
  }
}

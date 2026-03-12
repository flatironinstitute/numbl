import type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
} from "../numbl-core/runtime/plotUtils.js";

export type FiguresState = {
  currentHandle: number;
  figs: {
    [handle: number]: {
      holdOn: boolean;
      traces: PlotTrace[];
      plot3Traces: Plot3Trace[];
      surfTraces: SurfTrace[];
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
      const { [state.currentHandle]: _removed, ...remainingFigs } = state.figs;
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

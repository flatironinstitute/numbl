import type {
  PlotTrace,
  Plot3Trace,
  SurfTrace,
  ImagescTrace,
  ContourTrace,
  PlotInstruction,
} from "./types.js";

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

/** Actions accepted by the figures reducer: any PlotInstruction, plus "clear" for UI resets. */
export type FiguresAction = PlotInstruction | { type: "clear" };

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

/** Update a single axes property on the current figure. */
function updateAxes(
  state: FiguresState,
  update: Partial<AxesState>
): FiguresState {
  const fig = ensureFig(state);
  const axes = getAxes(fig);
  return {
    ...state,
    figs: {
      ...state.figs,
      [state.currentHandle]: setAxes(fig, { ...axes, ...update }),
    },
  };
}

/** Add traces with hold-state logic: clear other trace types unless holdOn. */
function addTraces(
  state: FiguresState,
  update: Partial<
    Pick<
      AxesState,
      "traces" | "plot3Traces" | "surfTraces" | "imagescTrace" | "contourTraces"
    >
  >
): FiguresState {
  const fig = ensureFig(state);
  const axes = getAxes(fig);
  const hold = axes.holdOn;
  return {
    ...state,
    figs: {
      ...state.figs,
      [state.currentHandle]: setAxes(fig, {
        ...axes,
        traces: update.traces ?? (hold ? axes.traces : []),
        plot3Traces: update.plot3Traces ?? (hold ? axes.plot3Traces : []),
        surfTraces: update.surfTraces ?? (hold ? axes.surfTraces : []),
        contourTraces: update.contourTraces ?? (hold ? axes.contourTraces : []),
        ...(update.imagescTrace !== undefined
          ? { imagescTrace: update.imagescTrace }
          : {}),
      }),
    },
  };
}

export const figuresReducer = (
  state: FiguresState,
  action: FiguresAction
): FiguresState => {
  switch (action.type) {
    case "set_figure_handle":
      return { ...state, currentHandle: action.handle };

    case "set_hold":
      return updateAxes(state, { holdOn: action.value });

    case "plot": {
      const axes = getAxes(ensureFig(state));
      return addTraces(state, {
        traces: axes.holdOn
          ? [...axes.traces, ...action.traces]
          : [...action.traces],
      });
    }

    case "plot3": {
      const axes = getAxes(ensureFig(state));
      return addTraces(state, {
        plot3Traces: axes.holdOn
          ? [...axes.plot3Traces, ...action.traces]
          : [...action.traces],
      });
    }

    case "surf": {
      const axes = getAxes(ensureFig(state));
      return addTraces(state, {
        surfTraces: axes.holdOn
          ? [...axes.surfTraces, action.trace]
          : [action.trace],
      });
    }

    case "imagesc":
      return addTraces(state, { imagescTrace: action.trace });

    case "contour": {
      const axes = getAxes(ensureFig(state));
      return addTraces(state, {
        contourTraces: axes.holdOn
          ? [...axes.contourTraces, action.trace]
          : [action.trace],
      });
    }

    case "mesh": {
      const axes = getAxes(ensureFig(state));
      return addTraces(state, {
        surfTraces: axes.holdOn
          ? [...axes.surfTraces, action.trace]
          : [action.trace],
      });
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
    case "clear":
      return initialFiguresState;

    case "set_title":
      return updateAxes(state, { title: action.text });
    case "set_xlabel":
      return updateAxes(state, { xlabel: action.text });
    case "set_ylabel":
      return updateAxes(state, { ylabel: action.text });
    case "set_zlabel":
      return updateAxes(state, { zlabel: action.text });
    case "set_shading":
      return updateAxes(state, { shading: action.shading });
    case "set_legend":
      return updateAxes(state, { legend: action.labels });
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
    case "set_grid":
      return updateAxes(state, { gridOn: action.value });
    case "set_colorbar":
      return updateAxes(state, { colorbar: action.value !== "off" });
    case "set_colormap":
      return updateAxes(state, { colormap: action.name });
    case "set_view":
      return updateAxes(state, { view: { az: action.az, el: action.el } });
    case "set_axis":
      return updateAxes(state, { axisMode: action.value });

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

    case "set_subplot": {
      const fig = ensureFig(state);
      const newFig: FigureState = {
        ...fig,
        subplotGrid: { rows: action.rows, cols: action.cols },
        currentAxesIndex: action.index,
      };
      if (!newFig.axes[action.index]) {
        newFig.axes = { ...newFig.axes, [action.index]: { ...defaultAxes } };
      }
      return {
        ...state,
        figs: { ...state.figs, [state.currentHandle]: newFig },
      };
    }

    default:
      return state;
  }
};

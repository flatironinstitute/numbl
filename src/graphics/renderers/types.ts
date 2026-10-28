import type { ColormapSpec } from "../../numbl-core/plotting/colormaps.js";
import type {
  TraceField2D,
  TraceValue,
} from "../../numbl-core/plotting/axesTraces.js";
import type { AxesState } from "../../numbl-core/plotting/figuresReducer.js";

/** Data → canvas (CSS pixel) coordinate mapping for the current axes. */
export interface PlotTransform {
  toCanvasX: (v: number) => number;
  toCanvasY: (v: number) => number;
}

/** What every renderer gets besides its own traces. */
export interface RenderContext {
  ctx: CanvasRenderingContext2D;
  /** The whole axes, for shared state such as `caxis`, `shading` or
   *  `areaBaseValue`. */
  axes: AxesState;
  cmap: ColormapSpec;
  dpr: number;
}

/** Draws one trace field inside the shared data axes (ticks, grid, labels,
 *  legend and colorbar are drawn by drawPlot). Called only when the field
 *  has content, with the canvas clipped to the plot area. */
export interface DataRenderer<F extends TraceField2D> {
  layer: "data";
  draw(rc: RenderContext, value: TraceValue<F>, tf: PlotTransform): void;
  /** Feed the scalar values this kind maps through the colormap; they set the
   *  automatic colorbar range. Omit for kinds that do not use the colormap. */
  colorValues?(value: TraceValue<F>, consume: (v: number) => void): void;
}

/** Takes over the whole canvas, with no shared axes (pie chart, heatmap). */
export interface StandaloneRenderer<F extends TraceField2D> {
  layer: "standalone";
  draw(
    rc: RenderContext,
    value: TraceValue<F>,
    width: number,
    height: number
  ): void;
}

export type CanvasRenderer<F extends TraceField2D> =
  | DataRenderer<F>
  | StandaloneRenderer<F>;

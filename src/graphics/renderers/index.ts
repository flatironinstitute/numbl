import type { TraceField2D } from "../../numbl-core/plotting/axesTraces.js";
import type { CanvasRenderer } from "./types.js";
import { pieRenderer } from "./pie.js";
import { heatmapRenderer } from "./heatmap.js";
import { imagescRenderer } from "./imagesc.js";
import { pcolorRenderer } from "./pcolor.js";
import { contourRenderer } from "./contour.js";
import { patchRenderer } from "./patch.js";
import { barRenderer, barhRenderer } from "./bar.js";
import { boxRenderer } from "./box.js";
import { areaRenderer } from "./area.js";
import { errorBarRenderer } from "./errorbar.js";
import { quiverRenderer } from "./quiver.js";
import { linesRenderer } from "./lines.js";

/** The 2-D canvas renderer for every trace field that is not 3-D-only. The
 *  mapped type makes this exhaustive: a new 2-D field in `AxesTraces` without
 *  a renderer here is a compile error.
 *
 *  Order matters. The first standalone renderer with content takes over the
 *  canvas; otherwise the data renderers draw in this order, so later entries
 *  paint over earlier ones (images and filled regions first, lines last). */
export const CANVAS_RENDERERS: {
  [F in TraceField2D]: CanvasRenderer<F>;
} = {
  pieTrace: pieRenderer,
  heatmapTrace: heatmapRenderer,
  imagescTrace: imagescRenderer,
  pcolorTraces: pcolorRenderer,
  contourTraces: contourRenderer,
  patchTraces: patchRenderer,
  barTraces: barRenderer,
  barhTraces: barhRenderer,
  boxTraces: boxRenderer,
  areaTraces: areaRenderer,
  errorBarTraces: errorBarRenderer,
  quiverTraces: quiverRenderer,
  traces: linesRenderer,
};

export const CANVAS_RENDER_ORDER = Object.keys(
  CANVAS_RENDERERS
) as TraceField2D[];

export function canvasRenderer<F extends TraceField2D>(
  field: F
): CanvasRenderer<F> {
  return CANVAS_RENDERERS[field] as CanvasRenderer<F>;
}

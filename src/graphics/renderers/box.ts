import type { BoxTrace } from "../../numbl-core/plotting/types.js";
import type { DataRenderer } from "./types.js";

function drawBoxes(
  ctx: CanvasRenderingContext2D,
  boxTraces: BoxTrace[],
  toCanvasX: (v: number) => number,
  toCanvasY: (v: number) => number
) {
  const defaultColors: [number, number, number][] = [
    [0.0, 0.447, 0.741],
    [0.85, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
  ];
  for (let bi = 0; bi < boxTraces.length; bi++) {
    const bt = boxTraces[bi];
    const [r, g, b] = bt.color ?? defaultColors[bi % defaultColors.length];
    const fillColor = `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},0.3)`;
    const strokeColor = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;

    const cx = toCanvasX(bt.x);
    const halfW = Math.abs(toCanvasX(bt.x + bt.width) - toCanvasX(bt.x)) / 2;

    const yQ1 = toCanvasY(bt.q1);
    const yQ3 = toCanvasY(bt.q3);
    const yMed = toCanvasY(bt.median);
    const yWLo = toCanvasY(bt.whiskerLow);
    const yWHi = toCanvasY(bt.whiskerHigh);

    // Box (Q1 to Q3)
    ctx.fillStyle = fillColor;
    ctx.fillRect(
      cx - halfW,
      Math.min(yQ1, yQ3),
      halfW * 2,
      Math.abs(yQ3 - yQ1)
    );
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(
      cx - halfW,
      Math.min(yQ1, yQ3),
      halfW * 2,
      Math.abs(yQ3 - yQ1)
    );

    // Median line
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - halfW, yMed);
    ctx.lineTo(cx + halfW, yMed);
    ctx.stroke();

    // Whiskers (vertical lines from box to whisker ends)
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 2]);
    ctx.beginPath();
    ctx.moveTo(cx, yQ3);
    ctx.lineTo(cx, yWHi);
    ctx.moveTo(cx, yQ1);
    ctx.lineTo(cx, yWLo);
    ctx.stroke();
    ctx.setLineDash([]);

    // Whisker caps (horizontal lines at whisker ends)
    const capW = halfW * 0.5;
    ctx.beginPath();
    ctx.moveTo(cx - capW, yWHi);
    ctx.lineTo(cx + capW, yWHi);
    ctx.moveTo(cx - capW, yWLo);
    ctx.lineTo(cx + capW, yWLo);
    ctx.stroke();

    // Outliers
    ctx.fillStyle = strokeColor;
    for (const o of bt.outliers) {
      const oy = toCanvasY(o);
      ctx.beginPath();
      ctx.arc(cx, oy, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

export const boxRenderer: DataRenderer<"boxTraces"> = {
  layer: "data",
  draw: ({ ctx }, traces, { toCanvasX, toCanvasY }) =>
    drawBoxes(ctx, traces, toCanvasX, toCanvasY),
};

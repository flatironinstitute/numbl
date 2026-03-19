import type { PlotTrace } from "./types.js";

export const TRACE_COLORS = [
  "#0072BD", // blue
  "#D95319", // red-orange
  "#EDB120", // yellow
  "#7E2F8E", // purple
  "#77AC30", // green
  "#4DBEEE", // cyan
  "#A2142F", // dark red
];

export function rgbToCSS(rgb: [number, number, number]): string {
  return `rgb(${Math.round(rgb[0] * 255)}, ${Math.round(rgb[1] * 255)}, ${Math.round(rgb[2] * 255)})`;
}

export function traceColor(trace: PlotTrace, index: number): string {
  if (trace.color) return rgbToCSS(trace.color);
  return TRACE_COLORS[index % TRACE_COLORS.length];
}

export function getLineDash(style: string | undefined): number[] {
  switch (style) {
    case "--":
      return [8, 4];
    case ":":
      return [2, 4];
    case "-.":
      return [8, 4, 2, 4];
    default:
      return [];
  }
}

export function niceTickStep(range: number, maxTicks: number): number {
  const rough = range / maxTicks;
  const pow = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / pow;
  let step: number;
  if (norm <= 1.5) step = 1;
  else if (norm <= 3.5) step = 2;
  else if (norm <= 7.5) step = 5;
  else step = 10;
  return step * pow;
}

export function generateTicks(
  min: number,
  max: number,
  maxTicks: number
): number[] {
  const step = niceTickStep(max - min, maxTicks);
  const ticks: number[] = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + step * 0.001; v += step) {
    ticks.push(v);
  }
  return ticks;
}

export function formatTick(v: number): string {
  if (Math.abs(v) < 1e-10) return "0";
  if (Math.abs(v) >= 1e4 || (Math.abs(v) < 0.01 && v !== 0))
    return v.toExponential(1);
  return parseFloat(v.toPrecision(6)).toString();
}

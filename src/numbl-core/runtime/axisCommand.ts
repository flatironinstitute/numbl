/**
 * `axis(...)` argument parsing — the single place that turns every `axis`
 * call form into `PlotInstruction`s.
 *
 * Supported forms (evaluated left to right, later options overriding earlier
 * ones):
 *   - a limit vector `[xmin xmax ymin ymax (zmin zmax (cmin cmax))]`, where a
 *     non-finite bound (`inf`/`-inf`) leaves that bound automatic;
 *   - a scalar `0/1` or logical `false/true` toggling axes visibility;
 *   - a style word (`tight`, `padded`, `fill`, `equal`, `image`, `square`,
 *     `vis3d`, `normal`, `tickaligned`);
 *   - a limit mode (`manual`, `auto`, `'auto x'`, `'auto y'`, `'auto z'`,
 *     `'auto xy'`, `'auto xz'`, `'auto yz'`);
 *   - a y-direction (`xy`, `ij`);
 *   - a visibility word (`on`, `off`).
 *
 * A leading axes handle (numbl has no real axes handles) is accepted and
 * skipped, so `axis(ax, ...)` works. The query forms `lim = axis` /
 * `lim = axis(ax)` carry no setting argument; the runtime detects them via
 * the `false` return value and computes the limits separately.
 */

import type { RuntimeValue } from "./types.js";
import {
  isRuntimeTensor,
  isRuntimeNumber,
  isRuntimeLogical,
  isRuntimeDummyHandle,
  isRuntimeGraphicsHandle,
} from "./types.js";
import type { PlotInstruction } from "../../graphics/types.js";
import { toString } from "./convert.js";
import { RuntimeError } from "./error.js";

/** Style words that select aspect-ratio / limit-method behavior. */
const STYLE_WORDS = new Set([
  "tight",
  "padded",
  "fill",
  "equal",
  "image",
  "square",
  "vis3d",
  "normal",
  "tickaligned",
]);

/** Styles that also switch the limit mode back to automatic (so the axes
 *  refit to their data), per MATLAB's documented property changes. */
const STYLE_RESETS_LIMITS = new Set([
  "tight",
  "padded",
  "equal",
  "image",
  "tickaligned",
]);

const AUTO_AXIS_TOKENS: Record<string, ("x" | "y" | "z")[]> = {
  auto: ["x", "y", "z"],
  "auto x": ["x"],
  "auto y": ["y"],
  "auto z": ["z"],
  "auto xy": ["x", "y"],
  "auto yx": ["x", "y"],
  "auto xz": ["x", "z"],
  "auto zx": ["x", "z"],
  "auto yz": ["y", "z"],
  "auto zy": ["y", "z"],
};

function isHandle(v: RuntimeValue): boolean {
  return isRuntimeDummyHandle(v) || isRuntimeGraphicsHandle(v);
}

/** Flatten a numeric/logical argument into a plain number array. */
function numericValues(v: RuntimeValue): number[] | null {
  if (isRuntimeTensor(v)) return Array.from(v.data);
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  return null;
}

/** A non-finite bound (inf/-inf) means "keep the automatic bound". */
function bound(v: number): number | null {
  return Number.isFinite(v) ? v : null;
}

/** Build an `auto`-clear instruction for the named axes. */
function clearLimits(axes: ("x" | "y" | "z")[]): PlotInstruction {
  const instr: PlotInstruction & { type: "set_axis_limits" } = {
    type: "set_axis_limits",
  };
  if (axes.includes("x")) instr.xlim = "auto";
  if (axes.includes("y")) instr.ylim = "auto";
  if (axes.includes("z")) instr.zlim = "auto";
  return instr;
}

/**
 * Process an `axis(...)` call, pushing instructions onto `instructions`.
 *
 * @param freezeLimits  Optional provider of the current limit vector
 *   `[xmin xmax ymin ymax (zmin zmax)]`, used to implement `axis manual`
 *   (freeze the current limits). When omitted, `axis manual` records the
 *   request without concrete values.
 * @returns `true` if any setting was applied. `false` means the call carried
 *   no setting argument (a query, e.g. `lim = axis`).
 */
export function applyAxisCommand(
  args: RuntimeValue[],
  instructions: PlotInstruction[],
  freezeLimits?: () => number[]
): boolean {
  let i = 0;
  while (i < args.length && isHandle(args[i])) i++; // skip leading ax handle

  let applied = false;
  const styles: string[] = [];

  for (; i < args.length; i++) {
    const arg = args[i];
    const nums = numericValues(arg);

    if (nums !== null) {
      applied = true;
      if (nums.length === 1) {
        // Visibility: axis(0)/axis(false) hide, axis(1)/axis(true) show.
        instructions.push({ type: "set_axis_visible", value: nums[0] !== 0 });
        continue;
      }
      if (nums.length === 4 || nums.length === 6 || nums.length === 8) {
        const instr: PlotInstruction & { type: "set_axis_limits" } = {
          type: "set_axis_limits",
          xlim: [bound(nums[0]), bound(nums[1])],
          ylim: [bound(nums[2]), bound(nums[3])],
        };
        if (nums.length >= 6) instr.zlim = [bound(nums[4]), bound(nums[5])];
        instructions.push(instr);
        if (
          nums.length === 8 &&
          Number.isFinite(nums[6]) &&
          Number.isFinite(nums[7])
        ) {
          instructions.push({ type: "set_caxis", limits: [nums[6], nums[7]] });
        }
        continue;
      }
      throw new RuntimeError(
        "axis: limit vector must have 4, 6, or 8 elements"
      );
    }

    // String / char keyword.
    const kw = toString(arg)
      .trim()
      .replace(/^["']|["']$/g, "")
      .toLowerCase();
    applied = true;

    if (STYLE_WORDS.has(kw)) {
      styles.push(kw);
      if (STYLE_RESETS_LIMITS.has(kw)) {
        instructions.push(clearLimits(["x", "y", "z"]));
      }
      continue;
    }
    if (kw === "manual") {
      const lim = freezeLimits?.();
      if (lim && lim.length >= 4) {
        const instr: PlotInstruction & { type: "set_axis_limits" } = {
          type: "set_axis_limits",
          xlim: [lim[0], lim[1]],
          ylim: [lim[2], lim[3]],
        };
        if (lim.length >= 6) instr.zlim = [lim[4], lim[5]];
        instructions.push(instr);
      }
      continue;
    }
    if (kw in AUTO_AXIS_TOKENS) {
      instructions.push(clearLimits(AUTO_AXIS_TOKENS[kw]));
      continue;
    }
    if (kw === "xy" || kw === "ij") {
      instructions.push({
        type: "set_axis_ydir",
        dir: kw === "ij" ? "reverse" : "normal",
      });
      continue;
    }
    if (kw === "on" || kw === "off") {
      instructions.push({ type: "set_axis_visible", value: kw === "on" });
      continue;
    }
    throw new RuntimeError(`axis: unknown option '${kw}'`);
  }

  // Collapse all style words from this call into one set_axis so combinations
  // like `axis equal tight` keep both flags.
  if (styles.length > 0) {
    instructions.push({ type: "set_axis", value: styles.join(" ") });
  }
  return applied;
}

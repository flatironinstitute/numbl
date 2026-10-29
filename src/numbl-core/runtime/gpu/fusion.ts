/**
 * Fusion of element-wise gpuArray operations.
 *
 * On a device that can evaluate a fused expression (`GpuKernels.fused`),
 * an element-wise operation does not run at once: its result is a lazy
 * gpuArray holding the expression (`RuntimeGpuArray.lazy`), and an
 * element-wise operation on a lazy operand extends the expression instead
 * of computing the operand. Reading the buffer (a reduction, `gather`,
 * indexing, any non-element-wise kernel) runs the whole expression as one
 * kernel. At the end of each statement, the lazy arrays still bound to a
 * variable are computed, so a value never stays pending across
 * statements, and the temporaries consumed within the statement are never
 * computed: `y = a .* b + c` is one kernel.
 */

import { RuntimeGpuArray, type ElementClass } from "../types.js";
import type { FusedExpr, FusedRef, GpuDevice } from "./deviceTypes.js";

/** Lazy arrays not yet computed or dropped. */
let pending: RuntimeGpuArray[] = [];

const MAX_INPUTS = 8;
const MAX_SCALARS = 16;
const MAX_NODES = 48;

/** An operand: a gpuArray (lazy or not) or a scalar. */
export type FuseOperand = { gpu: RuntimeGpuArray } | { scalar: number };

/**
 * `op` on `operands` as a lazy gpuArray of `shape` and `cls`. The operands
 * have `n` elements or one. A lazy operand is inlined while the expression
 * stays within the kernel's limits; otherwise it is computed and read as
 * an input.
 */
export function fuse(
  device: GpuDevice,
  op: string,
  operands: FuseOperand[],
  n: number,
  shape: number[],
  cls: ElementClass
): RuntimeGpuArray {
  const e: FusedExpr = { n, inputs: [], scalars: [], nodes: [] };
  const input = (buffer: unknown, m: number): FusedRef => {
    let k = e.inputs.findIndex(x => x.buffer === buffer);
    if (k < 0) {
      k = e.inputs.length;
      e.inputs.push({ buffer, n: m });
    }
    return { in: k };
  };
  const args = operands.map((o): FusedRef => {
    if ("scalar" in o) {
      e.scalars.push(o.scalar);
      return { s: e.scalars.length - 1 };
    }
    const g = o.gpu;
    const m = numelOf(g.shape);
    const c = g.lazy;
    if (
      c &&
      e.inputs.length + c.inputs.length <= MAX_INPUTS &&
      e.scalars.length + c.scalars.length <= MAX_SCALARS - 1 &&
      e.nodes.length + c.nodes.length < MAX_NODES
    ) {
      // Inline the operand's expression, renumbering its references.
      const ins = c.inputs.map(x => input(x.buffer, x.n));
      const s0 = e.scalars.length;
      e.scalars.push(...c.scalars);
      const t0 = e.nodes.length;
      for (const node of c.nodes)
        e.nodes.push({
          op: node.op,
          args: node.args.map(r =>
            "in" in r ? ins[r.in] : "s" in r ? { s: r.s + s0 } : { t: r.t + t0 }
          ),
        });
      return { t: e.nodes.length - 1 };
    }
    return input(g.buffer, m);
  });
  e.nodes.push({ op, args });
  // The expression holds its inputs until it is computed or dropped.
  for (const x of e.inputs) device.retain?.(x.buffer);
  const g = new RuntimeGpuArray(
    device,
    null,
    shape,
    cls,
    false,
    device.generation,
    e
  );
  pending.push(g);
  return g;
}

function numelOf(shape: number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/** At the end of a statement: compute the lazy arrays that a variable (or
 *  any slot other than a statement scope) holds. Arrays held only by a
 *  statement still running (an outer statement waiting on a function
 *  call) stay pending; dropped ones are forgotten. */
export function flushPendingGpu(): void {
  if (pending.length === 0) return;
  const keep: RuntimeGpuArray[] = [];
  for (const g of pending) {
    if (!g.lazy) continue;
    if (g._rc - g._scopeHolds > 0) void g.buffer;
    else if (g._rc > 0) keep.push(g);
  }
  pending = keep;
}

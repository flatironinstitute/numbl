/**
 * Shared infrastructure for scalar comparison builtins
 * (`eq`, `ne`, `lt`, `le`, `gt`, `ge`).
 *
 * Complex scalar handling (MATLAB rule):
 *   - `eq` / `ne` compare both real and imaginary parts.
 *   - `<` / `<=` / `>` / `>=` compare on the real part only; the
 *     imaginary part is dropped. (numbl matches.)
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  isNumeric,
  isScalar,
  isMultiElement,
  scalarLogical,
  type Type,
  type NumericType,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import {
  requireScalarRealOrComplex,
  exactDouble,
  exactScalarAsComplex,
} from "../_shared.js";
import { isTensor, makeTensor } from "../../../runtime/value.js";

export type CompareKind = "eq" | "ne" | "rel";

function isScalarComplex(t: Type): boolean {
  return isNumeric(t) && isScalar(t) && t.isComplex;
}

export function defineCompare(
  name: string,
  cOp: string,
  fold: (a: number, b: number) => boolean,
  kind: CompareKind = "rel"
): Builtin {
  return {
    name,
    transfer(argTypes, nargout) {
      if (argTypes.length !== 2) {
        throw new TypeError(
          `'${name}' expects 2 arg(s), got ${argTypes.length}`
        );
      }
      if (nargout !== 1) {
        throw new UnsupportedConstruct(
          `'${name}' does not support multi-output (nargout=${nargout})`
        );
      }
      // Elementwise tensor comparison: either operand may be a non-
      // scalar real numeric tensor. The result is a logical tensor of
      // the broadcast shape. v1: interpreter-only (the call hook
      // handles it); AOT lowering can still reject by failing to
      // produce a scalar emit for the tensor case — emit hooks below.
      const a = argTypes[0];
      const b = argTypes[1];
      const aTensor = isNumeric(a) && !isScalar(a);
      const bTensor = isNumeric(b) && !isScalar(b);
      if (aTensor || bTensor) {
        if (!isNumeric(a) || !isNumeric(b)) {
          throw new TypeError(
            `'${name}' elementwise: both args must be numeric`
          );
        }
        if (a.isComplex || b.isComplex) {
          throw new UnsupportedConstruct(
            `'${name}' elementwise: complex tensor compare is not ` +
              `supported in mtoc2 yet`
          );
        }
        // Result is a logical tensor shaped like the broadcast — we
        // don't have a static broadcast helper handy, so produce an
        // un-shape-known logical tensor type. The interpreter call
        // hook will compute the actual shape; AOT consumers will
        // surface a shape-unknown error at their emit boundary.
        return [
          {
            kind: "Numeric",
            elem: "logical",
            isComplex: false,
            dims: [{ kind: "unknown" }, { kind: "unknown" }],
            sign: "nonneg",
          } as NumericType,
        ];
      }
      requireScalarRealOrComplex(argTypes[0], `'${name}' arg 1`);
      requireScalarRealOrComplex(argTypes[1], `'${name}' arg 2`);
      if (isScalarComplex(argTypes[0]) || isScalarComplex(argTypes[1])) {
        const ax = exactScalarAsComplex(argTypes[0]);
        const bx = exactScalarAsComplex(argTypes[1]);
        if (ax !== undefined && bx !== undefined) {
          if (kind === "eq") {
            return [scalarLogical(ax.re === bx.re && ax.im === bx.im)];
          }
          if (kind === "ne") {
            return [scalarLogical(ax.re !== bx.re || ax.im !== bx.im)];
          }
          return [scalarLogical(fold(ax.re, bx.re))];
        }
        return [scalarLogical()];
      }
      const ax = exactDouble(argTypes[0]);
      const bx = exactDouble(argTypes[1]);
      if (ax !== undefined && bx !== undefined) {
        return [scalarLogical(fold(ax, bx))];
      }
      return [scalarLogical()];
    },
    emitC({ argsC, argTypes, useRuntime }) {
      // Tensor compare goes through `emitTensorFused` (per-slot path),
      // which calls this hook with scalar-versioned arg types. If we
      // see multi-element types here it means the fused path didn't
      // trigger (shape mismatch / broadcast / shape-unknown result
      // type) and a runtime helper would be needed. Reject explicitly
      // so the JIT bridge falls back to the interpreter cleanly.
      if (isMultiElement(argTypes[0]) || isMultiElement(argTypes[1])) {
        throw new UnsupportedConstruct(
          `'${name}' on tensors that don't share static shape ` +
            `(broadcast / runtime-unknown) is not yet supported by ` +
            `mtoc2's AOT backends; the interpreter handles it`
        );
      }
      const aCx = isScalarComplex(argTypes[0]);
      const bCx = isScalarComplex(argTypes[1]);
      if (aCx || bCx) {
        useRuntime("mtoc2_cscalar");
        // Materialize re/im for the operands; project a real-typed
        // operand to (re, 0) inline so the C expression doesn't have
        // to branch on which side is complex.
        const aReC = aCx ? `mtoc2_creal(${argsC[0]})` : `(${argsC[0]})`;
        const bReC = bCx ? `mtoc2_creal(${argsC[1]})` : `(${argsC[1]})`;
        if (kind === "eq") {
          const aImC = aCx ? `mtoc2_cimag(${argsC[0]})` : "0.0";
          const bImC = bCx ? `mtoc2_cimag(${argsC[1]})` : "0.0";
          return `(${aReC} == ${bReC} && ${aImC} == ${bImC})`;
        }
        if (kind === "ne") {
          const aImC = aCx ? `mtoc2_cimag(${argsC[0]})` : "0.0";
          const bImC = bCx ? `mtoc2_cimag(${argsC[1]})` : "0.0";
          return `(${aReC} != ${bReC} || ${aImC} != ${bImC})`;
        }
        return `(${aReC} ${cOp} ${bReC})`;
      }
      return `(${argsC[0]} ${cOp} ${argsC[1]})`;
    },
    // Scalar logical result emits as a bare JS boolean (mirroring
    // numbl's scalarEmit + interpreter, where RuntimeLogical IS just
    // a JS boolean). Downstream sites that need a numeric value
    // (tensor element writes, %d/%f formatting, etc.) either coerce
    // via JS implicit conversion or explicitly box with Number(v).
    emitJs({ argsJs, argTypes }) {
      if (isMultiElement(argTypes[0]) || isMultiElement(argTypes[1])) {
        throw new UnsupportedConstruct(
          `'${name}' on tensors that don't share static shape ` +
            `(broadcast / runtime-unknown) is not yet supported by ` +
            `mtoc2's AOT backends; the interpreter handles it`
        );
      }
      const aCx = isScalarComplex(argTypes[0]);
      const bCx = isScalarComplex(argTypes[1]);
      if (aCx || bCx) {
        const aRe = aCx ? `${argsJs[0]}.re` : `${argsJs[0]}`;
        const bRe = bCx ? `${argsJs[1]}.re` : `${argsJs[1]}`;
        if (kind === "eq") {
          const aIm = aCx ? `${argsJs[0]}.im` : `0`;
          const bIm = bCx ? `${argsJs[1]}.im` : `0`;
          return `(${aRe} === ${bRe} && ${aIm} === ${bIm})`;
        }
        if (kind === "ne") {
          const aIm = aCx ? `${argsJs[0]}.im` : `0`;
          const bIm = bCx ? `${argsJs[1]}.im` : `0`;
          return `(${aRe} !== ${bRe} || ${aIm} !== ${bIm})`;
        }
        return `(${aRe} ${cOp} ${bRe})`;
      }
      return `(${argsJs[0]} ${cOp} ${argsJs[1]})`;
    },
    call({ args, argTypes }) {
      // Elementwise tensor path: at least one side is a non-scalar
      // tensor. Numbl's `elementwise.ts` broadcasts scalar↔tensor and
      // requires matching shapes for tensor↔tensor; mirror that here.
      // Complex tensor compare is rejected at transfer.
      const aIsTensor = isTensor(args[0]);
      const bIsTensor = isTensor(args[1]);
      if (aIsTensor || bIsTensor) {
        const scalarTo = (v: unknown): number =>
          typeof v === "number"
            ? v
            : typeof v === "boolean"
              ? v
                ? 1
                : 0
              : Number(v);
        const op = (x: number, y: number): number =>
          kind === "eq"
            ? x === y
              ? 1
              : 0
            : kind === "ne"
              ? x !== y
                ? 1
                : 0
              : fold(x, y)
                ? 1
                : 0;
        if (aIsTensor && bIsTensor) {
          const at = args[0] as { data: ArrayLike<number>; shape: number[] };
          const bt = args[1] as { data: ArrayLike<number>; shape: number[] };
          if (at.data.length !== bt.data.length) {
            throw new TypeError(
              `'${name}' elementwise: tensor shapes don't match ` +
                `(${at.shape.join("×")} vs ${bt.shape.join("×")})`
            );
          }
          const data = new Float64Array(at.data.length);
          for (let i = 0; i < at.data.length; i++) {
            data[i] = op(at.data[i], bt.data[i]);
          }
          return [{ ...makeTensor(at.shape.slice(), data), isLogical: true }];
        }
        const t = (aIsTensor ? args[0] : args[1]) as {
          data: ArrayLike<number>;
          shape: number[];
        };
        const s = scalarTo(aIsTensor ? args[1] : args[0]);
        const data = new Float64Array(t.data.length);
        for (let i = 0; i < t.data.length; i++) {
          data[i] = aIsTensor ? op(t.data[i], s) : op(s, t.data[i]);
        }
        return [{ ...makeTensor(t.shape.slice(), data), isLogical: true }];
      }
      const aCx = isScalarComplex(argTypes[0]);
      const bCx = isScalarComplex(argTypes[1]);
      if (aCx || bCx) {
        const av = args[0];
        const bv = args[1];
        const aRe =
          typeof av === "number" ? av : (av as { re: number; im: number }).re;
        const aIm =
          typeof av === "number" ? 0 : (av as { re: number; im: number }).im;
        const bRe =
          typeof bv === "number" ? bv : (bv as { re: number; im: number }).re;
        const bIm =
          typeof bv === "number" ? 0 : (bv as { re: number; im: number }).im;
        if (kind === "eq") return [aRe === bRe && aIm === bIm];
        if (kind === "ne") return [aRe !== bRe || aIm !== bIm];
        return [fold(aRe, bRe)];
      }
      const av = typeof args[0] === "number" ? args[0] : Number(args[0]);
      const bv = typeof args[1] === "number" ? args[1] : Number(args[1]);
      return [fold(av, bv)];
    },
    elementwise: true,
  };
}

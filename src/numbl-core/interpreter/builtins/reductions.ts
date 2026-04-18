/**
 * Interpreter IBuiltins for reduction operations:
 * sum, prod, mean, std, var, median, mode, any, all,
 * cumsum, cumprod, cummax, cummin, diff.
 */

import type { RuntimeValue } from "../../runtime/types.js";
import {
  isRuntimeChar,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { RTV, toNumber, toString, RuntimeError } from "../../runtime/index.js";
import { type JitType, shapeAfterReduction } from "../../jit/jitTypes.js";
import { sparseToDense } from "../../helpers/sparse-arithmetic.js";
import { sparseSum } from "../../helpers/reduction-helpers.js";
import { uninitFloatX } from "../../runtime/alloc.js";
import { defineBuiltin } from "./types.js";
import {
  firstReduceDim,
  accumKernel,
  accumKernelOmitNaN,
  sliceKernel,
  sliceKernelOmitNaN,
  parseNanFlag,
  filterNaN,
  complexProd,
  scanLogical,
  logicalAlongDim,
  type ReductionKernel,
} from "../../helpers/reduction-helpers.js";
import { cumOp, diffOnce } from "../../helpers/reduction/cumulative.js";
import { OpReduce } from "../../ops/index.js";
import { rstr } from "../../runtime/runtime.js";

// ── Type rule helpers ──────────────────────────────────────────────────

/**
 * Parse standard reduction arg types: f(A, [dim|'all'], ['omitnan']).
 * Returns null if the args are ambiguous or unsupported for type analysis.
 */
function parseStdReductionArgs(argTypes: JitType[]): {
  inputType: JitType;
  dimType?: JitType;
  allFlag: boolean;
} | null {
  if (argTypes.length === 0) return null;
  const inputType = argTypes[0];
  if (inputType.kind === "unknown") return null;
  let rest = argTypes.slice(1);

  // Strip trailing char/string only if it's a known nanflag
  if (rest.length > 0) {
    const last = rest[rest.length - 1];
    if (last.kind === "char" || last.kind === "string") {
      const v = last.value?.toLowerCase();
      if (v === "omitnan" || v === "includenan") {
        rest = rest.slice(0, -1);
      } else if (v === undefined) {
        return null; // unknown char value — can't determine semantics
      }
      // Non-nanflag char — handled below as second arg
    }
  }

  if (rest.length === 0) return { inputType, allFlag: false };

  const second = rest[0];
  if (
    (second.kind === "char" || second.kind === "string") &&
    second.value === "all"
  ) {
    return { inputType, allFlag: true };
  }
  if (
    (second.kind === "char" || second.kind === "string") &&
    second.value === undefined
  ) {
    return null; // unknown char — can't tell if it's 'all' or a dim name
  }

  // Numeric or tensor dim arg
  if (
    second.kind === "number" ||
    second.kind === "boolean" ||
    second.kind === "tensor"
  ) {
    return { inputType, dimType: second, allFlag: false };
  }

  return null;
}

/**
 * Compute the output JitType for a reduction that collapses a dimension.
 */
function reductionOutputType(
  inputType: JitType,
  dimType: JitType | undefined,
  allFlag: boolean,
  opts?: { alwaysReal?: boolean; logicalOutput?: boolean }
): JitType[] | null {
  const isComplex =
    inputType.kind === "complex_or_number" ||
    (inputType.kind === "tensor" && inputType.isComplex === true);

  const scalarOut = (): JitType => {
    if (opts?.logicalOutput) return { kind: "boolean" };
    if (opts?.alwaysReal || !isComplex) return { kind: "number" };
    return { kind: "complex_or_number" };
  };

  const tensorOut = (shape?: number[]): JitType => {
    if (opts?.logicalOutput)
      return { kind: "tensor", isComplex: false, isLogical: true, shape };
    return {
      kind: "tensor",
      isComplex,
      ...(shape ? { shape } : {}),
    };
  };

  // Scalar/boolean/complex scalar input → scalar output
  if (
    inputType.kind === "number" ||
    inputType.kind === "boolean" ||
    inputType.kind === "complex_or_number"
  ) {
    return [scalarOut()];
  }

  if (inputType.kind === "sparse_matrix") {
    // Treat sparse like a 2D tensor for reduction type rules
    const shape =
      inputType.m !== undefined && inputType.n !== undefined
        ? [inputType.m, inputType.n]
        : undefined;
    return reductionOutputType(
      { kind: "tensor", isComplex: inputType.isComplex, shape },
      dimType,
      allFlag,
      opts
    );
  }

  if (inputType.kind !== "tensor") return null;

  // 'all' flag → always scalar
  if (allFlag) return [scalarOut()];

  // With explicit dim arg
  if (dimType) {
    const dim =
      dimType.kind === "number" && dimType.exact !== undefined
        ? dimType.exact
        : undefined;
    if (
      inputType.shape &&
      dim !== undefined &&
      dim >= 1 &&
      dim <= inputType.shape.length
    ) {
      const result = shapeAfterReduction(inputType.shape, dim);
      if (result.scalar) return [scalarOut()];
      return [tensorOut(result.shape)];
    }
    // Unknown dim or shape — return tensor with unknown shape
    return [tensorOut()];
  }

  // No dim, no 'all' — default reduction (firstReduceDim)
  if (!inputType.shape) {
    // Unknown shape: conservatively return scalar (like min/max pattern)
    return [scalarOut()];
  }
  const result = shapeAfterReduction(inputType.shape);
  if (result.scalar) return [scalarOut()];
  return [tensorOut(result.shape)];
}

// ── Generic reduction apply helper ─────────────────────────────────────

function reductionApply(
  name: string,
  args: RuntimeValue[],
  kernel: ReductionKernel,
  omitNanKernel?: ReductionKernel
): RuntimeValue {
  const { args: parsedArgs, omitNaN } = parseNanFlag(args);
  const k = omitNaN && omitNanKernel ? omitNanKernel : kernel;
  let v = parsedArgs[0];
  if (isRuntimeSparseMatrix(v)) {
    parsedArgs[0] = sparseToDense(v);
    v = parsedArgs[0];
  }
  if (isRuntimeNumber(v)) return v;
  if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
  if (isRuntimeComplexNumber(v)) return v;
  if (isRuntimeTensor(v)) {
    if (parsedArgs.length >= 2) {
      if (isRuntimeChar(parsedArgs[1]) && toString(parsedArgs[1]) === "all")
        return k.reduceAll(v);
      return k.reduceDim(v, Math.round(toNumber(parsedArgs[1])));
    }
    const d = firstReduceDim(v.shape);
    return d === 0 ? k.reduceAll(v) : k.reduceDim(v, d);
  }
  throw new RuntimeError(`${name}: argument must be numeric`);
}

// ── sum ────────────────────────────────────────────────────────────────

const sumKernel = accumKernel(
  (acc, val) => acc + val,
  0,
  undefined,
  OpReduce.SUM
);
const sumOmitNanKernel = accumKernelOmitNaN((acc, val) => acc + val, 0);

defineBuiltin({
  name: "sum",
  cases: [
    {
      match: argTypes => {
        const parsed = parseStdReductionArgs(argTypes);
        if (!parsed) return null;
        return reductionOutputType(
          parsed.inputType,
          parsed.dimType,
          parsed.allFlag
        );
      },
      apply: args => {
        if (isRuntimeSparseMatrix(args[0])) {
          const v =
            args[0] as import("../../runtime/types.js").RuntimeSparseMatrix;
          const dim =
            args.length >= 2
              ? Math.round(toNumber(args[1]))
              : v.m > 1
                ? 1
                : v.n > 1
                  ? 2
                  : 1;
          return sparseSum(v, dim);
        }
        return reductionApply("sum", args, sumKernel, sumOmitNanKernel);
      },
    },
  ],
  // JIT fast path: scalar is identity; real vector-shaped tensor collapses
  // to a scalar via a straight-line loop in $h.tSum. Matrix/complex/dim-arg
  // fall back to the generic ib_sum.
  jitEmit: (argCode, argTypes) => {
    if (argTypes.length !== 1) return null;
    const a = argTypes[0];
    if (a.kind === "number" || a.kind === "boolean") return argCode[0];
    if (a.kind === "tensor" && a.isComplex !== true) {
      const shape = a.shape;
      const isKnownVector =
        shape !== undefined &&
        (shape.length === 1 ||
          (shape.length === 2 && (shape[0] === 1 || shape[1] === 1)));
      if (isKnownVector) return `$h.tSum(${argCode[0]})`;
    }
    return null;
  },
});

// ── prod ───────────────────────────────────────────────────────────────

defineBuiltin({
  name: "prod",
  cases: [
    {
      match: argTypes => {
        const parsed = parseStdReductionArgs(argTypes);
        if (!parsed) return null;
        return reductionOutputType(
          parsed.inputType,
          parsed.dimType,
          parsed.allFlag
        );
      },
      apply: (args): RuntimeValue => {
        const { args: parsedArgs, omitNaN } = parseNanFlag(args);
        if (isRuntimeSparseMatrix(parsedArgs[0]))
          parsedArgs[0] = sparseToDense(parsedArgs[0]);
        const v = parsedArgs[0];
        if (isRuntimeNumber(v)) return v;
        if (isRuntimeLogical(v)) return RTV.num(v ? 1 : 0);
        if (isRuntimeComplexNumber(v)) return v;
        if (isRuntimeTensor(v)) {
          if (v.imag) {
            const dimArg =
              parsedArgs.length >= 2
                ? Math.round(toNumber(parsedArgs[1]))
                : undefined;
            return complexProd(v, dimArg);
          }
          const kernel = omitNaN
            ? accumKernelOmitNaN((acc, val) => acc * val, 1)
            : accumKernel((acc, val) => acc * val, 1, undefined, OpReduce.PROD);
          if (parsedArgs.length >= 2) {
            if (
              isRuntimeChar(parsedArgs[1]) &&
              toString(parsedArgs[1]) === "all"
            )
              return kernel.reduceAll(v);
            return kernel.reduceDim(v, Math.round(toNumber(parsedArgs[1])));
          }
          const d = firstReduceDim(v.shape);
          return d === 0 ? kernel.reduceAll(v) : kernel.reduceDim(v, d);
        }
        throw new RuntimeError("prod: argument must be numeric");
      },
    },
  ],
});

// ── mean ───────────────────────────────────────────────────────────────

const meanKernel = accumKernel(
  (acc, val) => acc + val,
  0,
  (sum, count) => sum / count,
  OpReduce.SUM
);
const meanOmitNanKernel = accumKernelOmitNaN(
  (acc, val) => acc + val,
  0,
  (sum, count) => (count === 0 ? NaN : sum / count)
);

defineBuiltin({
  name: "mean",
  cases: [
    {
      match: argTypes => {
        const parsed = parseStdReductionArgs(argTypes);
        if (!parsed) return null;
        return reductionOutputType(
          parsed.inputType,
          parsed.dimType,
          parsed.allFlag
        );
      },
      apply: args =>
        reductionApply("mean", args, meanKernel, meanOmitNanKernel),
    },
  ],
});

// ── std / var ──────────────────────────────────────────────────────────

function stdVarMatch(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length === 0) return null;
  const inputType = argTypes[0];
  if (inputType.kind === "unknown") return null;
  let rest = argTypes.slice(1);
  if (rest.length > 0) {
    const last = rest[rest.length - 1];
    if (last.kind === "char" || last.kind === "string") {
      const v = last.value?.toLowerCase();
      if (v === "omitnan" || v === "includenan") {
        rest = rest.slice(0, -1);
      } else if (v === undefined) {
        return null;
      }
    }
  }
  const dimType = rest.length >= 2 ? rest[1] : undefined;
  return reductionOutputType(inputType, dimType, false, { alwaysReal: true });
}

function stdVarApply(
  name: string,
  transform: (variance: number) => number
): (args: RuntimeValue[]) => RuntimeValue {
  return (args): RuntimeValue => {
    const { args: parsedArgs, omitNaN } = parseNanFlag(args);
    const v = parsedArgs[0];
    const w = parsedArgs.length >= 2 ? toNumber(parsedArgs[1]) : 0;
    let dimArg = 0;
    let allFlag = false;
    if (parsedArgs.length >= 3) {
      const d = parsedArgs[2];
      if (
        (isRuntimeChar(d) || isRuntimeString(d)) &&
        toString(d).toLowerCase() === "all"
      ) {
        allFlag = true;
      } else {
        dimArg = Math.round(toNumber(d));
      }
    }
    if (isRuntimeNumber(v)) return 0;
    if (isRuntimeComplexNumber(v)) return 0;
    if (isRuntimeTensor(v)) {
      const varianceOf = (slice: ArrayLike<number>): number => {
        let data: ArrayLike<number> = slice;
        if (omitNaN) data = filterNaN(slice);
        const n = data.length;
        if (n === 0) return NaN;
        if (n <= 1 && w === 0) return 0;
        let s = 0;
        for (let i = 0; i < n; i++) s += (data as number[])[i];
        const m = s / n;
        let ss = 0;
        for (let i = 0; i < n; i++) ss += ((data as number[])[i] - m) ** 2;
        return ss / (w === 1 ? n : n - 1);
      };
      const kernel = sliceKernel((s: ArrayLike<number>) =>
        transform(varianceOf(s))
      );
      if (allFlag) return kernel.reduceAll(v);
      if (dimArg > 0) return kernel.reduceDim(v, dimArg);
      const d = firstReduceDim(v.shape);
      return d === 0 ? kernel.reduceAll(v) : kernel.reduceDim(v, d);
    }
    throw new RuntimeError(`${name}: argument must be numeric`);
  };
}

defineBuiltin({
  name: "std",
  cases: [{ match: stdVarMatch, apply: stdVarApply("std", v => Math.sqrt(v)) }],
});

defineBuiltin({
  name: "var",
  cases: [{ match: stdVarMatch, apply: stdVarApply("var", v => v) }],
});

// ── median ─────────────────────────────────────────────────────────────

function medianOf(arr: ArrayLike<number>): number {
  const sorted = Array.from(arr).sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n % 2 === 1) return sorted[(n - 1) / 2];
  return (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

const medianKernel = sliceKernel(medianOf);
const medianOmitNanKernel = sliceKernelOmitNaN(medianOf);

defineBuiltin({
  name: "median",
  cases: [
    {
      match: argTypes => {
        const parsed = parseStdReductionArgs(argTypes);
        if (!parsed) return null;
        return reductionOutputType(
          parsed.inputType,
          parsed.dimType,
          parsed.allFlag,
          { alwaysReal: true }
        );
      },
      apply: args =>
        reductionApply("median", args, medianKernel, medianOmitNanKernel),
    },
  ],
});

// ── mode ───────────────────────────────────────────────────────────────

function modeOf(arr: ArrayLike<number>): number {
  const counts = new Map<number, number>();
  for (let i = 0; i < arr.length; i++) {
    counts.set(arr[i], (counts.get(arr[i]) ?? 0) + 1);
  }
  let bestVal = (arr as number[])[0],
    bestCount = 0;
  for (const [val, count] of counts) {
    if (count > bestCount || (count === bestCount && val < bestVal)) {
      bestVal = val;
      bestCount = count;
    }
  }
  return bestVal;
}

const modeKernel = sliceKernel(modeOf);
const modeOmitNanKernel = sliceKernelOmitNaN(modeOf);

defineBuiltin({
  name: "mode",
  cases: [
    {
      match: argTypes => {
        const parsed = parseStdReductionArgs(argTypes);
        if (!parsed) return null;
        return reductionOutputType(
          parsed.inputType,
          parsed.dimType,
          parsed.allFlag,
          { alwaysReal: true }
        );
      },
      apply: args =>
        reductionApply("mode", args, modeKernel, modeOmitNanKernel),
    },
  ],
});

// ── any / all ──────────────────────────────────────────────────────────

function anyAllApply(name: string, mode: "any" | "all") {
  return (args: RuntimeValue[]): RuntimeValue => {
    if (isRuntimeSparseMatrix(args[0])) args[0] = sparseToDense(args[0]);
    const v = args[0];
    if (isRuntimeNumber(v)) return RTV.logical(v !== 0);
    if (isRuntimeLogical(v)) return RTV.logical(v);
    if (isRuntimeComplexNumber(v)) return RTV.logical(v.re !== 0 || v.im !== 0);
    if (isRuntimeTensor(v)) {
      if (args.length === 1) {
        if (v.data.length === 0) return RTV.logical(mode === "all");
        const d = firstReduceDim(v.shape);
        if (d === 0) return RTV.logical(scanLogical(v.data, v.imag, mode));
        return logicalAlongDim(v, d, mode);
      }
      const arg2 = args[1];
      if (
        (isRuntimeString(arg2) || isRuntimeChar(arg2)) &&
        rstr(arg2).toLowerCase() === "all"
      ) {
        if (v.data.length === 0) return RTV.logical(mode === "all");
        return RTV.logical(scanLogical(v.data, v.imag, mode));
      }
      if (isRuntimeNumber(arg2)) {
        return logicalAlongDim(v, Math.round(arg2), mode);
      }
      if (isRuntimeTensor(arg2)) {
        const dims = Array.from(arg2.data).map(d => Math.round(d));
        let result: RuntimeValue = v;
        for (const dim of dims) {
          if (isRuntimeTensor(result)) {
            result = logicalAlongDim(result, dim, mode);
          }
        }
        return result;
      }
    }
    throw new RuntimeError(`${name}: invalid arguments`);
  };
}

function anyAllMatch(argTypes: JitType[]): JitType[] | null {
  const parsed = parseStdReductionArgs(argTypes);
  if (!parsed) return null;
  return reductionOutputType(parsed.inputType, parsed.dimType, parsed.allFlag, {
    logicalOutput: true,
  });
}

defineBuiltin({
  name: "any",
  cases: [{ match: anyAllMatch, apply: anyAllApply("any", "any") }],
});

defineBuiltin({
  name: "all",
  cases: [{ match: anyAllMatch, apply: anyAllApply("all", "all") }],
});

// ── Cumulative type rule ───────────────────────────────────────────────

/** For cumulative ops (cumsum, cumprod, etc.), output shape = input shape. */
function cumulativeOutputType(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length === 0) return null;
  const inputType = argTypes[0];
  if (inputType.kind === "number" || inputType.kind === "boolean")
    return [{ kind: "number" }];
  if (inputType.kind === "complex_or_number")
    return [{ kind: "complex_or_number" }];
  if (inputType.kind === "tensor") {
    return [
      {
        kind: "tensor",
        isComplex: inputType.isComplex,
        shape: inputType.shape,
        ndim: inputType.ndim,
      },
    ];
  }
  if (inputType.kind === "sparse_matrix") {
    const shape =
      inputType.m !== undefined && inputType.n !== undefined
        ? [inputType.m, inputType.n]
        : undefined;
    return [{ kind: "tensor", isComplex: inputType.isComplex, shape }];
  }
  return null;
}

// ── cumsum ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "cumsum",
  cases: [
    {
      match: argTypes => cumulativeOutputType(argTypes),
      apply: args => {
        if (isRuntimeSparseMatrix(args[0])) args[0] = sparseToDense(args[0]);
        return cumOp("cumsum", args, (acc, val) => acc + val, 0);
      },
    },
  ],
});

// ── cumprod ────────────────────────────────────────────────────────────

defineBuiltin({
  name: "cumprod",
  cases: [
    {
      match: argTypes => cumulativeOutputType(argTypes),
      apply: args => {
        if (isRuntimeSparseMatrix(args[0])) args[0] = sparseToDense(args[0]);
        return cumOp(
          "cumprod",
          args,
          (acc, val) => acc * val,
          1,
          (aRe, aIm, bRe, bIm) => [aRe * bRe - aIm * bIm, aRe * bIm + aIm * bRe]
        );
      },
    },
  ],
});

// ── cummax ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "cummax",
  cases: [
    {
      match: argTypes => cumulativeOutputType(argTypes),
      apply: args => cumOp("cummax", args, Math.max),
    },
  ],
});

// ── cummin ─────────────────────────────────────────────────────────────

defineBuiltin({
  name: "cummin",
  cases: [
    {
      match: argTypes => cumulativeOutputType(argTypes),
      apply: args => cumOp("cummin", args, Math.min),
    },
  ],
});

// ── diff ───────────────────────────────────────────────────────────────

/** Compute the JIT output type for diff(A, [n], [dim]). */
function diffOutputType(argTypes: JitType[]): JitType[] | null {
  if (argTypes.length === 0) return null;
  const inputType = argTypes[0];

  // diff of a scalar → empty [0, 0] tensor
  if (inputType.kind === "number" || inputType.kind === "boolean") {
    return [{ kind: "tensor", isComplex: false, shape: [0, 0] }];
  }
  if (inputType.kind === "sparse_matrix") {
    const shape =
      inputType.m !== undefined && inputType.n !== undefined
        ? [inputType.m, inputType.n]
        : undefined;
    return diffOutputType([
      { kind: "tensor", isComplex: inputType.isComplex, shape },
      ...argTypes.slice(1),
    ]);
  }
  if (inputType.kind !== "tensor") return null;

  const nType = argTypes.length >= 2 ? argTypes[1] : undefined;
  const dimType = argTypes.length >= 3 ? argTypes[2] : undefined;

  if (!inputType.shape) {
    return [{ kind: "tensor", isComplex: inputType.isComplex }];
  }

  // Extract n (default 1)
  const n = !nType
    ? 1
    : nType.kind === "number" && nType.exact !== undefined
      ? nType.exact
      : undefined;
  if (n === undefined) {
    return [{ kind: "tensor", isComplex: inputType.isComplex }];
  }

  const shape = inputType.shape;

  // Determine operating dimension (0-based internally)
  let opDim: number;
  if (dimType?.kind === "number" && dimType.exact !== undefined) {
    opDim = dimType.exact - 1;
  } else if (shape.length <= 1 || (shape.length === 2 && shape[0] === 1)) {
    opDim = shape.length === 2 && shape[0] === 1 ? 1 : 0;
  } else {
    opDim = 0;
  }

  const resultShape = [...shape];
  const dimSize = opDim < resultShape.length ? resultShape[opDim] : 1;
  resultShape[opDim] = Math.max(0, dimSize - n);

  // Strip trailing singletons (keep at least 2D)
  while (resultShape.length > 2 && resultShape[resultShape.length - 1] === 1)
    resultShape.pop();

  return [
    { kind: "tensor", isComplex: inputType.isComplex, shape: resultShape },
  ];
}

defineBuiltin({
  name: "diff",
  cases: [
    {
      match: argTypes => diffOutputType(argTypes),
      apply: (args): RuntimeValue => {
        if (args.length < 1)
          throw new RuntimeError("diff requires at least 1 argument");
        if (isRuntimeSparseMatrix(args[0])) args[0] = sparseToDense(args[0]);
        const n = args.length >= 2 ? Math.round(toNumber(args[1])) : 1;
        const dimArg =
          args.length >= 3 ? Math.round(toNumber(args[2])) : undefined;
        let result = args[0] as RuntimeValue;
        for (let i = 0; i < n; i++) {
          result = diffOnce(result, dimArg) as RuntimeValue;
        }
        return result;
      },
    },
  ],
});

// ── xor ────────────────────────────────────────────────────────────────

import {
  getBroadcastShape,
  broadcastIterate,
} from "../../helpers/arithmetic.js";

defineBuiltin({
  name: "xor",
  cases: [
    // Two scalars
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        const a = argTypes[0],
          b = argTypes[1];
        const aOk = a.kind === "number" || a.kind === "boolean";
        const bOk = b.kind === "number" || b.kind === "boolean";
        if (!aOk || !bOk) return null;
        return [{ kind: "boolean" }];
      },
      apply: (args): RuntimeValue => {
        const aVal = isRuntimeLogical(args[0])
          ? args[0]
          : toNumber(args[0]) !== 0;
        const bVal = isRuntimeLogical(args[1])
          ? args[1]
          : toNumber(args[1]) !== 0;
        return RTV.logical(aVal !== bVal);
      },
    },
    // At least one tensor
    {
      match: argTypes => {
        if (argTypes.length !== 2) return null;
        const a = argTypes[0],
          b = argTypes[1];
        const aOk =
          a.kind === "number" ||
          a.kind === "boolean" ||
          (a.kind === "tensor" && a.isComplex !== true);
        const bOk =
          b.kind === "number" ||
          b.kind === "boolean" ||
          (b.kind === "tensor" && b.isComplex !== true);
        if (!aOk || !bOk) return null;
        if (a.kind !== "tensor" && b.kind !== "tensor") return null;
        const t =
          a.kind === "tensor" ? a : (b as Extract<JitType, { kind: "tensor" }>);
        return [
          { kind: "tensor", isComplex: false, isLogical: true, shape: t.shape },
        ];
      },
      apply: (args): RuntimeValue => {
        const a = args[0],
          b = args[1];
        const aIsT = isRuntimeTensor(a);
        const bIsT = isRuntimeTensor(b);
        const aScalar = !aIsT ? (toNumber(a) !== 0 ? 1 : 0) : 0;
        const bScalar = !bIsT ? (toNumber(b) !== 0 ? 1 : 0) : 0;
        if (aIsT && bIsT) {
          if (a.data.length !== b.data.length) {
            const outShape = getBroadcastShape(a.shape, b.shape);
            if (!outShape)
              throw new RuntimeError("xor: incompatible array sizes");
            const n = outShape.reduce((p, c) => p * c, 1);
            const out = uninitFloatX(n);
            broadcastIterate(a.shape, b.shape, outShape, (ai, bi, oi) => {
              out[oi] = (a.data[ai] !== 0) !== (b.data[bi] !== 0) ? 1 : 0;
            });
            const result = RTV.tensor(out, outShape);
            result._isLogical = true;
            return result;
          }
          const n = a.data.length;
          const out = uninitFloatX(n);
          for (let i = 0; i < n; i++) {
            out[i] = (a.data[i] !== 0) !== (b.data[i] !== 0) ? 1 : 0;
          }
          const result = RTV.tensor(out, a.shape);
          result._isLogical = true;
          return result;
        }
        const t = aIsT ? a : b;
        const s = aIsT ? bScalar : aScalar;
        if (!isRuntimeTensor(t))
          throw new RuntimeError("xor: unexpected state");
        const n = t.data.length;
        const out = uninitFloatX(n);
        for (let i = 0; i < n; i++) {
          out[i] = (t.data[i] !== 0) !== (s !== 0) ? 1 : 0;
        }
        const result = RTV.tensor(out, t.shape);
        result._isLogical = true;
        return result;
      },
    },
  ],
});

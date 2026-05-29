/**
 * `meshgrid` — 2-D coordinate grid builder.
 *
 * Forms supported:
 *   - `[X, Y] = meshgrid(x, y)` — `x` length M, `y` length N.
 *                                  Returns X (N×M, each row = x) and
 *                                  Y (N×M, each column = y).
 *   - `[X, Y] = meshgrid(x)`    — shorthand for `meshgrid(x, x)`.
 *   - `Z = meshgrid(x, y)` / `Z = meshgrid(x)` — returns just X.
 */

import { TypeError, UnsupportedConstruct } from "../../../lowering/errors.js";
import {
  EXACT_ARRAY_MAX_ELEMENTS,
  isDimOne,
  isMultiElement,
  tensorDouble,
  tensorDoubleFromDims,
  typeToString,
  type DimInfo,
  type NumericType,
  type Type,
} from "../../../lowering/types.js";
import type { Builtin } from "../../registry.js";
import { exactRealArray, requireRealDouble } from "../_shared.js";
import type { RuntimeTensor } from "../../../runtime/value.js";
import {
  mtoc2_meshgrid_x as jsMeshgridX,
  mtoc2_meshgrid as jsMeshgrid,
  mtoc2_meshgrid_1arg as jsMeshgrid1,
} from "../../runtime/snippets.gen.js";

function requireVecInput(t: Type, what: string): NumericType {
  requireRealDouble(t, what);
  const tN = t as NumericType;
  if (!isMultiElement(tN)) {
    throw new UnsupportedConstruct(
      `${what} must be a 1-D vector (1×N or N×1); scalar inputs are not ` +
        `supported`
    );
  }
  if (tN.dims.length !== 2) {
    throw new UnsupportedConstruct(
      `${what} must be a 1-D vector (got ${typeToString(tN)})`
    );
  }
  const isRow = isDimOne(tN.dims[0]);
  const isCol = isDimOne(tN.dims[1]);
  if (!isRow && !isCol) {
    throw new UnsupportedConstruct(
      `${what} must be a 1-D vector (1×N or N×1); got ${typeToString(tN)}`
    );
  }
  return tN;
}

function vecDim(t: NumericType): DimInfo {
  return isDimOne(t.dims[0]) ? t.dims[1] : t.dims[0];
}

function meshgridDataX(x: Float64Array, y: Float64Array): Float64Array {
  const M = x.length;
  const N = y.length;
  const out = new Float64Array(N * M);
  for (let j = 0; j < M; j++) {
    for (let i = 0; i < N; i++) {
      out[i + j * N] = x[j];
    }
  }
  return out;
}

function meshgridDataY(x: Float64Array, y: Float64Array): Float64Array {
  const M = x.length;
  const N = y.length;
  const out = new Float64Array(N * M);
  for (let j = 0; j < M; j++) {
    for (let i = 0; i < N; i++) {
      out[i + j * N] = y[i];
    }
  }
  return out;
}

function meshgridOutType(
  xN: NumericType,
  yN: NumericType,
  pick: "X" | "Y"
): NumericType {
  const xDim = vecDim(xN);
  const yDim = vecDim(yN);

  const xArr = exactRealArray(xN);
  const yArr = exactRealArray(yN);
  if (
    xDim.kind === "exact" &&
    yDim.kind === "exact" &&
    xArr !== undefined &&
    yArr !== undefined &&
    xArr.length === xDim.value &&
    yArr.length === yDim.value
  ) {
    const total = xDim.value * yDim.value;
    if (total <= EXACT_ARRAY_MAX_ELEMENTS) {
      const data =
        pick === "X" ? meshgridDataX(xArr, yArr) : meshgridDataY(xArr, yArr);
      return tensorDouble([yDim.value, xDim.value], data);
    }
  }

  return tensorDoubleFromDims([yDim, xDim]);
}

export const meshgrid: Builtin = {
  name: "meshgrid",
  transfer(argTypes, nargout) {
    if (argTypes.length < 1 || argTypes.length > 2) {
      throw new TypeError(
        `'meshgrid' expects 1..2 arg(s), got ${argTypes.length}`
      );
    }
    const xN = requireVecInput(argTypes[0], "'meshgrid' arg 1");
    const yN =
      argTypes.length >= 2
        ? requireVecInput(argTypes[1], "'meshgrid' arg 2")
        : xN;
    if (nargout === 1) {
      return [meshgridOutType(xN, yN, "X")];
    }
    if (nargout === 2) {
      return [meshgridOutType(xN, yN, "X"), meshgridOutType(xN, yN, "Y")];
    }
    throw new UnsupportedConstruct(
      `'meshgrid' supports 1..2 output(s); got nargout=${nargout}`
    );
  },
  emitC({ argsC, argTypes, nargout, outArgsC, useRuntime }) {
    useRuntime("mtoc2_meshgrid");
    if (nargout === 1) {
      const x = argsC[0];
      const y = argsC.length >= 2 ? argsC[1] : argsC[0];
      return `mtoc2_meshgrid_x(${x}, ${y})`;
    }
    // Multi-output: 1-arg shorthand uses a dedicated helper that takes
    // a single source vector; 2-arg uses the full helper with both.
    const outs = outArgsC ?? [];
    if (argTypes.length === 1) {
      return `mtoc2_meshgrid_1arg(${argsC[0]}, ${outs.join(", ")})`;
    }
    return `mtoc2_meshgrid(${argsC[0]}, ${argsC[1]}, ${outs.join(", ")})`;
  },
  emitJs({ argsJs, argTypes, nargout, useRuntime }) {
    useRuntime("mtoc2_meshgrid");
    if (nargout === 1) {
      const x = argsJs[0];
      const y = argsJs.length >= 2 ? argsJs[1] : argsJs[0];
      return `mtoc2_meshgrid_x(${x}, ${y})`;
    }
    // JS multi-output: helper returns `{X, Y}`; destructure inline so
    // the emitJs multi-assign wrapper can spread it.
    if (argTypes.length === 1) {
      return `(o => [o.X, o.Y])(mtoc2_meshgrid_1arg(${argsJs[0]}))`;
    }
    return `(o => [o.X, o.Y])(mtoc2_meshgrid(${argsJs[0]}, ${argsJs[1]}))`;
  },
  call({ args, argTypes, nargout }) {
    const xv = args[0] as RuntimeTensor;
    if (nargout === 1) {
      const y = argTypes.length >= 2 ? (args[1] as RuntimeTensor) : xv;
      return [jsMeshgridX(xv, y) as unknown as RuntimeTensor];
    }
    if (argTypes.length === 1) {
      const o = jsMeshgrid1(xv) as unknown as {
        X: RuntimeTensor;
        Y: RuntimeTensor;
      };
      return [o.X, o.Y];
    }
    const o = jsMeshgrid(xv, args[1] as RuntimeTensor) as unknown as {
      X: RuntimeTensor;
      Y: RuntimeTensor;
    };
    return [o.X, o.Y];
  },
};

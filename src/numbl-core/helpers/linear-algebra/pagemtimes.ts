/**
 * pagemtimes - Page-wise matrix multiplication.
 *
 * Z = pagemtimes(X, Y) computes Z(:,:,i) = X(:,:,i) * Y(:,:,i).
 * If one input is 2-D, it is broadcast across all pages.
 * Dimensions beyond the first two are broadcast using compatible sizes.
 *
 * Z = pagemtimes(X, transpX, Y, transpY) optionally transposes/ctransposes.
 */

import { RTV, RuntimeError } from "../../runtime/index.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeChar,
  isRuntimeNumber,
  isRuntimeString,
  isRuntimeTensor,
  RuntimeTensor,
  RuntimeValue,
} from "../../runtime/types.js";
import { rstr } from "../../runtime/runtime.js";
import { register, builtinSingle } from "../registry.js";
import { unknownMatrix } from "../check-helpers.js";

function matMulPage(
  aData: FloatXArrayType,
  aOffset: number,
  aRows: number,
  aCols: number,
  bData: FloatXArrayType,
  bOffset: number,
  bRows: number,
  bCols: number,
  outData: FloatXArrayType,
  outOffset: number
): void {
  for (let j = 0; j < bCols; j++) {
    for (let i = 0; i < aRows; i++) {
      let sum = 0;
      for (let k = 0; k < aCols; k++) {
        sum += aData[aOffset + i + k * aRows] * bData[bOffset + k + j * bRows];
      }
      outData[outOffset + i + j * aRows] = sum;
    }
  }
}

function transposePage(
  data: FloatXArrayType,
  offset: number,
  rows: number,
  cols: number,
  out: FloatXArrayType,
  outOffset: number
): void {
  for (let j = 0; j < cols; j++) {
    for (let i = 0; i < rows; i++) {
      out[outOffset + j + i * cols] = data[offset + i + j * rows];
    }
  }
}

export function registerPagemtimes(): void {
  register(
    "pagemtimes",
    builtinSingle(
      args => {
        let X: RuntimeValue, Y: RuntimeValue;
        let transpX = "none",
          transpY = "none";

        if (args.length === 2) {
          X = args[0];
          Y = args[1];
        } else if (args.length === 4) {
          X = args[0];
          Y = args[2];
          const tX = args[1];
          const tY = args[3];
          if (
            !(isRuntimeString(tX) || isRuntimeChar(tX)) ||
            !(isRuntimeString(tY) || isRuntimeChar(tY))
          ) {
            throw new RuntimeError(
              "pagemtimes: transpose options must be strings"
            );
          }
          transpX = rstr(tX);
          transpY = rstr(tY);
        } else {
          throw new RuntimeError("pagemtimes requires 2 or 4 arguments");
        }

        // Normalize scalars to 1x1 tensors
        const xT: RuntimeTensor = isRuntimeNumber(X)
          ? (RTV.tensor(new FloatXArray([X]), [1, 1]) as RuntimeTensor)
          : (X as RuntimeTensor);
        const yT: RuntimeTensor = isRuntimeNumber(Y)
          ? (RTV.tensor(new FloatXArray([Y]), [1, 1]) as RuntimeTensor)
          : (Y as RuntimeTensor);

        if (!isRuntimeTensor(xT) || !isRuntimeTensor(yT)) {
          throw new RuntimeError("pagemtimes: arguments must be numeric");
        }

        // Get page dimensions for X and Y
        const xShape = xT.shape.length < 2 ? [1, xT.shape[0] || 1] : xT.shape;
        const yShape = yT.shape.length < 2 ? [1, yT.shape[0] || 1] : yT.shape;

        let xRows = xShape[0],
          xCols = xShape[1];
        let yRows = yShape[0],
          yCols = yShape[1];

        // After transpose, the effective dimensions change
        if (transpX === "transpose" || transpX === "ctranspose") {
          [xRows, xCols] = [xCols, xRows];
        }
        if (transpY === "transpose" || transpY === "ctranspose") {
          [yRows, yCols] = [yCols, yRows];
        }

        if (xCols !== yRows) {
          throw new RuntimeError(
            `pagemtimes: inner matrix dimensions must agree: ${xCols} vs ${yRows}`
          );
        }

        // Get extra dimensions (beyond first 2)
        const xExtra = xShape.slice(2);
        const yExtra = yShape.slice(2);

        // Broadcast extra dimensions
        const maxExtraDims = Math.max(xExtra.length, yExtra.length);
        const outExtra: number[] = [];
        for (let d = 0; d < maxExtraDims; d++) {
          const xd = d < xExtra.length ? xExtra[d] : 1;
          const yd = d < yExtra.length ? yExtra[d] : 1;
          if (xd !== yd && xd !== 1 && yd !== 1) {
            throw new RuntimeError(
              `pagemtimes: dimensions beyond first two must be compatible`
            );
          }
          outExtra.push(Math.max(xd, yd));
        }

        const outShape = [xRows, yCols, ...outExtra];
        const pageSize = xRows * yCols;
        const xPageSize = xShape[0] * xShape[1];
        const yPageSize = yShape[0] * yShape[1];

        // Total number of pages
        const totalPages = outExtra.reduce((a, b) => a * b, 1);

        const result = new FloatXArray(pageSize * totalPages);

        // Temp buffers for transpose if needed
        const needTranspX = transpX === "transpose" || transpX === "ctranspose";
        const needTranspY = transpY === "transpose" || transpY === "ctranspose";
        const xTransBuf = needTranspX ? new FloatXArray(xPageSize) : null;
        const yTransBuf = needTranspY ? new FloatXArray(yPageSize) : null;

        // Compute strides for extra dimensions
        const xExtraStrides: number[] = [];
        const yExtraStrides: number[] = [];
        let xStride = 1,
          yStride = 1;
        for (let d = 0; d < maxExtraDims; d++) {
          xExtraStrides.push(xStride);
          yExtraStrides.push(yStride);
          xStride *= d < xExtra.length ? xExtra[d] : 1;
          yStride *= d < yExtra.length ? yExtra[d] : 1;
        }

        for (let p = 0; p < totalPages; p++) {
          // Compute page indices for X and Y using broadcasting
          let xPageIdx = 0,
            yPageIdx = 0;
          let rem = p;
          for (let d = maxExtraDims - 1; d >= 0; d--) {
            const outD = outExtra[d];
            const idx = rem % outD;
            rem = (rem - idx) / outD;
            const xd = d < xExtra.length ? xExtra[d] : 1;
            const yd = d < yExtra.length ? yExtra[d] : 1;
            xPageIdx += (xd === 1 ? 0 : idx) * xExtraStrides[d];
            yPageIdx += (yd === 1 ? 0 : idx) * yExtraStrides[d];
          }

          const xOff = xPageIdx * xPageSize;
          const yOff = yPageIdx * yPageSize;
          const outOff = p * pageSize;

          let aData = xT.data,
            aOff = xOff,
            aR = xShape[0],
            aC = xShape[1];
          let bData = yT.data,
            bOff = yOff,
            bR = yShape[0],
            bC = yShape[1];

          if (needTranspX && xTransBuf) {
            transposePage(xT.data, xOff, xShape[0], xShape[1], xTransBuf, 0);
            aData = xTransBuf;
            aOff = 0;
            aR = xCols; // original cols become rows after transpose
            aC = xRows; // original rows become cols after transpose
            // Wait, after transpose: aR = xCols (original), aC = xRows (original)
            // But xRows/xCols are already the *effective* (post-transpose) dims
            // So aR = xRows (effective), aC = xCols (effective)
            aR = xRows;
            aC = xCols;
          }

          if (needTranspY && yTransBuf) {
            transposePage(yT.data, yOff, yShape[0], yShape[1], yTransBuf, 0);
            bData = yTransBuf;
            bOff = 0;
            bR = yRows;
            bC = yCols;
          }

          matMulPage(aData, aOff, aR, aC, bData, bOff, bR, bC, result, outOff);
        }

        // Squeeze trailing singleton dims (keep at least 2)
        while (outShape.length > 2 && outShape[outShape.length - 1] === 1) {
          outShape.pop();
        }

        return RTV.tensor(result, outShape);
      },
      { outputType: unknownMatrix() }
    )
  );
}

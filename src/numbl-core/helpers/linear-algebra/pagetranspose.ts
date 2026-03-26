/**
 * pagetranspose - Page-wise transpose.
 *
 * Y = pagetranspose(X) transposes each page X(:,:,i) → Y(:,:,i).
 */

import { RTV, RuntimeError } from "../../runtime/index.js";
import { FloatXArray, isRuntimeTensor } from "../../runtime/types.js";
import { register, builtinSingle } from "../registry.js";
import { unknownMatrix } from "../check-helpers.js";

export function registerPagetranspose(): void {
  register(
    "pagetranspose",
    builtinSingle(
      args => {
        if (args.length !== 1)
          throw new RuntimeError("pagetranspose requires exactly 1 argument");
        const X = args[0];
        if (!isRuntimeTensor(X))
          throw new RuntimeError(
            "pagetranspose: input must be a numeric array"
          );

        const xShape = X.shape;
        const rows = xShape[0];
        const cols = xShape.length >= 2 ? xShape[1] : 1;
        const extraDims = xShape.slice(2);
        const totalPages = extraDims.reduce((a, b) => a * b, 1);
        const pageSize = rows * cols;

        const outData = new FloatXArray(X.data.length);
        const outShape = [cols, rows, ...extraDims];

        for (let p = 0; p < totalPages; p++) {
          const inOff = p * pageSize;
          const outOff = p * pageSize;
          // Transpose: out[j + i*cols] = in[i + j*rows] (column-major)
          for (let j = 0; j < cols; j++) {
            for (let i = 0; i < rows; i++) {
              outData[outOff + j + i * cols] = X.data[inOff + i + j * rows];
            }
          }
        }

        // Squeeze trailing singleton dims (keep at least 2)
        while (outShape.length > 2 && outShape[outShape.length - 1] === 1) {
          outShape.pop();
        }

        return RTV.tensor(outData, outShape);
      },
      { outputType: unknownMatrix() }
    )
  );
}

/**
 * Block diagonal matrix builtin function.
 *
 * B = blkdiag(A1,...,AN) returns the block diagonal matrix created by
 * aligning the input matrices A1,...,AN along the diagonal of B.
 */

import { RTV, RuntimeError } from "../../runtime/index.js";
import { tensorSize2D } from "../../runtime/utils.js";
import {
  FloatXArray,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { register, builtinSingle } from "../registry.js";
import { unknownMatrix } from "../check-helpers.js";

export function registerBlkdiag(): void {
  register(
    "blkdiag",
    builtinSingle(
      args => {
        if (args.length === 0)
          throw new RuntimeError("blkdiag requires at least 1 argument");

        // Normalize all inputs to tensors
        const blocks = args.map(a => {
          if (isRuntimeNumber(a)) {
            return RTV.tensor(new FloatXArray([a]), [1, 1]);
          }
          if (!isRuntimeTensor(a)) {
            throw new RuntimeError("blkdiag: arguments must be numeric");
          }
          return a;
        });

        // Compute total dimensions
        let totalRows = 0;
        let totalCols = 0;
        const dims: [number, number][] = [];
        for (const block of blocks) {
          const [m, n] = tensorSize2D(block);
          dims.push([m, n]);
          totalRows += m;
          totalCols += n;
        }

        // Fill result (column-major)
        const result = new FloatXArray(totalRows * totalCols); // initialized to 0

        let rowOffset = 0;
        let colOffset = 0;
        for (let k = 0; k < blocks.length; k++) {
          const [m, n] = dims[k];
          const data = blocks[k].data;
          for (let j = 0; j < n; j++) {
            for (let i = 0; i < m; i++) {
              result[rowOffset + i + (colOffset + j) * totalRows] =
                data[i + j * m];
            }
          }
          rowOffset += m;
          colOffset += n;
        }

        return RTV.tensor(result, [totalRows, totalCols]);
      },
      { outputType: unknownMatrix() }
    )
  );
}

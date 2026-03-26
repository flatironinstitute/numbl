/**
 * Kronecker tensor product builtin function.
 *
 * K = kron(A, B) returns the Kronecker tensor product of A and B.
 * If A is m-by-n and B is p-by-q, then K is (m*p)-by-(n*q).
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

export function registerKron(): void {
  register(
    "kron",
    builtinSingle(
      args => {
        if (args.length !== 2)
          throw new RuntimeError("kron requires 2 arguments");

        // Normalize scalars to 1x1 tensors
        const a = args[0];
        const b = args[1];
        const A = isRuntimeNumber(a)
          ? RTV.tensor(new FloatXArray([a]), [1, 1])
          : a;
        const B = isRuntimeNumber(b)
          ? RTV.tensor(new FloatXArray([b]), [1, 1])
          : b;

        if (!isRuntimeTensor(A) || !isRuntimeTensor(B))
          throw new RuntimeError("kron: arguments must be numeric");

        const [m, n] = tensorSize2D(A);
        const [p, q] = tensorSize2D(B);

        const rows = m * p;
        const cols = n * q;
        const result = new FloatXArray(rows * cols);

        // Column-major storage: result[row + col * rows]
        // K(i,j) = A(ia, ja) * B(ib, jb)
        // where ia = floor(i/p), ib = i mod p, ja = floor(j/q), jb = j mod q
        for (let ja = 0; ja < n; ja++) {
          for (let ia = 0; ia < m; ia++) {
            const aVal = A.data[ia + ja * m]; // column-major
            for (let jb = 0; jb < q; jb++) {
              for (let ib = 0; ib < p; ib++) {
                const bVal = B.data[ib + jb * p]; // column-major
                const row = ia * p + ib;
                const col = ja * q + jb;
                result[row + col * rows] = aVal * bVal;
              }
            }
          }
        }

        return RTV.tensor(result, [rows, cols]);
      },
      { outputType: unknownMatrix() }
    )
  );
}

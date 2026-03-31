// QR factorization. Reuses ts-lapack's proven implementations.
// The blocked outer loop with dlarfb is a TODO for future optimization.

import { dgeqrf_optimized } from "../../ts-lapack/src/SRC/dgeqrf_optimized.js";
import { dorgqr_optimized } from "../../ts-lapack/src/SRC/dorgqr_optimized.js";

export { dgeqrf_optimized as dgeqrf, dorgqr_optimized as dorgqr };

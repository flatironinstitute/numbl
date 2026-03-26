/**
 * QR decomposition builtin function
 */

import {
  colMajorIndex,
  RTV,
  RuntimeError,
  tensorSize2D,
} from "../../runtime/index.js";
import {
  FloatXArray,
  FloatXArrayType,
  isRuntimeNumber,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { getEffectiveBridge } from "../../native/bridge-resolve.js";
import { register } from "../registry.js";
import {
  out,
  isMatrixLike,
  parseEconArg,
  parseEconArgRuntime,
  toF64,
  unknownMatrix,
} from "../check-helpers.js";

// ── LAPACK helper ─────────────────────────────────────────────────────────────

/**
 * QR decomposition via LAPACK (dgeqrf + dorgqr).
 * Returns null if the bridge or its qr method is unavailable.
 */
function qrLapack(
  data: FloatXArrayType,
  m: number,
  n: number,
  econ: boolean,
  wantQ: boolean
): { Q: Float64Array; R: Float64Array } | null {
  const bridge = getEffectiveBridge("qr", "qr");
  if (!bridge?.qr) return null;
  return bridge.qr(toF64(data), m, n, econ, wantQ);
}

function qrLapackComplex(
  dataRe: FloatXArrayType,
  dataIm: FloatXArrayType,
  m: number,
  n: number,
  econ: boolean,
  wantQ: boolean
): {
  QRe?: Float64Array;
  QIm?: Float64Array;
  RRe: Float64Array;
  RIm: Float64Array;
} | null {
  const bridge = getEffectiveBridge("qr", "qrComplex");
  if (!bridge?.qrComplex) return null;
  return bridge.qrComplex(toF64(dataRe), toF64(dataIm), m, n, econ, wantQ);
}

export function registerQr(): void {
  /**
   * QR decomposition using Householder reflections.
   * Supports: [Q, R] = qr(A)       — full QR
   *           [Q, R] = qr(A, 0)    — economy/thin QR
   *           [Q, R] = qr(A, 'econ') — economy/thin QR
   */
  register("qr", [
    {
      check: (argTypes, nargout) => {
        if (
          nargout < 1 ||
          nargout > 2 ||
          argTypes.length < 1 ||
          argTypes.length > 2
        )
          return null;
        if (parseEconArg(argTypes[1]) === null) return null;
        if (!isMatrixLike(argTypes[0])) return null;
        const cpx =
          (argTypes[0].kind === "Tensor" && argTypes[0].isComplex) || undefined;
        if (nargout === 1) return out(unknownMatrix(cpx));
        return out(unknownMatrix(cpx), unknownMatrix(cpx));
      },
      apply: (args, nargout) => {
        if (args.length < 1)
          throw new RuntimeError("qr requires at least 1 argument");
        const A = args[0];
        if (isRuntimeNumber(A)) {
          // Scalar case
          // todo: check that this is right
          const val = A;
          const s = val >= 0 ? 1 : -1;
          if (nargout === 1)
            return RTV.tensor(new FloatXArray([s * val]), [1, 1]);
          const Q = RTV.tensor(new FloatXArray([s]), [1, 1]);
          const R = RTV.tensor(new FloatXArray([s * val]), [1, 1]);
          return [Q, R];
        }
        if (!isRuntimeTensor(A))
          throw new RuntimeError("qr: argument must be numeric");

        const econ = parseEconArgRuntime(args[1]);

        const [m, n] = tensorSize2D(A);
        const k = Math.min(m, n);

        // ── Complex QR via LAPACK ───────────────────────────────────────────
        if (A.imag) {
          const result = qrLapackComplex(
            A.data,
            A.imag,
            m,
            n,
            econ,
            nargout === 2
          );
          if (!result)
            throw new RuntimeError(
              "qr: complex QR requires the native LAPACK addon"
            );
          if (nargout === 1) {
            const rRows = econ ? k : m;
            return RTV.tensor(
              new FloatXArray(result.RRe),
              [rRows, n],
              new FloatXArray(result.RIm)
            );
          }
          const qCols = econ ? k : m;
          return [
            RTV.tensor(
              new FloatXArray(result.QRe!),
              [m, qCols],
              new FloatXArray(result.QIm!)
            ),
            RTV.tensor(
              new FloatXArray(result.RRe),
              [econ ? k : m, n],
              new FloatXArray(result.RIm)
            ),
          ];
        }

        // ── Real QR via LAPACK ──────────────────────────────────────────────
        const lapackResult = qrLapack(A.data, m, n, econ, nargout === 2);
        if (lapackResult) {
          if (nargout === 1) {
            const rRows = econ ? k : m;
            return RTV.tensor(new FloatXArray(lapackResult.R), [rRows, n]);
          }
          const qCols = econ ? k : m;
          return [
            RTV.tensor(new FloatXArray(lapackResult.Q), [m, qCols]),
            RTV.tensor(new FloatXArray(lapackResult.R), [econ ? k : m, n]),
          ];
        }

        /*
        Important: we disabled the ts lapack qr implementation because it is much slower than this pure ts implementation.
        TODO: We should figure out why this is the case because it would be great to understand how to optimize the ts lapack code!
        */

        console.warn(
          "LAPACK bridge not available, using pure JS QR decomposition (slow)"
        );

        // ── JS fallback (Householder reflections) ───────────────────────────
        // Copy A into a working matrix (column-major)
        const R_data = new FloatXArray(A.data);

        // Store Householder vectors for Q computation
        const vecs: FloatXArrayType[] = [];
        const taus: number[] = [];

        for (let j = 0; j < k; j++) {
          // Extract column j from row j..m-1
          const len = m - j;
          const x = new FloatXArray(len);
          for (let i = 0; i < len; i++) {
            x[i] = R_data[colMajorIndex(j + i, j, m)];
          }

          // Compute Householder vector
          let normx = 0;
          for (let i = 0; i < len; i++) normx += x[i] * x[i];
          normx = Math.sqrt(normx);

          if (normx === 0) {
            vecs.push(new FloatXArray(len));
            taus.push(0);
            continue;
          }

          // Choose sign to avoid cancellation
          const sign = x[0] >= 0 ? 1 : -1;
          const alpha = -sign * normx;

          // v = x - alpha * e1, then normalize
          const v = new FloatXArray(len);
          v[0] = x[0] - alpha;
          for (let i = 1; i < len; i++) v[i] = x[i];

          let vnorm = 0;
          for (let i = 0; i < len; i++) vnorm += v[i] * v[i];
          const tau = vnorm === 0 ? 0 : 2.0 / vnorm;

          vecs.push(v);
          taus.push(tau);

          // Apply Householder reflection to R: R(j:m, j:n) -= tau * v * (v' * R(j:m, j:n))
          for (let c = j; c < n; c++) {
            // Compute dot = v' * R(j:m, c)
            let dot = 0;
            for (let i = 0; i < len; i++) {
              dot += v[i] * R_data[colMajorIndex(j + i, c, m)];
            }
            // R(j:m, c) -= tau * dot * v
            const scale = tau * dot;
            for (let i = 0; i < len; i++) {
              R_data[colMajorIndex(j + i, c, m)] -= scale * v[i];
            }
          }
        }

        // If only R is needed, skip Q computation entirely.
        // R_data is fully finalized here; the Q-building code below only
        // creates and modifies Q_data and never touches R_data.
        if (nargout === 1) {
          if (econ) {
            const R_econ = new FloatXArray(k * n);
            for (let r = 0; r < k; r++) {
              for (let c = 0; c < n; c++) {
                R_econ[colMajorIndex(r, c, k)] = R_data[colMajorIndex(r, c, m)];
              }
            }
            return RTV.tensor(R_econ, [k, n]);
          } else {
            return RTV.tensor(new FloatXArray(R_data.slice(0, m * n)), [m, n]);
          }
        }

        // Build Q by applying Householder reflections to identity
        if (econ) {
          // Economy: Q is m x k
          const qCols = k;
          const Q_data = new FloatXArray(m * qCols);
          // Start with identity (m x k)
          for (let i = 0; i < Math.min(m, qCols); i++) {
            Q_data[colMajorIndex(i, i, m)] = 1;
          }

          // Apply reflections in reverse order
          for (let j = k - 1; j >= 0; j--) {
            const v = vecs[j];
            const tau = taus[j];
            if (tau === 0) continue;
            const len = m - j;

            for (let c = j; c < qCols; c++) {
              let dot = 0;
              for (let i = 0; i < len; i++) {
                dot += v[i] * Q_data[colMajorIndex(j + i, c, m)];
              }
              const scale = tau * dot;
              for (let i = 0; i < len; i++) {
                Q_data[colMajorIndex(j + i, c, m)] -= scale * v[i];
              }
            }
          }

          // Economy R is k x n
          const R_econ = new FloatXArray(k * n);
          for (let r = 0; r < k; r++) {
            for (let c = 0; c < n; c++) {
              R_econ[colMajorIndex(r, c, k)] = R_data[colMajorIndex(r, c, m)];
            }
          }

          const Q = RTV.tensor(Q_data, [m, qCols]);
          const R = RTV.tensor(R_econ, [k, n]);
          return [Q, R];
        } else {
          // Full: Q is m x m
          const Q_data = new FloatXArray(m * m);
          // Start with identity (m x m)
          for (let i = 0; i < m; i++) {
            Q_data[colMajorIndex(i, i, m)] = 1;
          }

          // Apply reflections in reverse order
          for (let j = k - 1; j >= 0; j--) {
            const v = vecs[j];
            const tau = taus[j];
            if (tau === 0) continue;
            const len = m - j;

            for (let c = j; c < m; c++) {
              let dot = 0;
              for (let i = 0; i < len; i++) {
                dot += v[i] * Q_data[colMajorIndex(j + i, c, m)];
              }
              const scale = tau * dot;
              for (let i = 0; i < len; i++) {
                Q_data[colMajorIndex(j + i, c, m)] -= scale * v[i];
              }
            }
          }

          const Q = RTV.tensor(Q_data, [m, m]);
          const R_full = RTV.tensor(new FloatXArray(R_data.slice(0, m * n)), [
            m,
            n,
          ]);
          return [Q, R_full];
        }
      },
    },
  ]);
}

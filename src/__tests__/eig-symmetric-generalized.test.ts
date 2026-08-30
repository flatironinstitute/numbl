import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";
import { isRuntimeTensor } from "../numbl-core/runtime/types.js";
import { getTsLapackBridge } from "../numbl-core/native/ts-lapack-bridge.js";

// These run on the ts-lapack bridge (no native addon under vitest), which is
// the path the browser build takes.

function tensor(code: string, varName: string) {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (!isRuntimeTensor(v)) throw new Error(`${varName} is not a tensor`);
  return v;
}

function scalar(code: string, varName: string): number {
  const result = executeCode(code);
  const v = result.variableValues[varName];
  if (typeof v !== "number") throw new Error(`${varName} is not a number`);
  return v;
}

describe("eigSym (ts-lapack dsyev)", () => {
  it("returns ascending eigenvalues and an orthonormal basis", () => {
    const n = 6;
    // S = Q diag(d) Q' with a repeated eigenvalue, Q a Householder reflector.
    const u = new Float64Array(n);
    for (let i = 0; i < n; i++) u[i] = i + 1 - (n + 1) / 2;
    let nrm = 0;
    for (let i = 0; i < n; i++) nrm += u[i] * u[i];
    nrm = Math.sqrt(nrm);
    for (let i = 0; i < n; i++) u[i] /= nrm;
    const d = [1, 3, 3, 3, 7, 9];
    const S = new Float64Array(n * n);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let k = 0; k < n; k++) {
          const qik = (i === k ? 1 : 0) - 2 * u[i] * u[k];
          const qjk = (j === k ? 1 : 0) - 2 * u[j] * u[k];
          s += qik * d[k] * qjk;
        }
        S[i + j * n] = s;
      }
    }
    const bridge = getTsLapackBridge();
    const r = bridge.eigSym!(S, n, true);
    for (let i = 0; i < n; i++) expect(r.w[i]).toBeCloseTo(d[i], 10);
    const V = r.V!;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += V[k + i * n] * V[k + j * n];
        expect(Math.abs(s - (i === j ? 1 : 0))).toBeLessThan(1e-12);
      }
    }
  });

  it("agrees with the eigenvalues-only path", () => {
    const S = Float64Array.from([2, 1, 0, 1, 3, 1, 0, 1, 4]);
    const bridge = getTsLapackBridge();
    const withV = bridge.eigSym!(S, 3, true);
    const noV = bridge.eigSym!(S, 3, false);
    expect(noV.V).toBeUndefined();
    for (let i = 0; i < 3; i++) expect(noV.w[i]).toBeCloseTo(withV.w[i], 12);
  });
});

describe("eig dispatch", () => {
  it("routes an exactly symmetric matrix to the symmetric solver", () => {
    // Repeated eigenvalue: dgeev would hand back a non-orthogonal basis here.
    const V = tensor(
      "A = [2 0 0; 0 3 0; 0 0 3]; A = A + 0; [V, D] = eig(A);",
      "V"
    );
    const dev = scalar(
      "A = [2 0 0; 0 3 0; 0 0 3]; [V, D] = eig(A); dev = max(max(abs(V'*V - eye(3))));",
      "dev"
    );
    expect(V.shape).toEqual([3, 3]);
    expect(dev).toBeLessThan(1e-14);
  });

  it("keeps a nearly-symmetric matrix on the general path", () => {
    // Exact test, as in MATLAB: one ulp of asymmetry is still asymmetric. The
    // giveaway is the ordering — only the symmetric path sorts ascending.
    const sym = tensor("A = [5 0.1 0; 0.1 1 0.1; 0 0.1 3]; e = eig(A);", "e");
    expect(sym.data[0]).toBeLessThan(sym.data[1]);
    expect(sym.data[1]).toBeLessThan(sym.data[2]);
    const near = tensor(
      "A = [5 0.1 0; 0.1 1 0.1; 0 0.1 3]; A(2,1) = A(2,1) + 1e-15; e = eig(A);",
      "e"
    );
    // dgeev's deflation order, matching MATLAB: 5.0025, 0.9925, 3.0050
    expect(near.data[0]).toBeGreaterThan(near.data[1]);
  });
});

describe("eig(A, B)", () => {
  it("solves the generalized problem instead of ignoring B", () => {
    const res = scalar(
      "A = [4 1 0; 1 3 1; 0 1 2]; B = [2 0 0; 0 5 0; 0 0 9];" +
        "[V, D] = eig(A, B); res = max(max(abs(A*V - B*V*D)));",
      "res"
    );
    expect(res).toBeLessThan(1e-12);
    const diff = scalar(
      "A = [4 1 0; 1 3 1; 0 1 2]; B = [2 0 0; 0 5 0; 0 0 9];" +
        "diff = max(abs(sort(eig(A, B)) - sort(eig(A))));",
      "diff"
    );
    expect(diff).toBeGreaterThan(1e-3);
  });

  it("normalizes the symmetric-definite case so that V'*B*V = I", () => {
    const dev = scalar(
      "A = [4 1 0; 1 3 1; 0 1 2]; B = [2 0 0; 0 5 0; 0 0 9];" +
        "[V, D] = eig(A, B); dev = max(max(abs(V'*B*V - eye(3))));",
      "dev"
    );
    expect(dev).toBeLessThan(1e-12);
  });

  it("errors rather than answering a different problem when QZ is unavailable", () => {
    // ts-lapack has no dgges/dggev, so a non symmetric-definite pencil has to
    // fail loudly on this path.
    expect(() => executeCode("e = eig([1 2; 3 4], [1 0; 0 -1]);")).toThrow(
      /native LAPACK addon/
    );
  });

  it("rejects a mismatched B", () => {
    expect(() =>
      executeCode("e = eig([1 2; 3 4], [1 2 3; 4 5 6; 7 8 9]);")
    ).toThrow(/same size/);
  });

  it("rejects an unknown option", () => {
    expect(() => executeCode("e = eig([1 2; 3 4], 'nosuchoption');")).toThrow(
      /unknown option/
    );
  });
});

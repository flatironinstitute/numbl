/**
 * Undirected graph object (`graph`) and a subset of its methods.
 *
 * MATLAB's `graph` is a value class representing an undirected graph. Here a
 * graph is a class_instance with className="graph" and three internal fields:
 *   _n         number of nodes
 *   _A         symmetric weighted adjacency as a RuntimeSparseMatrix
 *   _weighted  whether the graph carries edge weights
 *
 * Supported constructors:
 *   graph()                          empty graph
 *   graph(A)                         adjacency matrix (A must be symmetric)
 *   graph(A, 'omitselfloops')        drop the diagonal
 *   graph(A, 'upper' | 'lower')      read one triangle of A
 *   graph(s, t)                      edge list (unweighted)
 *   graph(s, t, w)                   edge list (weighted)
 *   graph(s, t, w, num)              edge list, num nodes
 *
 * Supported methods (called as functions, e.g. conncomp(G)):
 *   conncomp(G) / [bins, sizes] = conncomp(G)   connected components
 *   laplacian(G)                                 graph Laplacian (unweighted)
 *   addedge(G, s, t [, w])                       add edges, returns a new graph
 *   numnodes(G), numedges(G)                     counts
 *   degree(G)                                    per-node degree (column vector)
 *
 * Note: laplacian ignores edge weights (uses the binary adjacency and the
 * node-degree count), matching MATLAB.
 */

import type { RuntimeValue, RuntimeSparseMatrix } from "../../runtime/types.js";
import {
  RuntimeClassInstance,
  isRuntimeChar,
  isRuntimeClassInstance,
  isRuntimeComplexNumber,
  isRuntimeLogical,
  isRuntimeNumber,
  isRuntimeSparseMatrix,
  isRuntimeString,
  isRuntimeTensor,
} from "../../runtime/types.js";
import { RTV, RuntimeError } from "../../runtime/index.js";
import { toNumber } from "../../runtime/convert.js";
import { allocFloat64Array } from "../../runtime/alloc.js";
import { builtin } from "./builtin.js";
import { undefinedForTypeError } from "./value-class.js";
import { builtinError } from "./args.js";
import { imagAllZero } from "../../helpers/effectively-real.js";

// ── Internal representation ────────────────────────────────────────────────

interface Edge {
  u: number; // 1-based, u <= v
  v: number;
  w: number;
}

function isGraph(v: RuntimeValue): v is RuntimeClassInstance {
  return isRuntimeClassInstance(v) && v.className === "graph";
}

/** The graph a method is called on. The graph functions are methods of
 *  MATLAB's graph class, so a call without a graph first argument is an
 *  undefined function there. */
function requireGraph(args: RuntimeValue[], fn: string): RuntimeClassInstance {
  if (args.length === 0)
    throw builtinError(
      "MATLAB:UndefinedFunction",
      `Unrecognized function or variable '${fn}'.`
    );
  const v = args[0];
  if (!isGraph(v)) throw undefinedForTypeError(fn, v);
  return v;
}

/** A real numeric argument (scalar or array) as a flat list. Complex
 *  input (with a nonzero imaginary part) raises `err`, as does any
 *  non-numeric value. */
function realNumbers(v: RuntimeValue, err: () => RuntimeError): number[] {
  if (isRuntimeNumber(v)) return [v];
  if (isRuntimeLogical(v)) return [v ? 1 : 0];
  if (isRuntimeComplexNumber(v)) {
    if (v.im !== 0) throw err();
    return [v.re];
  }
  if (isRuntimeTensor(v)) {
    if (v.imag && !imagAllZero(v.imag)) throw err();
    return Array.from(v.data);
  }
  throw err();
}

const invalidSource = () =>
  builtinError(
    "MATLAB:graphfun:graphbuiltin:InvalidSRC",
    "Source must be a numeric array of positive integer node indices."
  );
const invalidTarget = () =>
  builtinError(
    "MATLAB:graphfun:graphbuiltin:InvalidTARG",
    "Target must be a numeric array of positive integer node indices."
  );
const invalidWeights = () =>
  builtinError(
    "MATLAB:graphfun:graph:InvalidWeights",
    "Graph edge weights must be double or single, real, and not sparse."
  );
const invalidAdjacency = () =>
  builtinError(
    "MATLAB:graphfun:graphbuiltin:InvalidAdjacency",
    "Invalid data type. Adjacency matrix must be a real double, single, or logical square matrix."
  );
const invalidAddedgeNode = () =>
  builtinError(
    "MATLAB:graphfun:addedge:InvalidNodeID",
    "Numeric node IDs must be positive integers."
  );

function graphN(g: RuntimeClassInstance): number {
  return toNumber(g.fields.get("_n") ?? 0);
}

function graphWeighted(g: RuntimeClassInstance): boolean {
  const w = g.fields.get("_weighted");
  return w === true;
}

function graphAdj(g: RuntimeClassInstance): RuntimeSparseMatrix {
  const a = g.fields.get("_A");
  if (a === undefined || !isRuntimeSparseMatrix(a)) {
    throw new RuntimeError("graph: corrupt internal adjacency");
  }
  return a;
}

/** Build a symmetric sparse adjacency matrix from an edge list. */
function buildSymAdj(n: number, edges: Edge[]): RuntimeSparseMatrix {
  // Triplets in (col, row) order, summing duplicates (parallel edges).
  const triplets: { col: number; row: number; val: number }[] = [];
  for (const e of edges) {
    if (e.w === 0) continue;
    const u = e.u - 1;
    const v = e.v - 1;
    if (u === v) {
      triplets.push({ col: u, row: u, val: e.w });
    } else {
      triplets.push({ col: v, row: u, val: e.w });
      triplets.push({ col: u, row: v, val: e.w });
    }
  }
  triplets.sort((a, b) => a.col - b.col || a.row - b.row);
  const ir: number[] = [];
  const pr: number[] = [];
  const cols: number[] = [];
  let prevCol = -1;
  let prevRow = -1;
  for (const t of triplets) {
    if (t.col === prevCol && t.row === prevRow) {
      pr[pr.length - 1] += t.val;
    } else {
      ir.push(t.row);
      pr.push(t.val);
      cols.push(t.col);
      prevCol = t.col;
      prevRow = t.row;
    }
  }
  const jc = new Int32Array(n + 1);
  let ci = 0;
  for (let c = 0; c < n; c++) {
    jc[c] = ci;
    while (ci < cols.length && cols[ci] === c) ci++;
  }
  jc[n] = ci;
  return RTV.sparseMatrix(n, n, new Int32Array(ir), jc, allocFloat64Array(pr));
}

function makeGraph(
  n: number,
  edges: Edge[],
  weighted: boolean
): RuntimeClassInstance {
  const fields = new Map<string, RuntimeValue>();
  fields.set("_n", n);
  fields.set("_A", buildSymAdj(n, edges));
  fields.set("_weighted", weighted);
  return new RuntimeClassInstance("graph", fields, false);
}

/** Iterate the off-diagonal neighbors r (0-based) of column c in `A`. */
function eachNeighbor(
  A: RuntimeSparseMatrix,
  c: number,
  cb: (r: number, val: number) => void
): void {
  for (let k = A.jc[c]; k < A.jc[c + 1]; k++) {
    const r = A.ir[k];
    if (r !== c) cb(r, A.pr[k]);
  }
}

/** Recover the undirected edge list (u <= v) from a symmetric adjacency. */
function edgesFromAdj(A: RuntimeSparseMatrix): Edge[] {
  const edges: Edge[] = [];
  for (let c = 0; c < A.n; c++) {
    for (let k = A.jc[c]; k < A.jc[c + 1]; k++) {
      const r = A.ir[k];
      if (r < c) edges.push({ u: r + 1, v: c + 1, w: A.pr[k] });
      else if (r === c) edges.push({ u: r + 1, v: r + 1, w: A.pr[k] });
    }
  }
  return edges;
}

// ── Adjacency-matrix → edge list ───────────────────────────────────────────

type Triangle = "sym" | "upper" | "lower";

function squareDim(A: RuntimeValue, fn: string): number {
  if (isRuntimeNumber(A) || isRuntimeLogical(A)) return 1;
  if (
    isRuntimeComplexNumber(A) ||
    (isRuntimeTensor(A) && A.imag && !imagAllZero(A.imag)) ||
    (isRuntimeSparseMatrix(A) && A.pi && !imagAllZero(A.pi))
  )
    throw invalidAdjacency();
  if (isRuntimeSparseMatrix(A)) {
    if (A.m !== A.n) throw new RuntimeError(`${fn}: adjacency must be square`);
    return A.n;
  }
  if (isRuntimeTensor(A)) {
    const rows = A.shape[0] ?? 1;
    const cols = A.shape[1] ?? 1;
    if (rows !== cols)
      throw new RuntimeError(`${fn}: adjacency must be square`);
    return rows;
  }
  throw new RuntimeError(`${fn}: adjacency must be a numeric matrix`);
}

/** True when an entry at (r, c) should be taken as the canonical edge. */
function takeEntry(r: number, c: number, triangle: Triangle): boolean {
  if (r === c) return true; // self-loop, handled by caller
  if (triangle === "lower") return r > c;
  return r < c; // 'sym' (assume symmetric) and 'upper'
}

function adjToEdges(
  A: RuntimeValue,
  omitSelfLoops: boolean,
  triangle: Triangle,
  fn: string
): { n: number; edges: Edge[]; weighted: boolean } {
  const n = squareDim(A, fn);
  const weighted = !(isRuntimeTensor(A) && A._isLogical);
  const edges: Edge[] = [];

  const push = (r: number, c: number, val: number) => {
    if (val === 0) return;
    if (r === c) {
      if (!omitSelfLoops) edges.push({ u: r + 1, v: r + 1, w: val });
    } else if (takeEntry(r, c, triangle)) {
      edges.push({ u: Math.min(r, c) + 1, v: Math.max(r, c) + 1, w: val });
    }
  };

  if (isRuntimeSparseMatrix(A)) {
    for (let c = 0; c < A.n; c++) {
      for (let k = A.jc[c]; k < A.jc[c + 1]; k++) push(A.ir[k], c, A.pr[k]);
    }
  } else if (isRuntimeTensor(A)) {
    const m = A.shape[0] ?? 1;
    for (let c = 0; c < n; c++) {
      for (let r = 0; r < n; r++) {
        const val = A.data[c * m + r];
        if (val !== 0) push(r, c, val);
      }
    }
  } else if (isRuntimeNumber(A) || isRuntimeLogical(A)) {
    const val = isRuntimeNumber(A) ? A : A ? 1 : 0;
    if (val !== 0) push(0, 0, val);
  }
  return { n, edges, weighted };
}

// ── Argument helpers ───────────────────────────────────────────────────────

function isNumericArg(v: RuntimeValue): boolean {
  return (
    isRuntimeNumber(v) ||
    isRuntimeLogical(v) ||
    isRuntimeComplexNumber(v) ||
    isRuntimeTensor(v) ||
    isRuntimeSparseMatrix(v)
  );
}

function argString(v: RuntimeValue): string | null {
  if (isRuntimeString(v)) return v;
  if (isRuntimeChar(v)) return v.value;
  return null;
}

// ── graph constructor ──────────────────────────────────────────────────────

builtin({
  name: "graph",
  help: {
    signatures: [
      "G = graph(A)",
      "G = graph(A, 'omitselfloops')",
      "G = graph(A, 'upper' | 'lower')",
      "G = graph(s, t)",
      "G = graph(s, t, w)",
      "G = graph(s, t, w, num)",
    ],
    description:
      "Create an undirected graph from an adjacency matrix A (which must be " +
      "symmetric) or from edge endpoint lists s and t with optional weights w.",
  },
  // MATLAB reports too many inputs as maxrhs and too many outputs as
  // TooManyOutputs (checked in apply).
  nargin: [0, 5],
  nargout: [0, Infinity],
  apply: (args, nargout) => {
    if (nargout > 1)
      throw builtinError("MATLAB:TooManyOutputs", "Too many output arguments.");
    if (args.length === 0) return makeGraph(0, [], false);

    // Edge-list form: graph(s, t, ...) where the 2nd arg is numeric.
    if (args.length >= 2 && isNumericArg(args[1])) {
      const s = realNumbers(args[0], invalidSource);
      const t = realNumbers(args[1], invalidTarget);
      if (s.length !== t.length) {
        throw new RuntimeError("graph: s and t must have the same length");
      }
      let weighted = false;
      let w: number[] | null = null;
      let num: number | null = null;
      let omitSelfLoops = false;
      if (args.length >= 3 && isNumericArg(args[2])) {
        w = realNumbers(args[2], invalidWeights);
        weighted = true;
        if (args.length >= 4 && isNumericArg(args[3])) {
          num = Math.floor(toNumber(args[3]));
        }
      }
      for (let i = 2; i < args.length; i++) {
        const str = argString(args[i]);
        if (str && str.toLowerCase() === "omitselfloops") omitSelfLoops = true;
      }
      let n = num ?? 0;
      for (let i = 0; i < s.length; i++) n = Math.max(n, s[i], t[i]);
      const edges: Edge[] = [];
      for (let i = 0; i < s.length; i++) {
        const u = Math.min(s[i], t[i]);
        const v = Math.max(s[i], t[i]);
        if (omitSelfLoops && u === v) continue;
        edges.push({ u, v, w: w ? w[w.length === 1 ? 0 : i] : 1 });
      }
      return makeGraph(n, edges, weighted);
    }

    // Adjacency-matrix form: graph(A, options...).
    let omitSelfLoops = false;
    let triangle: Triangle = "sym";
    for (let i = 1; i < args.length; i++) {
      const str = argString(args[i]);
      if (!str) continue;
      const lc = str.toLowerCase();
      if (lc === "omitselfloops") omitSelfLoops = true;
      else if (lc === "upper") triangle = "upper";
      else if (lc === "lower") triangle = "lower";
    }
    const { n, edges, weighted } = adjToEdges(
      args[0],
      omitSelfLoops,
      triangle,
      "graph"
    );
    return makeGraph(n, edges, weighted);
  },
});

// ── conncomp ───────────────────────────────────────────────────────────────

/** Check conncomp's name-value options. Only the default vector output is
 *  supported; 'Type' applies to directed graphs and is ignored here (MATLAB
 *  warns and ignores it for undirected graphs). */
function checkConncompOptions(opts: RuntimeValue[]): void {
  if (opts.length % 2 !== 0)
    throw new RuntimeError("conncomp: name-value arguments must come in pairs");
  for (let i = 0; i < opts.length; i += 2) {
    const name = argString(opts[i])?.toLowerCase();
    const value = argString(opts[i + 1])?.toLowerCase();
    if (name === "outputform") {
      if (value === "vector") continue;
      if (value === "cell")
        throw new RuntimeError(
          "conncomp: 'OutputForm', 'cell' is not supported"
        );
      throw new RuntimeError(
        "conncomp: 'OutputForm' must be 'vector' or 'cell'"
      );
    }
    if (name === "type") continue;
    throw new RuntimeError(
      "conncomp: parameter name must be 'OutputForm' or 'Type'"
    );
  }
}

builtin({
  name: "conncomp",
  help: {
    signatures: ["bins = conncomp(G)", "[bins, binsizes] = conncomp(G)"],
    description:
      "Connected components of an undirected graph G. Returns a row vector " +
      "labeling each node with its component index, and optionally the size " +
      "of each component.",
  },
  nargout: 2,
  mfile: true,
  apply: (args, nargout) => {
    const g = requireGraph(args, "conncomp");
    checkConncompOptions(args.slice(1));
    const n = graphN(g);
    const A = graphAdj(g);
    const bins = new Float64Array(n);
    const sizes: number[] = [];
    let comp = 0;
    const stack: number[] = [];
    for (let start = 0; start < n; start++) {
      if (bins[start] !== 0) continue;
      comp++;
      let count = 0;
      bins[start] = comp;
      stack.length = 0;
      stack.push(start);
      while (stack.length > 0) {
        const node = stack.pop() as number;
        count++;
        eachNeighbor(A, node, r => {
          if (bins[r] === 0) {
            bins[r] = comp;
            stack.push(r);
          }
        });
      }
      sizes.push(count);
    }
    const binsT = RTV.tensor(bins, [1, n]);
    if (nargout >= 2) {
      return [binsT, RTV.tensor(allocFloat64Array(sizes), [1, sizes.length])];
    }
    return binsT;
  },
});

// ── laplacian ──────────────────────────────────────────────────────────────

builtin({
  name: "laplacian",
  help: {
    signatures: ["L = laplacian(G)"],
    description:
      "Graph Laplacian matrix L = D - A of an undirected graph G, where A is " +
      "the (binary, unweighted) adjacency matrix and D the diagonal node-degree " +
      "matrix. Returns a sparse matrix.",
  },
  // G is required; requireGraph raises MATLAB's error for a missing graph.
  nargin: [0, 1],
  mfile: true,
  apply: args => {
    const g = requireGraph(args, "laplacian");
    const n = graphN(g);
    const A = graphAdj(g);
    const edges: Edge[] = [];
    for (let c = 0; c < n; c++) {
      let degree = 0;
      eachNeighbor(A, c, r => {
        degree++;
        if (r < c) edges.push({ u: r + 1, v: c + 1, w: -1 });
      });
      edges.push({ u: c + 1, v: c + 1, w: degree });
    }
    // Reuse the symmetric builder: off-diagonal -1 mirrored, diagonal once.
    return buildSymAdj(n, edges);
  },
});

// ── addedge ────────────────────────────────────────────────────────────────

builtin({
  name: "addedge",
  help: {
    signatures: ["H = addedge(G, s, t)", "H = addedge(G, s, t, w)"],
    description:
      "Add one or more edges between nodes s and t to graph G, returning a new " +
      "graph. Weights w are required when G is a weighted graph, and not " +
      "allowed for an unweighted one.",
  },
  nargin: [0, 4],
  mfile: true,
  apply: args => {
    const g = requireGraph(args, "addedge");
    if (args.length < 3)
      throw builtinError("MATLAB:minrhs", "Not enough input arguments.");
    const weighted = graphWeighted(g);
    const nodeIds = (v: RuntimeValue) => {
      const ids = realNumbers(v, invalidAddedgeNode);
      if (ids.some(x => !(Number.isInteger(x) && x >= 1)))
        throw invalidAddedgeNode();
      return ids;
    };
    const s = nodeIds(args[1]);
    const t = nodeIds(args[2]);
    if (s.length !== t.length) {
      throw new RuntimeError("addedge: s and t must have the same length");
    }
    let w: number[] | null = null;
    if (args.length >= 4) {
      if (!weighted)
        throw builtinError(
          "MATLAB:graphfun:addedge:NoWeights",
          "Unable to add weighted edges to an unweighted graph."
        );
      w = realNumbers(args[3], invalidWeights);
    } else if (weighted) {
      throw new RuntimeError(
        "addedge: Must specify weights when adding an edge to a weighted graph."
      );
    }
    const edges = edgesFromAdj(graphAdj(g));
    let n = graphN(g);
    for (let i = 0; i < s.length; i++) {
      const u = Math.min(s[i], t[i]);
      const v = Math.max(s[i], t[i]);
      n = Math.max(n, u, v);
      edges.push({ u, v, w: w ? w[w.length === 1 ? 0 : i] : 1 });
    }
    return makeGraph(n, edges, weighted);
  },
});

// ── numnodes / numedges / degree ───────────────────────────────────────────

builtin({
  name: "numnodes",
  help: {
    signatures: ["n = numnodes(G)"],
    description: "Number of nodes in graph G.",
  },
  nargin: [0, 1],
  mfile: true,
  apply: args => RTV.num(graphN(requireGraph(args, "numnodes"))),
});

builtin({
  name: "numedges",
  help: {
    signatures: ["m = numedges(G)"],
    description: "Number of edges in graph G.",
  },
  nargin: [0, 1],
  mfile: true,
  apply: args => {
    const g = requireGraph(args, "numedges");
    return RTV.num(edgesFromAdj(graphAdj(g)).length);
  },
});

builtin({
  name: "degree",
  help: {
    signatures: ["d = degree(G)", "d = degree(G, nodeIDs)"],
    description:
      "Degree of each node in graph G, returned as a column vector of edge " +
      "counts (self-loops excluded). With numeric nodeIDs, the degrees of " +
      "those nodes, in the shape of nodeIDs.",
  },
  nargin: [0, 2],
  mfile: true,
  apply: args => {
    const g = requireGraph(args, "degree");
    const n = graphN(g);
    const A = graphAdj(g);
    const d = new Float64Array(n);
    for (let c = 0; c < n; c++) eachNeighbor(A, c, () => (d[c] += 1));
    if (args.length < 2) return RTV.tensor(d, [n, 1]);
    const idArg = args[1];
    const badId = () =>
      builtinError(
        "MATLAB:graphfun:graph:InvalidNodeID",
        `Numeric node IDs must be positive integers not greater than the number of nodes in the graph (${n}).`
      );
    const ids = realNumbers(idArg, badId);
    const out = allocFloat64Array(ids.length);
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (!(Number.isInteger(id) && id >= 1 && id <= n)) throw badId();
      out[i] = d[id - 1];
    }
    if (!isRuntimeTensor(idArg)) return out[0];
    return RTV.tensor(out, [...idArg.shape]);
  },
});

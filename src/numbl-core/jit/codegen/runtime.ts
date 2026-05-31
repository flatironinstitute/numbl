/**
 * mtoc2 runtime helpers — small C snippets inlined into the generated
 * source on demand.
 *
 * Each helper lives in its own .h file under `runtime/` so it can be
 * edited with normal C tooling (clangd, syntax highlighting). The .h
 * sources are inlined into `runtime/snippets.gen.ts` by
 * `scripts/build_runtime_snippets.ts`; this module reads them from
 * there (rather than the filesystem at load time) so the translator
 * bundles in the browser.
 *
 * Each helper is referenced by a stable name (e.g. "mtoc2_disp_double").
 * Snippets can declare other snippets they depend on; the activator
 * pulls dependencies in first.
 */

import {
  C_SNIPPETS,
  JS_SNIPPETS,
  JS_IMPORTS,
} from "../builtins/runtime/snippets.gen.js";
import { getBuiltin } from "../builtins/index.js";
import type { Builtin } from "../builtins/registry.js";

/** Minimal Workspace shape consulted at emit time. Importing the
 *  concrete Workspace class would create a cycle (workspace ↔ codegen);
 *  the structural type below captures exactly what codegen needs. */
export interface WorkspaceLike {
  getUserBuiltin(name: string): Builtin | undefined;
}

export interface RuntimeSnippet {
  /** Standard-library headers parsed out of the source file. */
  headers: ReadonlyArray<string>;
  /** Body of the snippet (definitions only — `#include`s removed). */
  code: string;
  /** Paired JS body, when a `<basename>.js` sibling exists in the
   *  runtime directory. Populated by `loadSnippet` from
   *  `JS_SNIPPETS[<basename>.js]`. `emitJs` activates and inlines
   *  this; if absent, the snippet is C-only (the builtin that owns
   *  it has no JS path yet). */
  jsCode?: string;
  /** Other helpers (by name) this snippet depends on. The activator
   *  pulls these in first so their definitions come before this
   *  snippet's. Cycles are not supported — keep the graph acyclic. */
  deps: ReadonlyArray<string>;
  /** `.h` filename this snippet was loaded from (e.g.
   *  `tensor_alloc_nd.h`). Used to build the JS-import dep map at
   *  activation time. Optional because user-supplied inline snippets
   *  have no on-disk file. */
  srcFilename?: string;
}

/**
 * Parse a raw snippet source string into its `headers` and `code` parts.
 *
 * The strict `#include` pattern accepted is:
 *   optional-whitespace `#` optional-whitespace `include`
 *   whitespace `<header>` or `"header"` optional-trailing-whitespace
 *
 * Any line whose trimmed form starts with `#include` but does NOT match
 * that pattern is rejected.
 */
export function parseSnippetSource(raw: string): {
  headers: string[];
  code: string;
} {
  const headers: string[] = [];
  const bodyLines: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    const m = line.match(/^\s*#\s*include\s+(<[^>]+>|"[^"]+")\s*$/);
    if (m) {
      headers.push(m[1]);
    } else if (/^\s*#\s*include\b/.test(line)) {
      throw new Error(
        `runtime snippet: unexpected #include form: ${JSON.stringify(line)}; ` +
          `expected '#include <header>' or '#include "header"' with no trailing content`
      );
    } else {
      bodyLines.push(line);
    }
  }
  while (bodyLines.length && bodyLines[0].trim() === "") bodyLines.shift();
  while (bodyLines.length && bodyLines[bodyLines.length - 1].trim() === "")
    bodyLines.pop();
  return { headers, code: bodyLines.join("\n") + "\n" };
}

function loadSnippet(
  filename: string,
  deps: ReadonlyArray<string> = []
): RuntimeSnippet {
  const raw = C_SNIPPETS[filename];
  if (raw === undefined) {
    throw new Error(
      `runtime snippet '${filename}' not found in snippets.gen.ts; ` +
        `re-run 'npm run build:snippets' after adding the .h file`
    );
  }
  const { headers, code } = parseSnippetSource(raw);
  // Optional paired `.js` sibling (same basename) — auto-bound when
  // present so `emitJs` can render the JS body without each call
  // site looking it up separately.
  const jsName = filename.replace(/\.h$/, ".js");
  const jsCode = JS_SNIPPETS[jsName];
  const snippet: RuntimeSnippet = {
    headers,
    code,
    deps,
    srcFilename: filename,
  };
  if (jsCode !== undefined) snippet.jsCode = jsCode;
  return snippet;
}

// ── Registry ──────────────────────────────────────────────────────────

/** Snippet name → snippet definition. The name is the C identifier the
 *  emitted code calls (e.g. `mtoc2_disp_double`). */
const REGISTRY: ReadonlyMap<string, RuntimeSnippet> = new Map<
  string,
  RuntimeSnippet
>([
  // JIT host output hook — defines `mtoc2_stdout` (routes through a
  // host-bound writer when present, else libc stdout) and the exported
  // `mtoc2_set_host_write`. A dep of every snippet that emits program
  // output so numbl can capture --opt 2 output like the other paths.
  ["mtoc2_host_output", loadSnippet("host_output.h")],
  ["mtoc2_format_double", loadSnippet("format_double.h")],
  [
    "mtoc2_disp_double",
    loadSnippet("disp_double.h", ["mtoc2_format_double", "mtoc2_host_output"]),
  ],

  // ── Scalar complex helpers ────────────────────────────────────────
  // Every `double _Complex` operation in mtoc2-emitted user code routes
  // through one of these `static inline` wrappers (`mtoc2_cmake`,
  // `mtoc2_cadd`, `mtoc2_cmul`, `mtoc2_creal`, `mtoc2_cnonzero`, …)
  // instead of using C99's operator overloading or the `I` macro
  // inline. Native cost is zero (the compiler inlines back to the
  // same instructions C99 would have generated); the win is that
  // the c2js JS backend can ship a matching set of helpers operating
  // on a `{re, im}` representation without learning to type-track
  // expressions. Any builtin whose codegen produces scalar-complex
  // C lists `mtoc2_cscalar` in its `runtimeDeps`.
  ["mtoc2_cscalar", loadSnippet("cscalar.h")],
  // Smith's algorithm-based complex divide that matches numbl's
  // signed-Inf-on-zero-divisor behavior. Activated by `rdivide`'s
  // complex scalar path.
  ["mtoc2_cdiv", loadSnippet("cdiv.h")],
  // Numbl-compatible complex formatter. Mirrors `formatComplex` byte-
  // for-byte so cross-runner stdout aligns.
  [
    "mtoc2_format_complex",
    loadSnippet("format_complex.h", ["mtoc2_format_double"]),
  ],
  [
    "mtoc2_disp_complex",
    loadSnippet("disp_complex.h", [
      "mtoc2_format_complex",
      "mtoc2_host_output",
    ]),
  ],

  // ── Text (string + char tensor) ───────────────────────────────────
  // Two distinct owned kinds: `mtoc2_string_t` (scalar handle,
  // double-quoted) and `mtoc2_char_tensor_t` (1×N row-vector of bytes,
  // single-quoted). Each ships the four owned-value helpers
  // (empty/assign/copy/free) plus a literal builder that points at
  // a `.rodata` C string (no allocation). The non-owning
  // `mtoc2_text_view_t` adapter lets read-only helpers (disp, future
  // error/strcmp/fprintf) walk either source uniformly.
  ["mtoc2_string_t", loadSnippet("string.h")],
  ["mtoc2_string_empty", loadSnippet("string_empty.h", ["mtoc2_string_t"])],
  ["mtoc2_string_free", loadSnippet("string_free.h", ["mtoc2_string_t"])],
  [
    "mtoc2_string_assign",
    loadSnippet("string_assign.h", ["mtoc2_string_t", "mtoc2_string_free"]),
  ],
  ["mtoc2_string_copy", loadSnippet("string_copy.h", ["mtoc2_string_t"])],
  [
    "mtoc2_string_from_literal",
    loadSnippet("string_from_literal.h", ["mtoc2_string_t"]),
  ],
  ["mtoc2_char_tensor_t", loadSnippet("char_tensor.h")],
  [
    "mtoc2_char_tensor_empty",
    loadSnippet("char_tensor_empty.h", ["mtoc2_char_tensor_t"]),
  ],
  [
    "mtoc2_char_tensor_free",
    loadSnippet("char_tensor_free.h", ["mtoc2_char_tensor_t"]),
  ],
  [
    "mtoc2_char_tensor_assign",
    loadSnippet("char_tensor_assign.h", [
      "mtoc2_char_tensor_t",
      "mtoc2_char_tensor_free",
    ]),
  ],
  [
    "mtoc2_char_tensor_copy",
    loadSnippet("char_tensor_copy.h", ["mtoc2_char_tensor_t"]),
  ],
  [
    "mtoc2_char_tensor_from_literal",
    loadSnippet("char_tensor_from_literal.h", ["mtoc2_char_tensor_t"]),
  ],
  [
    "mtoc2_text_view_t",
    loadSnippet("text_view.h", ["mtoc2_string_t", "mtoc2_char_tensor_t"]),
  ],
  [
    "mtoc2_disp_text",
    loadSnippet("disp_text.h", ["mtoc2_text_view_t", "mtoc2_host_output"]),
  ],
  ["mtoc2_strcmp", loadSnippet("strcmp.h", ["mtoc2_text_view_t"])],
  ["mtoc2_strcmpi", loadSnippet("strcmp.h", ["mtoc2_text_view_t"])],
  // One snippet provides `mtoc2_dot_real` (vector → scalar),
  // `mtoc2_dot_real_matrix` (matrix → 1×N row vector), and the
  // `_complex` siblings (complex result for any-complex input).
  [
    "mtoc2_dot_real",
    loadSnippet("tensor_dot.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_cscalar",
      "mtoc2_tensor_alloc_nd_complex",
    ]),
  ],
  [
    "mtoc2_uniquetol_real",
    loadSnippet("tensor_uniquetol.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      // JS sibling allocates via the standard ND helper.
      "mtoc2_tensor_alloc_nd",
    ]),
  ],

  // ── Format engine + fprintf ───────────────────────────────────────
  // `format_engine.h` is the numbl-compatible printf walker shared by
  // fprintf and (future) sprintf. It declares `mtoc2_fprintf_arg_t`
  // (a tagged union) for arg transport; codegen builds a per-call
  // compound-literal array. Depends on the tensor type for the tensor-
  // flattening slot kind and on the text view for the format string.
  [
    "mtoc2_format_engine",
    loadSnippet("format_engine.h", ["mtoc2_text_view_t", "mtoc2_tensor_t"]),
  ],
  [
    "mtoc2_fprintf",
    loadSnippet("fprintf.h", ["mtoc2_format_engine", "mtoc2_host_output"]),
  ],
  ["mtoc2_error_fmt", loadSnippet("error_fmt.h", ["mtoc2_format_engine"])],
  // JS-aot identifier-bearing variant. The C side reuses
  // `mtoc2_error_fmt` (no identifier in the c-aot path) and ignores
  // this entry; the JS sibling is what threads the identifier
  // through to the interpreter's TryCatch handler.
  ["mtoc2_error_fmt_id", loadSnippet("error_fmt.h", ["mtoc2_format_engine"])],
  // Warning. Same shape as error but the c-aot path is real (no
  // throw); both runtimes prefix `Warning: ` and append `\n`.
  ["mtoc2_warning_fmt", loadSnippet("warning_fmt.h", ["mtoc2_format_engine"])],
  [
    "mtoc2_warning_fmt_id",
    loadSnippet("warning_fmt.h", ["mtoc2_format_engine"]),
  ],
  [
    "mtoc2_assert_scalar_fmt",
    loadSnippet("assert_fmt.h", ["mtoc2_format_engine"]),
  ],
  [
    "mtoc2_sprintf_str",
    loadSnippet("sprintf.h", [
      "mtoc2_format_engine",
      "mtoc2_string_t",
      "mtoc2_char_tensor_t",
    ]),
  ],
  [
    "mtoc2_sprintf_char",
    { headers: [], code: "", deps: ["mtoc2_sprintf_str"] },
  ],

  // JS-only scalar-index bounds-check helpers (paired with a stub
  // `.h`). The C path's scalar IndexLoad uses its own
  // `mtoc2_idx_lin` / `mtoc2_idx_axis` in `oob.h`; this snippet
  // exists so emitJs can activate the JS helpers without bringing
  // the C side along (the .h body is empty).
  ["mtoc2_scalar_index", loadSnippet("scalar_index.h")],
  // JS-only deep-clone helper (paired with a stub `.h`). The c-aot
  // path uses per-typedef `_copy` helpers from emitNamedTypedef for
  // the same purpose; on the JS side, owned-typed Var/MemberLoad/
  // capture-load RHSes wrap through this helper at emit time so a
  // subsequent member or indexed write through one alias doesn't
  // mutate the other.
  ["mtoc2_deep_clone", loadSnippet("deep_clone.h")],

  // ── Tensor (real, multi-element) ──────────────────────────────────
  // Storage shape + alloc + the four "owned value" helpers (copy,
  // assign, free, plus the `from_row`/`from_matrix` literal builders).
  // Matches mtoc's runtime — no refcount, no COW; the codegen
  // invariant is "every tensor RHS is freshly owned, every Var read
  // wraps in mtoc2_tensor_copy".
  ["mtoc2_alloc", loadSnippet("alloc.h")],
  ["mtoc2_tensor_t", loadSnippet("tensor.h")],
  [
    "mtoc2_tensor_alloc",
    loadSnippet("tensor_alloc.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],
  ["mtoc2_tensor_empty", loadSnippet("tensor_empty.h", ["mtoc2_tensor_t"])],
  [
    "mtoc2_tensor_copy",
    loadSnippet("tensor_copy.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_tensor_empty",
    ]),
  ],
  ["mtoc2_tensor_assign", loadSnippet("tensor_assign.h", ["mtoc2_tensor_t"])],
  ["mtoc2_tensor_free", loadSnippet("tensor_free.h", ["mtoc2_tensor_t"])],
  [
    "mtoc2_tensor_from_row",
    loadSnippet("tensor_from_row.h", ["mtoc2_tensor_alloc"]),
  ],
  [
    "mtoc2_tensor_from_matrix",
    loadSnippet("tensor_from_matrix.h", ["mtoc2_tensor_alloc"]),
  ],
  // ND alloc + fill helpers — additive to the 2-D `mtoc2_tensor_alloc`
  // fast path. `zeros`/`ones` builtins emit calls into these regardless
  // of rank (the C compiler inlines the small fill loop just fine).
  [
    "mtoc2_tensor_alloc_nd",
    loadSnippet("tensor_alloc_nd.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],
  [
    "mtoc2_tensor_zeros_nd",
    loadSnippet("tensor_zeros_nd.h", ["mtoc2_tensor_alloc_nd"]),
  ],
  [
    "mtoc2_tensor_ones_nd",
    loadSnippet("tensor_ones_nd.h", ["mtoc2_tensor_alloc_nd"]),
  ],
  // Single-eval companion to the `_nd` helpers for the
  // `zeros(n)` / `ones(n)` n×n shorthand when `n` is a runtime
  // expression. Taking the dim by parameter avoids duplicating any
  // side-effecting source-level expression.
  [
    "mtoc2_tensor_zeros_square",
    loadSnippet("tensor_zeros_square.h", ["mtoc2_tensor_zeros_nd"]),
  ],
  [
    "mtoc2_tensor_ones_square",
    loadSnippet("tensor_ones_square.h", ["mtoc2_tensor_ones_nd"]),
  ],
  // 2-D identity matrix. One file defines both the rectangular and
  // single-eval square entry points; the snippet is keyed under one
  // name (`mtoc2_tensor_eye`) and the `eye` builtin pulls it in via
  // a single dep.
  [
    "mtoc2_tensor_eye",
    loadSnippet("tensor_eye.h", ["mtoc2_tensor_alloc"]),
    // JS sibling allocates via `mtoc2_tensor_alloc` (already a dep).
  ],
  // Parameterized fill (used by the `nan` / `NaN` / `Inf` / `inf`
  // shape-constructor branch). The `_nd` helper takes the fill value
  // as a leading `double`; `_square` is the single-eval companion for
  // MATLAB's `nan(n)` n×n shorthand.
  [
    "mtoc2_tensor_fill_nd",
    loadSnippet("tensor_fill_nd.h", [
      "mtoc2_tensor_alloc_nd",
      "mtoc2_tensor_alloc_nd_complex",
    ]),
  ],
  [
    "mtoc2_tensor_fill_square",
    loadSnippet("tensor_fill_square.h", ["mtoc2_tensor_fill_nd"]),
  ],
  [
    "mtoc2_reshape_nd",
    loadSnippet("tensor_reshape_nd.h", [
      "mtoc2_tensor_t",
      "mtoc2_tensor_alloc_nd",
      // The JS sibling references `mtoc2_tensor_alloc_nd` too.
    ]),
  ],
  [
    "mtoc2_reshape_nd_complex",
    loadSnippet("tensor_reshape_nd_complex.h", [
      "mtoc2_tensor_t",
      "mtoc2_tensor_alloc_nd_complex",
    ]),
  ],
  // 2-D non-conjugate transpose. The `.'` and `'` unary operators
  // both map here (the conjugate variant matters only for complex
  // tensors, which mtoc2 doesn't yet have).
  [
    "mtoc2_tensor_transpose",
    loadSnippet("tensor_transpose.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      // JS sibling's `mtoc2_tensor_transpose` calls `mtoc2_tensor_alloc`.
      "mtoc2_tensor_alloc",
    ]),
  ],
  // Complex sibling of `mtoc2_tensor_transpose` — the non-conjugating
  // `.'` operator on a complex tensor. The conjugate `'` operator
  // lowers to `transpose(conj(z))` at the lowering layer (see
  // `index.ts::unaryOpBuiltin`), so this helper only sees the plain
  // transpose.
  [
    "mtoc2_tensor_transpose_complex",
    loadSnippet("tensor_transpose_complex.h", [
      "mtoc2_tensor_t",
      "mtoc2_tensor_alloc_nd_complex",
    ]),
  ],
  // Real 2-D matrix multiplication `A * B`. `mtimes` builtin
  // activates this snippet when both operands are tensors; the scalar
  // paths stay routed through the elementwise `times` snippet.
  [
    "mtoc2_tensor_mtimes_real",
    // The C path needs `mtoc2_alloc` only; the JS path's paired
    // `tensor_mtimes_real.js` calls `mtoc2_tensor_alloc` so we
    // declare it as a dep too. Activating an extra helper on the C
    // side is a benign no-op (the compiler doesn't emit unused
    // statics into the binary if dead-stripped).
    loadSnippet("tensor_mtimes_real.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_tensor_alloc",
    ]),
  ],
  [
    "mtoc2_tensor_mtimes_complex",
    loadSnippet("tensor_mtimes_complex.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],
  // `size(t)` row-vector form. The `size(t, k)` scalar form emits
  // inline C; this snippet covers only the variadic-shape return.
  [
    "mtoc2_tensor_size_row",
    loadSnippet("tensor_size.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],
  // Generic axis-flip: `flipud` → axis 0, `fliplr` → axis 1,
  // `flip(t, k)` → axis (k-1). All three source-level builtins share
  // this helper.
  [
    "mtoc2_tensor_flip",
    loadSnippet("tensor_flip.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      // JS sibling allocates via `mtoc2_tensor_alloc_nd`.
      "mtoc2_tensor_alloc_nd",
      // `flip_complex` (same file) needs the complex allocator.
      "mtoc2_tensor_alloc_nd_complex",
    ]),
  ],
  // `sort(a)` (single-output) and `[v, i] = sort(a)` (two-output).
  // Stable ascending sort over the column-major flat buffer; the
  // type system restricts the input to a 1-D vector for v1, but the
  // helper itself walks any shape.
  [
    "mtoc2_sort_real",
    loadSnippet("tensor_sort_real.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_tensor_assign",
      // JS sibling allocs via `mtoc2_tensor_alloc_nd`.
      "mtoc2_tensor_alloc_nd",
      // `sort_complex` (same file) needs the complex allocator.
      "mtoc2_tensor_alloc_nd_complex",
    ]),
  ],
  // `meshgrid(x, y)` — single-output (returns X) plus multi-output
  // `[X, Y] = meshgrid(x, y)` / `[X, Y] = meshgrid(x)`. One snippet
  // defines all three entry points (the 1-arg multi-output thunks
  // through the 2-arg version).
  [
    "mtoc2_meshgrid",
    loadSnippet("tensor_meshgrid.h", [
      "mtoc2_tensor_t",
      "mtoc2_tensor_alloc",
      "mtoc2_tensor_assign",
    ]),
  ],
  // `assert(cond, msg)` runtime check (scalar cond). The truthy fast
  // path returns immediately; failure writes the message to stderr
  // and aborts.
  ["mtoc2_assert_scalar", loadSnippet("assert_fail.h")],
  // `norm(v)` vector 2-norm. One snippet defines both `_real` and
  // `_complex` variants; the type-system decides which one to call.
  ["mtoc2_tensor_norm", loadSnippet("tensor_norm.h", ["mtoc2_tensor_t"])],
  // `isequal` over real numeric values. `_tt` (tensor vs tensor) and
  // `_st` (scalar vs tensor); scalar-vs-scalar is emitted inline.
  ["mtoc2_isequal", loadSnippet("isequal.h", ["mtoc2_tensor_t"])],
  // Tensor → logical-tensor predicates (`isnan`, `isinf`, `isfinite`,
  // `logical`) and their `_complex` siblings (same file).
  [
    "mtoc2_tensor_predicate",
    loadSnippet("tensor_predicate.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],
  // Prefix scans (`cumsum`, `cumprod`) — real and complex siblings
  // in one file; the builtin dispatches on op kind + input type.
  [
    "mtoc2_tensor_cumulative",
    loadSnippet("tensor_cumulative.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_tensor_alloc_nd",
      "mtoc2_tensor_alloc_nd_complex",
    ]),
  ],
  // `repmat` (real + complex sibling in the same file).
  [
    "mtoc2_tensor_repmat",
    loadSnippet("tensor_repmat.h", [
      "mtoc2_tensor_t",
      "mtoc2_tensor_alloc_nd",
      "mtoc2_tensor_alloc_nd_complex",
    ]),
  ],
  // `diag` (3 real entry points + 3 complex siblings in the same file).
  [
    "mtoc2_tensor_diag",
    loadSnippet("tensor_diag.h", [
      "mtoc2_tensor_t",
      "mtoc2_tensor_alloc",
      "mtoc2_tensor_alloc_nd_complex",
    ]),
  ],
  // `triu` / `tril` (real + complex siblings in the same file).
  [
    "mtoc2_tensor_triangular",
    loadSnippet("tensor_triangular.h", [
      "mtoc2_tensor_t",
      "mtoc2_tensor_alloc",
      "mtoc2_tensor_alloc_nd_complex",
    ]),
  ],
  // `cat` (real + complex sibling; complex variant uses
  // `mtoc2_cat_complex_arg_t`).
  [
    "mtoc2_tensor_cat",
    loadSnippet("tensor_cat.h", [
      "mtoc2_tensor_t",
      "mtoc2_tensor_alloc_nd",
      "mtoc2_tensor_alloc_nd_complex",
    ]),
  ],
  // `besselh(nu, 1, x)` for nu in {0, 1} via POSIX `j0/j1/y0/y1`.
  // One snippet defines both scalar and tensor entry points; the
  // builtin dispatches via the exact value of `nu`.
  [
    "mtoc2_tensor_besselh",
    loadSnippet("tensor_besselh.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],
  // Elementwise logical ops on tensors (`~` for real and complex).
  // `|` / `&` will share the same snippet when they land. Result
  // tensors are logical-typed; the storage is still `double` (the
  // type system carries the logical flag).
  [
    "mtoc2_tensor_logical_real",
    loadSnippet("tensor_logical_real.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],
  [
    "mtoc2_disp_tensor",
    loadSnippet("disp_tensor.h", [
      "mtoc2_tensor_t",
      "mtoc2_format_double",
      "mtoc2_host_output",
    ]),
  ],
  [
    "mtoc2_disp_tensor_inline",
    loadSnippet("disp_tensor_inline.h", [
      "mtoc2_tensor_t",
      "mtoc2_format_double",
      "mtoc2_host_output",
    ]),
  ],

  // JS-only generic struct disp. The C path generates per-typedef
  // <name>_disp via emitNamedTypedef and never activates this snippet;
  // it's the only entry point for `disp(struct)` on the JS side.
  ["mtoc2_disp_struct", loadSnippet("disp_struct.h")],

  // Cell array runtime. C-aot uses per-shape typedefs emitted from
  // `emitCellTypedef.ts`; the snippet here is the JS sibling shipped
  // to js-aot / interpreter. Basename is `cell` so the JS bundle
  // contains `cell.js`. The `.h` is a topic placeholder (see
  // `cell.h`) — no C registration needed.
  ["mtoc2_cell_make", loadSnippet("cell.h")],
  ["mtoc2_cell_empty", loadSnippet("cell.h")],
  ["mtoc2_format_cell", loadSnippet("cell.h")],

  // ── Complex tensor lifecycle ──────────────────────────────────────
  // Sibling helpers of the real tensor family for the complex-typed
  // path. `mtoc2_tensor_assign` / `_free` / `_empty` are shape-agnostic
  // — both lanes are freed unconditionally, so they handle real and
  // complex alike — but `_alloc` / `_copy` / `_from_row` / `_from_matrix`
  // need explicit complex variants that allocate both lanes.
  [
    "mtoc2_tensor_alloc_complex",
    loadSnippet("tensor_alloc_complex.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],
  [
    "mtoc2_tensor_alloc_nd_complex",
    loadSnippet("tensor_alloc_nd_complex.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],
  [
    "mtoc2_tensor_copy_complex",
    loadSnippet("tensor_copy_complex.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_tensor_empty",
    ]),
  ],
  [
    "mtoc2_tensor_from_row_complex",
    loadSnippet("tensor_from_row_complex.h", ["mtoc2_tensor_alloc_complex"]),
  ],
  [
    "mtoc2_tensor_from_matrix_complex",
    loadSnippet("tensor_from_matrix_complex.h", ["mtoc2_tensor_alloc_complex"]),
  ],
  [
    "mtoc2_disp_tensor_complex",
    loadSnippet("disp_tensor_complex.h", [
      "mtoc2_tensor_t",
      "mtoc2_format_complex",
      "mtoc2_cscalar",
      "mtoc2_host_output",
    ]),
  ],
  // Elementwise binary + unary ops on complex tensors. Mirrors the
  // real elemwise snippet's `_tt`/`_ts`/`_st`/`_bcast_tt` shape, per
  // op (plus/minus/times/rdivide) plus a unary `uminus`. Activated
  // when either operand of `+ - .* ./` has `isComplex` set; mixed
  // real+complex sites at the type-system layer route through these
  // helpers with the real operand wrapped in `mtoc2_cmake(x, 0)` at
  // emit time.
  [
    "mtoc2_tensor_elemwise_complex",
    loadSnippet("tensor_elemwise_complex.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_tensor_alloc_nd_complex",
      "mtoc2_cscalar",
      "mtoc2_cdiv",
    ]),
  ],
  // Elementwise unary math on complex tensors. Mirrors
  // `tensor_unary_real_math.h` but the per-op kernel routes through
  // `mtoc2_c*` wrappers in `cscalar.h` (rather than bare libm
  // `c*` calls) so c2js can substitute its `{re, im}` JS impls.
  // Activated by the per-builtin unary math files (sqrt, exp, log,
  // sin/cos/tan, atan, floor/ceil/round/fix, sign, abs) when the
  // input type carries `isComplex`.
  [
    "mtoc2_tensor_unary_complex_math",
    loadSnippet("tensor_unary_complex_math.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_tensor_alloc_nd",
      "mtoc2_tensor_alloc_nd_complex",
      "mtoc2_cscalar",
    ]),
  ],
  // Sum / prod / mean / min / max / any / all reductions on complex
  // tensors. Each builtin's transfer + codegen now dispatches on
  // `isComplex`; the complex path activates this snippet and emits
  // `mtoc2_<name>_complex_all` / `_complex_dim` calls.
  [
    "mtoc2_tensor_reduce_complex",
    loadSnippet("tensor_reduce_complex.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_tensor_alloc_nd",
      "mtoc2_tensor_alloc_nd_complex",
      "mtoc2_cscalar",
      "mtoc2_cdiv",
    ]),
  ],

  // ── Elementwise binary/unary on real tensors ──────────────────────
  // One snippet covers all 11 funcs (4×_tt, 4×_ts, 2×_st, 1×uminus).
  // Builtins activate by op-specific synthetic name; all map to the
  // same underlying snippet via dep, so dedupe is automatic.
  [
    "mtoc2_tensor_elemwise_real",
    loadSnippet("tensor_elemwise_real.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],

  // Elementwise binary functions on real tensors — `mod`, `rem`,
  // `atan2`, `hypot`. Sibling of `tensor_elemwise_real.h` but the
  // kernel is `FN(a,b)` instead of `a OP b`. Carried in its own
  // snippet so the four function builtins activate it independent
  // of the infix-op set.
  [
    "mtoc2_tensor_elemwise_real_fn",
    loadSnippet("tensor_elemwise_real_fn.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],

  // Elementwise unary math on real tensors (`cos`/`sin`/`sqrt`/`abs`/
  // `log`/…). One shared snippet defines `mtoc2_tensor_<name>` for
  // every per-name helper; every unary math builtin activates this
  // single snippet via its runtimeDeps.
  [
    "mtoc2_tensor_unary_real_math",
    loadSnippet("tensor_unary_real_math.h", ["mtoc2_tensor_t", "mtoc2_alloc"]),
  ],

  // ── Reductions (sum/prod/mean/min/max/any/all on real tensors) ────
  // One shared snippet defines every `_all` + `_dim` variant via C-side
  // macros. Reducer builtins all set `runtimeDeps: ["mtoc2_tensor_reduce_real"]`
  // directly (see `defineReducer` in builtins/reduction/_shape.ts).
  [
    "mtoc2_tensor_reduce_real",
    loadSnippet("tensor_reduce_real.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_tensor_alloc_nd",
    ]),
  ],
  // `var` / `std` two-pass reductions (mean then sum-of-squares).
  [
    "mtoc2_tensor_var",
    loadSnippet("tensor_var.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_tensor_alloc_nd",
    ]),
  ],
  // `diff(A)` / `diff(A, 1, dim)` — first-order forward difference.
  [
    "mtoc2_tensor_diff",
    loadSnippet("tensor_diff.h", [
      "mtoc2_tensor_t",
      "mtoc2_alloc",
      "mtoc2_tensor_alloc_nd",
    ]),
  ],

  // ── length / numel (tensor → scalar shape queries) ───────────────
  ["mtoc2_length", loadSnippet("length.h", ["mtoc2_tensor_t"])],
  ["mtoc2_numel", loadSnippet("numel.h", ["mtoc2_tensor_t"])],

  // ── Indexing / slicing / range-as-value ──────────────────────────
  // `mtoc2_loop_count` powers every `start:step:end` count computation
  // (single-slot slice reads/writes, multi-slot per-axis setup, and the
  // MakeRange value-form helper).
  ["mtoc2_loop_count", loadSnippet("loop_count.h")],
  ["mtoc2_range_value", loadSnippet("range_value.h")],
  // OOB family — `mtoc2_idx_axis`, `mtoc2_idx_lin`, and
  // `mtoc2_check_axis_range` share `mtoc2_oob_abort`; activated as
  // one snippet so any IndexLoad/IndexStore/IndexSlice site that
  // pulls in one bounds-check fn pulls in the others' definitions
  // for free.
  ["mtoc2_oob_abort", loadSnippet("oob.h", ["mtoc2_tensor_t"])],
  // Grow-aware indexed-STORE bounds checks (`mtoc2_idx_lin_grow`,
  // `mtoc2_idx_axis_grow`) plus the setjmp-based bail infra
  // (`mtoc2_grow_bail_reset` / `_check`). A store past the runtime
  // extent longjmp's to the host-entry function's guard for a clean
  // host-side bail to the interpreter (which grows the array). See
  // `grow_bail.h`. C-only (no `.js` sibling): the JS path's grow
  // helpers live in `scalar_index.js` and bail via a tagged throw.
  [
    "mtoc2_grow_bail",
    loadSnippet("grow_bail.h", ["mtoc2_tensor_t", "mtoc2_oob_abort"]),
  ],
  [
    "mtoc2_tensor_make_range",
    loadSnippet("tensor_make_range.h", [
      "mtoc2_tensor_t",
      "mtoc2_tensor_alloc_nd",
      "mtoc2_loop_count",
      "mtoc2_range_value",
    ]),
  ],
  // `linspace(a, b, n)` — 1×n row of linearly-spaced values from a to b.
  [
    "mtoc2_tensor_linspace",
    loadSnippet("tensor_linspace.h", ["mtoc2_tensor_t", "mtoc2_tensor_alloc"]),
  ],
  // `logspace(a, b, n)` — 1×n row of logarithmically-spaced values
  // from 10^a to 10^b.
  [
    "mtoc2_tensor_logspace",
    loadSnippet("tensor_logspace.h", ["mtoc2_tensor_t", "mtoc2_tensor_alloc"]),
  ],
  // Logical-mask indexing — `a(mask)` linear read/write and per-axis
  // `M(:, mask)` reads. The helper scans the mask column-major and
  // writes 0-based source indices into a caller-allocated buffer;
  // codegen-emitted setup blocks invoke it, then iterate the buffer
  // just like an IndexVec slot. Aborts on out-of-range truthy entries.
  [
    "mtoc2_logical_mask_indices",
    loadSnippet("tensor_logical_mask.h", ["mtoc2_tensor_t", "mtoc2_oob_abort"]),
  ],

  // ── tic / toc (wall-clock stopwatch) ─────────────────────────────
  // One snippet covers `mtoc2_tic`, `mtoc2_toc`, and the bare-`toc;`
  // print form `mtoc2_toc_print`. Activated by the `tic`/`toc`
  // builtins and (for the print form) by the lowerer when a bare
  // `toc;` ExprStmt synthesizes a direct Call to `mtoc2_toc_print`.
  ["mtoc2_tic_toc", loadSnippet("tictoc.h")],

  // ── plot dispatch ────────────────────────────────────────────────
  // Every plotting builtin (`plot`, `surf`, `imagesc`, `bar`,
  // `figure`, `hold`, `xlabel`, …) routes through this single helper
  // — see `plot_dispatch.h` for the wire format. Reuses the tagged
  // `mtoc2_fprintf_arg_t` ABI from format_engine.h so there is no
  // per-builtin C surface.
  [
    "mtoc2_plot_dispatch",
    loadSnippet("plot_dispatch.h", ["mtoc2_format_engine"]),
  ],
]);

/** Look up a snippet from the global registry. Throws if unregistered. */
function lookupGlobalSnippet(name: string): RuntimeSnippet {
  const s = REGISTRY.get(name);
  if (!s) {
    throw new Error(`runtime snippet '${name}' not registered`);
  }
  return s;
}

/** State-aware lookup. Consults the state's `extraSnippets` (registered
 *  by user `.mtoc2.js` builtins via `useRuntime({name, code, ...})`)
 *  first, then falls back to the global mtoc2 registry. */
function lookupSnippet(state: RuntimeState, name: string): RuntimeSnippet {
  const extra = state.extraSnippets.get(name);
  if (extra !== undefined) return extra;
  return lookupGlobalSnippet(name);
}

/** Public re-export for callers that don't have a `RuntimeState` (the
 *  build-snippets script). Resolves against the global registry only. */
export function getRuntimeSnippet(name: string): RuntimeSnippet {
  return lookupGlobalSnippet(name);
}

// ── Activation state + topological emit ────────────────────────────────

/** Inline snippet definition supplied by a user `.mtoc2.js` builtin's
 *  `emitC` / `emitJs` via `useRuntime({...})`. Once registered on a
 *  `RuntimeState`, subsequent activations of the same `name` are
 *  deduplicated.
 *
 *  `code` is the C body (required for `emitC` consumers); `jsCode`
 *  is the optional JS body (required for `emitJs` consumers but
 *  ignored on the C path). A builtin that supports both backends
 *  supplies both; one that only supports C may omit `jsCode`. */
export interface InlineSnippet {
  name: string;
  code: string;
  jsCode?: string;
  headers?: ReadonlyArray<string>;
  deps?: ReadonlyArray<string>;
}

export interface RuntimeState {
  /** Insertion-ordered set of activated snippet names. */
  active: Set<string>;
  /** Counter for `MultiAssignCall` discard-temp suffixes so adjacent
   *  multi-output call sites don't collide on `_mtoc2_discard_<N>_<i>`
   *  names. Incremented each time a multi-output call is emitted. */
  multiAssignCallCounter: number;
  /** Workspace-scoped snippets registered via the user `useRuntime`
   *  callback. Looked up by name alongside the global registry. */
  extraSnippets: Map<string, RuntimeSnippet>;
  /** Workspace reference for resolving names to `.mtoc2.js` user
   *  builtins during emit. Undefined for non-workspace-driven codegen
   *  (the snippet-build script, vitest unit tests that bypass the
   *  workspace). */
  workspace?: WorkspaceLike;
  /** JS-side helper: cName list of the currently-emitting user
   *  function's outputs. Set by `emitFunction` before walking the
   *  body and cleared after so `ReturnFromFunction` can emit a
   *  return matching the function's shape (bare value / array /
   *  nothing). Undefined at top level. */
  currentFnOutputs?: ReadonlyArray<string>;
  /** Set when the program contains a scalar indexed STORE, which
   *  emits the grow-aware `mtoc2_idx_*_grow` bounds checks. The
   *  host-entry function (storage class `""`) then arms a `setjmp`
   *  guard so a runtime grow (`v(k) = x`, k past the runtime extent)
   *  `longjmp`s back and returns cleanly for a host-side bail to the
   *  interpreter, instead of `exit(1)`. Computed by a pre-pass in
   *  `emit.ts` so it's set regardless of function emission order. */
  usedGrowBail?: boolean;
}

export function newRuntimeState(workspace?: WorkspaceLike): RuntimeState {
  return {
    active: new Set(),
    multiAssignCallCounter: 0,
    extraSnippets: new Map(),
    ...(workspace !== undefined ? { workspace } : {}),
  };
}

/** Look up a `Builtin` for an emit-time call. Consults the workspace's
 *  `.mtoc2.js` user builtins first (so a `foo.mtoc2.js` overrides any
 *  identically-named built-in for callers in this workspace), then
 *  falls back to the global registry. */
export function lookupBuiltin(
  state: RuntimeState,
  name: string
): Builtin | undefined {
  const user = state.workspace?.getUserBuiltin(name);
  if (user !== undefined) return user;
  return getBuiltin(name);
}

/** Reverse map: `.h` basename → registered snippet name. Built lazily
 *  the first time `useRuntimeByName` resolves JS-import deps, since
 *  the registry must be fully constructed before this map is built.
 *  Identifies which snippet supplies each cross-snippet JS import. */
let H_FILENAME_TO_NAME: Map<string, string> | undefined;

function getFilenameToName(): Map<string, string> {
  if (H_FILENAME_TO_NAME !== undefined) return H_FILENAME_TO_NAME;
  const m = new Map<string, string>();
  for (const [name, snip] of REGISTRY.entries()) {
    if (snip.srcFilename !== undefined) {
      m.set(snip.srcFilename, name);
    }
  }
  H_FILENAME_TO_NAME = m;
  return m;
}

/** Activate a snippet by its registered name. Pulls its `deps` in first
 *  so they appear before it in the final emission order. Also pulls in
 *  JS-import deps inferred from the snippet's paired `.js` sibling
 *  (built by `scripts/build_runtime_snippets.ts` into `JS_IMPORTS`),
 *  so cross-snippet JS calls resolve at runtime even when the C deps
 *  field doesn't enumerate them. Idempotent. */
export function useRuntimeByName(state: RuntimeState, name: string): void {
  if (state.active.has(name)) return;
  const s = lookupSnippet(state, name);
  for (const d of s.deps) useRuntimeByName(state, d);
  if (s.srcFilename !== undefined) {
    const jsName = s.srcFilename.replace(/\.h$/, ".js");
    const imports = JS_IMPORTS[jsName];
    if (imports !== undefined && imports.length > 0) {
      const map = getFilenameToName();
      for (const importedBasename of imports) {
        const hName = importedBasename.replace(/\.js$/, ".h");
        const depName = map.get(hName);
        if (depName !== undefined) useRuntimeByName(state, depName);
      }
    }
  }
  state.active.add(name);
}

/** Register-and-activate an inline snippet supplied by a user builtin.
 *  First call registers the snippet on the state; subsequent calls
 *  (with the same name) are no-ops at the registration step and just
 *  re-activate through the global pathway. */
export function useRuntimeInline(
  state: RuntimeState,
  snippet: InlineSnippet
): void {
  if (!state.extraSnippets.has(snippet.name)) {
    const stored: RuntimeSnippet = {
      headers: snippet.headers ?? [],
      code: snippet.code,
      deps: snippet.deps ?? [],
    };
    if (snippet.jsCode !== undefined) stored.jsCode = snippet.jsCode;
    state.extraSnippets.set(snippet.name, stored);
  }
  useRuntimeByName(state, snippet.name);
}

/** Build the `useRuntime` callback passed into a `Builtin.emit` call.
 *  Accepts either a name (resolved against the global mtoc2 runtime
 *  registry) or an `InlineSnippet` (registered on the current state).
 *  Centralized here so every emit call site activates snippets
 *  uniformly. */
export function makeEmitUseRuntime(
  state: RuntimeState
): (spec: string | InlineSnippet) => void {
  return spec => {
    if (typeof spec === "string") {
      useRuntimeByName(state, spec);
    } else {
      useRuntimeInline(state, spec);
    }
  };
}

/** All headers required by the activated snippets, in insertion order. */
export function collectRuntimeHeaders(state: RuntimeState): string[] {
  const headers = new Set<string>();
  for (const name of state.active) {
    const s = lookupSnippet(state, name);
    for (const h of s.headers) {
      // Headers are emitted verbatim as `#include ${h}`, so each must
      // already carry its delimiters (`<stdio.h>` or `"foo.h"`). A bare
      // `stdio.h` would emit `#include stdio.h` — invalid C. File-loaded
      // snippets get this for free from parseSnippetSource; inline
      // snippets declare `headers` by hand and have silently shipped the
      // unbracketed form, crashing the C compile instead of the include
      // deduping against BASE_HEADERS.
      if (!/^(<[^>]+>|"[^"]+")$/.test(h)) {
        throw new Error(
          `runtime snippet '${name}': header ${JSON.stringify(h)} must be ` +
            `bracketed (<header.h>) or quoted ("header.h")`
        );
      }
      headers.add(h);
    }
  }
  return Array.from(headers);
}

/** Concatenated bodies (definitions) of activated snippets, in
 *  activation (dep-respecting) order. */
export function renderRuntimeBodies(state: RuntimeState): string {
  if (state.active.size === 0) return "";
  const bodies: string[] = [];
  for (const name of state.active) {
    const s = lookupSnippet(state, name);
    bodies.push(s.code);
  }
  return bodies.join("\n");
}

/** JS counterpart of `renderRuntimeBodies` — concatenates the `jsCode`
 *  fields of activated snippets, in dependency order, skipping
 *  snippets that have no JS body. The result is plain JS source
 *  ready to inline at the top of the emitted module so subsequent
 *  call-site code can reference helpers by bare name.
 *
 *  Symmetric with the C side: the same `state.active` set drives
 *  both renderers — a builtin activates `mtoc2_disp_double` once,
 *  the C path emits `disp_double.h`'s body, the JS path emits
 *  `disp_double.js`'s body. */
export function renderJsRuntimeBodies(state: RuntimeState): string {
  if (state.active.size === 0) return "";
  const bodies: string[] = [];
  for (const name of state.active) {
    const s = lookupSnippet(state, name);
    if (s.jsCode !== undefined) bodies.push(s.jsCode);
  }
  return bodies.join("\n");
}

/** Stable always-emitted header set — every translation includes these
 *  so the user-level code can rely on stdio/math without each builtin
 *  re-declaring them. Kept tiny on purpose. */
export const BASE_HEADERS: ReadonlyArray<string> = [
  "<stdio.h>",
  "<stdlib.h>",
  "<math.h>",
  "<complex.h>",
];

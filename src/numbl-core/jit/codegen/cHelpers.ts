/**
 * Shared C-type and owned-helper renderers used by both `emit.ts` and
 * `emitNamedTypedef.ts`. Lives in its own module so the two emitters
 * don't have to import each other (the previous arrangement had
 * `emitNamedTypedef.ts` duplicating these helpers to avoid a circular
 * import).
 */

import {
  cellTypedefName,
  classTypedefName,
  handleTypedefName,
  isCell,
  isHandle,
  isMultiElement,
  isNumeric,
  isScalar,
  isScalarRealNumeric,
  structTypedefName,
  type Type,
} from "../lowering/types.js";

/** C type for any JIT IR-typed value: tensor / string / char →
 *  per-kind runtime struct; struct / class / handle → per-shape
 *  generated typedef; everything else (scalar real numeric) → `double`. */
export function cTypeFor(t: Type): string {
  if (isMultiElement(t)) return "jit_tensor_t";
  if (isHandle(t)) return handleTypedefName(t);
  if (t.kind === "Struct") return structTypedefName(t);
  if (t.kind === "Class") return classTypedefName(t);
  if (t.kind === "Cell") return cellTypedefName(t);
  if (t.kind === "String") return "jit_string_t";
  if (t.kind === "Char") return "jit_char_tensor_t";
  if (t.kind === "Numeric" && t.isComplex) return "double _Complex";
  return "double";
}

/** Per-owned-kind helper-name family. Tensors / strings / chars use
 *  the global `jit_*` runtime snippet names; structs, classes, and
 *  handles use their program-emitted `<typedef>_*` family. */
export interface OwnedHelpers {
  empty: string;
  assign: string;
  copy: string;
  free: string;
  /** When true, `empty`/`assign`/`copy`/`free` are loaded from the
   *  runtime snippet registry. When false, they're emitted directly
   *  into the generated C by `emitNamedTypedef` and need no
   *  `useRuntimeByName` activation. */
  isRuntime: boolean;
}

/** Owned-helper family for any type. Returns `null` for non-owned
 *  (POD) types — those get plain assignment / no copy / no free. */
export function ownedHelpersFor(t: Type): OwnedHelpers | null {
  if (isMultiElement(t)) {
    // `_empty` / `_assign` / `_free` are shape-agnostic — they touch
    // both lanes unconditionally with `free(NULL)` no-ops covering the
    // real case. `_copy` must dispatch on `isComplex` because it has
    // to know how many lanes to memcpy.
    const isComplex = t.kind === "Numeric" && t.isComplex;
    return {
      empty: "jit_tensor_empty",
      assign: "jit_tensor_assign",
      copy: isComplex ? "jit_tensor_copy_complex" : "jit_tensor_copy",
      free: "jit_tensor_free",
      isRuntime: true,
    };
  }
  if (t.kind === "String") {
    return {
      empty: "jit_string_empty",
      assign: "jit_string_assign",
      copy: "jit_string_copy",
      free: "jit_string_free",
      isRuntime: true,
    };
  }
  if (t.kind === "Char") {
    return {
      empty: "jit_char_tensor_empty",
      assign: "jit_char_tensor_assign",
      copy: "jit_char_tensor_copy",
      free: "jit_char_tensor_free",
      isRuntime: true,
    };
  }
  if (t.kind === "Struct") {
    const name = structTypedefName(t);
    return {
      empty: `${name}_empty`,
      assign: `${name}_assign`,
      copy: `${name}_copy`,
      free: `${name}_free`,
      isRuntime: false,
    };
  }
  if (t.kind === "Class") {
    const name = classTypedefName(t);
    return {
      empty: `${name}_empty`,
      assign: `${name}_assign`,
      copy: `${name}_copy`,
      free: `${name}_free`,
      isRuntime: false,
    };
  }
  if (t.kind === "Handle") {
    const name = handleTypedefName(t);
    return {
      empty: `${name}_empty`,
      assign: `${name}_assign`,
      copy: `${name}_copy`,
      free: `${name}_free`,
      isRuntime: false,
    };
  }
  if (t.kind === "Cell") {
    const name = cellTypedefName(t);
    return {
      empty: `${name}_empty`,
      assign: `${name}_assign`,
      copy: `${name}_copy`,
      free: `${name}_free`,
      isRuntime: false,
    };
  }
  return null;
}

/** Asserting form: returns helpers for a type known to be owned, and
 *  throws if it isn't. Use this at sites where the caller has already
 *  proven the type is owned (e.g. via `isOwned(t)`) and the null check
 *  would be a redundant nuisance. */
export function requireOwnedHelpers(t: Type): OwnedHelpers {
  const h = ownedHelpersFor(t);
  if (h === null) {
    throw new Error(`requireOwnedHelpers: non-owned type ${t.kind}`);
  }
  return h;
}

/** C expression that disps one value of type `t`, including the
 *  trailing newline that matches numbl's `disp` contract. Activates
 *  any runtime snippets the chosen helper needs via `useRuntime`.
 *  Returns `null` when no disp helper exists for the type (Class /
 *  Handle / Void / Unknown — call sites decide whether to skip or
 *  raise). Single dispatch shared by the `disp` builtin's `emitC`
 *  and the per-field disp inside struct typedefs, so the two can't
 *  drift apart when a new value kind becomes a valid struct field. */
export function emitDispCallC(
  t: Type,
  valueC: string,
  useRuntime: (name: string) => void
): string | null {
  if (t.kind === "Struct") {
    return `${structTypedefName(t)}_disp(${valueC})`;
  }
  if (isCell(t)) {
    // The cell typedef's `_disp` renders `{e1, e2, ...}` without a
    // trailing newline; pair it with an explicit newline via a comma
    // expression so the value form has the same statement shape as the
    // other helpers (one expression, prints + newline). The newline goes
    // through jit_stdout — NOT raw printf — so it shares the host-write
    // channel with the cell content (a raw printf would flush separately
    // and float to the front of captured --opt 2 output).
    useRuntime("jit_host_output");
    return `(${cellTypedefName(t)}_disp(${valueC}), jit_stdout_s("\\n"))`;
  }
  if (t.kind === "String") {
    useRuntime("jit_disp_text");
    return `jit_disp_text(jit_text_from_string(${valueC}))`;
  }
  if (t.kind === "Char") {
    useRuntime("jit_disp_text");
    return `jit_disp_text(jit_text_from_char_tensor(${valueC}))`;
  }
  if (isNumeric(t) && t.isComplex && isScalar(t)) {
    useRuntime("jit_disp_complex");
    return `jit_disp_complex(${valueC})`;
  }
  if (isNumeric(t) && t.isComplex && !isScalar(t)) {
    useRuntime("jit_disp_tensor_complex");
    return `jit_disp_tensor_complex(${valueC})`;
  }
  if (isNumeric(t) && !isScalarRealNumeric(t)) {
    useRuntime("jit_disp_tensor");
    return `jit_disp_tensor(${valueC})`;
  }
  if (isScalarRealNumeric(t)) {
    useRuntime("jit_disp_double");
    return `jit_disp_double(${valueC})`;
  }
  return null;
}

/** Format a JS number as a C double literal that round-trips. Single
 *  source of truth — both the per-expression emit path and the fused
 *  per-element template emitter route through here so NaN / ±Infinity
 *  format consistently as `NAN` / `INFINITY` / `(-INFINITY)`. */
export function formatDouble(v: number): string {
  if (Number.isNaN(v)) return "NAN";
  if (!Number.isFinite(v)) return v > 0 ? "INFINITY" : "(-INFINITY)";
  if (Number.isInteger(v) && Math.abs(v) < 1e16) {
    return `${v}.0`;
  }
  const s = v.toString();
  if (!s.includes(".") && !s.includes("e") && !s.includes("E")) {
    return `${s}.0`;
  }
  return s;
}

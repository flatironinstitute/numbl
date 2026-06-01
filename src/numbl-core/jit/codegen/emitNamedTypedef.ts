/**
 * Per-shape struct/class typedef + owned-helper emission.
 *
 * Structs and class instances live as generated C `typedef struct
 * <name> { ... } <name>;` definitions, one per distinct canonical
 * shape. Each typedef ships with the four owned-kind helpers —
 * `_empty()`, `_assign()`, `_copy()`, `_free()` — so the existing
 * codegen pipeline (predeclare at function top, scope-exit free,
 * early-free, ANF) treats struct/class locals uniformly with tensors.
 *
 * Structs additionally get a `_disp()` helper used by `disp(s)`; v1
 * doesn't emit `_disp` for class instances.
 *
 * The renderer walks each field once and emits the right helper
 * pattern for that field:
 *   - scalar (Numeric / Handle / Unknown): plain C assignment / no
 *     free / no recursive copy.
 *   - owned (multi-element Numeric / nested Struct / nested Class):
 *     route through the field's own owned-kind helper family.
 *
 * Tensor fields call `mtoc2_tensor_*` directly; nested struct/class
 * fields call the corresponding `<inner-typedef>_*` helper, so a
 * topological order over the typedef-dependency graph guarantees each
 * helper is defined before the one that uses it.
 */

import {
  classTypedefName,
  handleTypedefName,
  structTypedefName,
  type ClassType,
  type HandleType,
  type StructType,
  type Type,
} from "../lowering/types.js";
import { cTypeFor, emitDispCallC, ownedHelpersFor } from "./cHelpers.js";
import { useRuntimeByName, type RuntimeState } from "./runtime.js";

export interface NamedTypedefSpec {
  /** Typedef name. The C output uses this as both the struct tag and
   *  the typedef alias. */
  name: string;
  /** Field list — canonical order (sorted by name). */
  fields: ReadonlyArray<{ name: string; ty: Type }>;
  /** Emit a `_disp(value)` helper (true for structs, false for
   *  classes in v1). */
  emitDisp: boolean;
  /** Source description for the function-leading comment (e.g.
   *  `struct{x:double,y:double}` or `class Foo`). */
  pretty: string;
}

/** Emit the typedef and its four (or five) owned-kind helpers as
 *  one block of C source. Activates any runtime snippets the helpers
 *  call into (e.g. `mtoc2_tensor_copy`). */
export function emitNamedTypedef(
  spec: NamedTypedefSpec,
  state: RuntimeState
): string {
  // Pull in field-level runtime helpers. Owned fields whose helpers
  // live in the runtime snippet registry (tensor / string / char)
  // activate all four; nested struct/class/handle/cell fields don't
  // need runtime activation — their helpers are program-emitted and
  // ordered by the typedef topological sort.
  for (const f of spec.fields) {
    const ops = ownedHelpersFor(f.ty);
    if (ops !== null && ops.isRuntime) {
      useRuntimeByName(state, ops.empty);
      useRuntimeByName(state, ops.assign);
      useRuntimeByName(state, ops.copy);
      useRuntimeByName(state, ops.free);
    }
  }

  const isEmpty = spec.fields.length === 0;

  const lines: string[] = [];
  lines.push(`/* ${spec.pretty} */`);
  lines.push(`typedef struct ${spec.name} {`);
  if (isEmpty) {
    // C99 forbids empty structs; emit a placeholder field. Used by
    // no-capture function handles (`mtoc2_handle_empty_t`).
    lines.push(`  char _placeholder;`);
  } else {
    for (const f of spec.fields) {
      lines.push(`  ${cTypeFor(f.ty)} ${f.name};`);
    }
  }
  lines.push(`} ${spec.name};`);
  lines.push("");

  // _empty(): zero-initialize every field. Owned fields get their
  // field-type _empty() (NULL pointers for tensors; recursive empty
  // for nested struct/class/handle). Scalars default to 0.0.
  lines.push(`static ${spec.name} ${spec.name}_empty(void) {`);
  lines.push(`  ${spec.name} v;`);
  if (isEmpty) {
    lines.push(`  v._placeholder = 0;`);
  } else {
    for (const f of spec.fields) {
      const ops = ownedHelpersFor(f.ty);
      if (ops !== null) {
        lines.push(`  v.${f.name} = ${ops.empty}();`);
      } else {
        lines.push(`  v.${f.name} = 0.0;`);
      }
    }
  }
  lines.push(`  return v;`);
  lines.push(`}`);
  lines.push("");

  // _free(): release every owned field. Scalars are POD — skip them.
  lines.push(`static void ${spec.name}_free(${spec.name} *p) {`);
  if (isEmpty) {
    lines.push(`  (void)p;`);
  } else {
    for (const f of spec.fields) {
      const ops = ownedHelpersFor(f.ty);
      if (ops !== null) {
        lines.push(`  ${ops.free}(&p->${f.name});`);
      }
    }
  }
  lines.push(`}`);
  lines.push("");

  // _copy(): deep copy every owned field via its _copy; scalars plain-
  // assign.
  lines.push(`static ${spec.name} ${spec.name}_copy(${spec.name} v) {`);
  lines.push(`  ${spec.name} out;`);
  if (isEmpty) {
    lines.push(`  out._placeholder = v._placeholder;`);
  } else {
    for (const f of spec.fields) {
      const ops = ownedHelpersFor(f.ty);
      if (ops !== null) {
        lines.push(`  out.${f.name} = ${ops.copy}(v.${f.name});`);
      } else {
        lines.push(`  out.${f.name} = v.${f.name};`);
      }
    }
  }
  lines.push(`  return out;`);
  lines.push(`}`);
  lines.push("");

  // _assign(): consume-replace. Free the prior owned slot, then move
  // the rhs into place. The rhs is expected to already be a freshly-
  // owned value (the calling convention is "consume the rhs").
  lines.push(
    `static void ${spec.name}_assign(${spec.name} *p, ${spec.name} rhs) {`
  );
  lines.push(`  ${spec.name}_free(p);`);
  lines.push(`  *p = rhs;`);
  lines.push(`}`);
  lines.push("");

  if (spec.emitDisp) {
    lines.push(...emitStructDisp(spec, state));
  }

  return lines.join("\n");
}

/** Emit `<name>_disp(v)` — prints a `key: value` line per field. The
 *  format is intentionally narrow: it matches what numbl's `disp` on
 *  a scalar struct produces for the common scalar / tensor field
 *  shapes (so the cross-runner stays byte-exact). Anything outside
 *  that subset would fail at lowering, not here. */
function emitStructDisp(spec: NamedTypedefSpec, state: RuntimeState): string[] {
  // Each `disp_double`/`disp_tensor` call pulls in its own snippet on
  // demand; nested struct/class disps recurse via the inner helper
  // (only structs emit disp — class fields are rejected at this
  // codegen site since v1 forbids class properties of class type).
  const lines: string[] = [];
  lines.push(`static void ${spec.name}_disp(${spec.name} v) {`);
  if (spec.fields.length === 0) {
    // numbl prints nothing for a zero-field struct; mirror that.
    lines.push(`  (void)v;`);
    lines.push(`}`);
    lines.push("");
    return lines;
  }
  // numbl's `disp(struct)` format is:
  //   `    <name>: <value>\n`
  // where the value is rendered the same way `disp(value)` would on
  // its own — so for a tensor the row data follows the colon on the
  // same line (with the tensor's leading whitespace intact) and
  // subsequent rows wrap to their own lines. We share the per-value
  // dispatch with the `disp` builtin via `emitDispCallC` so a new
  // field kind only has to land in one place.
  const useRt = (name: string) => useRuntimeByName(state, name);
  // Field labels / framing must go through the same host-write channel
  // as the values (mtoc2_disp_double / disp_tensor) — a raw printf would
  // land on libc stdout while values route through mtoc2_host_write,
  // reordering the output under a captured/streamed --opt 2 run.
  useRt("mtoc2_host_output");
  for (const f of spec.fields) {
    if (f.ty.kind === "Class") {
      // Class / handle disp not supported in v1; emit nothing rather
      // than failing — lowering rejects `disp(class_instance)` /
      // `disp(handle)` up front so this path is unreachable for
      // well-formed input.
      lines.push(`  /* skipping class-typed field '${f.name}' in disp */`);
      continue;
    }
    if (f.ty.kind === "Handle") {
      lines.push(`  /* skipping handle-typed field '${f.name}' in disp */`);
      continue;
    }
    lines.push(`  mtoc2_stdout_s("    ${f.name}: ");`);
    if (f.ty.kind === "Struct") {
      // A nested struct's own _disp emits its own header/leading
      // newline; numbl prints a blank then the nested fields.
      lines.push(`  mtoc2_stdout_s("\\n");`);
      lines.push(`  ${structTypedefName(f.ty)}_disp(v.${f.name});`);
      continue;
    }
    const call = emitDispCallC(f.ty, `v.${f.name}`, useRt);
    if (call === null) {
      // Defensive: lowering accepts any non-Void / non-Unknown
      // value as a struct field, so a null here means a new owned
      // kind was added to the type lattice without a paired disp
      // dispatch. Emit a placeholder comment to keep the C valid;
      // user-visible disp output is a separate concern from the
      // typedef compiling.
      lines.push(
        `  /* skipping ${f.ty.kind}-typed field '${f.name}' in disp */`
      );
      continue;
    }
    lines.push(`  ${call};`);
  }
  lines.push(`}`);
  lines.push("");
  return lines;
}

/** Helper for emit.ts: build a `NamedTypedefSpec` for a `StructType`. */
export function specForStruct(t: StructType): NamedTypedefSpec {
  return {
    name: structTypedefName(t),
    fields: t.fields.map(f => ({ name: f.name, ty: f.ty })),
    emitDisp: true,
    pretty: `struct{${t.fields.map(f => f.name).join(", ")}}`,
  };
}

/** Helper for emit.ts: build a `NamedTypedefSpec` for a `ClassType`. */
export function specForClass(t: ClassType): NamedTypedefSpec {
  return {
    name: classTypedefName(t),
    fields: t.properties.map(p => ({ name: p.name, ty: p.ty })),
    emitDisp: false,
    pretty: `class ${t.className}`,
  };
}

/** Helper for emit.ts: build a `NamedTypedefSpec` for a `HandleType`.
 *  Captures map to fields named `cap_<captureName>` — matching the
 *  layout that `HandleCaptureLoad` codegen reads. The no-capture
 *  shape emits as a placeholder-bearing struct (handled by the empty-
 *  fields branch in `emitNamedTypedef`). Handles never emit `_disp` —
 *  `disp(handle)` is rejected at lowering. */
export function specForHandle(t: HandleType): NamedTypedefSpec {
  return {
    name: handleTypedefName(t),
    fields: t.captures.map(c => ({ name: `cap_${c.name}`, ty: c.ty })),
    emitDisp: false,
    pretty:
      t.captures.length === 0
        ? `handle @${t.targetName}`
        : `handle @${t.targetName}{${t.captures.map(c => c.name).join(", ")}}`,
  };
}

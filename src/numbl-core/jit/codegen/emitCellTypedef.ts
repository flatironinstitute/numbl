/**
 * Per-shape cell-array typedef + owned-helper emission. Sibling of
 * `emitNamedTypedef` (which covers struct / class / handle). Cells
 * differ enough that they get their own emitter:
 *
 *  - **tuple-mode** cells have one C field per slot
 *    (`slot_0`, `slot_1`, …, `slot_{n-1}`) — direct analogue of a
 *    struct with positional fields. Helpers walk each slot once.
 *  - **uniform-mode** cells have a small descriptor (`ndim`, `dims[]`,
 *    `slots[]`) and a heap-allocated slot buffer. Helpers loop over
 *    the slot count, calling the elem type's owned helpers on each.
 *
 *  Both modes ship the same five-helper family used elsewhere
 *  (`_empty / _assign / _copy / _free / _disp`).
 *
 *  Output format for `disp(c)` matches numbl's `formatCell`
 *  (`runtime/display.ts:244-251`): single line `{e1, e2, ...}` with
 *  chars in `'...'`, strings in `"..."`, other values via the
 *  scalar / tensor / struct disp helper. Empty cell renders as `{}`.
 */

import {
  cellTypedefName,
  isNumeric,
  isMultiElement,
  shapeNumel,
  type CellType,
  type Type,
} from "../lowering/types.js";
import { cTypeFor, ownedHelpersFor } from "./cHelpers.js";
import { useRuntimeByName, type RuntimeState } from "./runtime.js";

export function emitCellTypedef(t: CellType, state: RuntimeState): string {
  const name = cellTypedefName(t);
  if (t.mode === "tuple") return emitCellTupleTypedef(t, name, state);
  return emitCellUniformTypedef(t, name, state);
}

function emitCellTupleTypedef(
  t: CellType,
  name: string,
  state: RuntimeState
): string {
  const elements = t.elements ?? [];
  const slotCount = elements.length;
  const pretty = `cell[${(t.shape ?? []).join("×")}]`;
  const lines: string[] = [];

  // Activate runtime helpers for any tensor / char / string slots.
  for (const el of elements) activateSlotRuntime(el, state);

  lines.push(`/* ${pretty} (tuple) */`);
  lines.push(`typedef struct ${name} {`);
  if (slotCount === 0) {
    lines.push(`  char _placeholder;`);
  } else {
    for (let i = 0; i < slotCount; i++) {
      lines.push(`  ${cTypeFor(elements[i])} slot_${i};`);
    }
  }
  lines.push(`} ${name};`);
  lines.push("");

  // _empty
  lines.push(`static ${name} ${name}_empty(void) {`);
  lines.push(`  ${name} v;`);
  if (slotCount === 0) {
    lines.push(`  v._placeholder = 0;`);
  } else {
    for (let i = 0; i < slotCount; i++) {
      const ops = ownedHelpersFor(elements[i]);
      if (ops !== null) {
        lines.push(`  v.slot_${i} = ${ops.empty}();`);
      } else {
        lines.push(`  v.slot_${i} = 0.0;`);
      }
    }
  }
  lines.push(`  return v;`);
  lines.push(`}`);
  lines.push("");

  // _free
  lines.push(`static void ${name}_free(${name} *p) {`);
  if (slotCount === 0) {
    lines.push(`  (void)p;`);
  } else {
    for (let i = 0; i < slotCount; i++) {
      const ops = ownedHelpersFor(elements[i]);
      if (ops !== null) {
        lines.push(`  ${ops.free}(&p->slot_${i});`);
      }
    }
  }
  lines.push(`}`);
  lines.push("");

  // _copy
  lines.push(`static ${name} ${name}_copy(${name} v) {`);
  lines.push(`  ${name} out;`);
  if (slotCount === 0) {
    lines.push(`  out._placeholder = v._placeholder;`);
  } else {
    for (let i = 0; i < slotCount; i++) {
      const ops = ownedHelpersFor(elements[i]);
      if (ops !== null) {
        lines.push(`  out.slot_${i} = ${ops.copy}(v.slot_${i});`);
      } else {
        lines.push(`  out.slot_${i} = v.slot_${i};`);
      }
    }
  }
  lines.push(`  return out;`);
  lines.push(`}`);
  lines.push("");

  // _assign
  lines.push(`static void ${name}_assign(${name} *p, ${name} rhs) {`);
  lines.push(`  ${name}_free(p);`);
  lines.push(`  *p = rhs;`);
  lines.push(`}`);
  lines.push("");

  // _disp: `{e1, e2, ...}` — numbl format.
  lines.push(`static void ${name}_disp(${name} v) {`);
  if (slotCount === 0) {
    lines.push(`  printf("{}");`);
    lines.push(`}`);
    lines.push("");
    return lines.join("\n");
  }
  lines.push(`  printf("{");`);
  for (let i = 0; i < slotCount; i++) {
    if (i > 0) lines.push(`  printf(", ");`);
    lines.push(...emitSlotInlineDisp(elements[i], `v.slot_${i}`, state));
  }
  lines.push(`  printf("}");`);
  lines.push(`}`);
  lines.push("");

  return lines.join("\n");
}

function emitCellUniformTypedef(
  t: CellType,
  name: string,
  state: RuntimeState
): string {
  const elem = t.elem;
  if (elem === undefined) {
    throw new Error(`emitCellUniformTypedef: uniform cell has no elem type`);
  }
  const pretty = `cell[${(t.shape ?? t.dims.map(d => (d.kind === "exact" ? d.value : "?"))).join("×")}] uniform`;
  const lines: string[] = [];

  activateSlotRuntime(elem, state);

  // The slot buffer is an array of the elem's C representation. Each
  // slot follows the same owned-value protocol as a struct field of
  // that type: `_empty` zero-init via the slot's `_empty` helper,
  // `_free` releases each owned slot then frees the buffer, `_copy`
  // walks each slot through the elem's `_copy`.
  lines.push(`/* ${pretty} */`);
  lines.push(`typedef struct ${name} {`);
  lines.push(`  size_t ndim;`);
  lines.push(`  size_t dims[8]; /* MTOC2_MAX_NDIM */`);
  lines.push(`  size_t nslots;`);
  lines.push(`  ${cTypeFor(elem)} *slots;`);
  lines.push(`} ${name};`);
  lines.push("");

  // _empty: zero-rank, no slots, NULL buffer.
  lines.push(`static ${name} ${name}_empty(void) {`);
  lines.push(`  ${name} v;`);
  lines.push(`  v.ndim = 0;`);
  lines.push(`  v.nslots = 0;`);
  lines.push(`  v.slots = NULL;`);
  lines.push(`  for (size_t _i = 0; _i < 8; _i++) v.dims[_i] = 0;`);
  lines.push(`  return v;`);
  lines.push(`}`);
  lines.push("");

  // _free: walk owned slots, free each, then free the buffer. NULL-safe.
  lines.push(`static void ${name}_free(${name} *p) {`);
  lines.push(`  if (p->slots != NULL) {`);
  const ops = ownedHelpersFor(elem);
  if (ops !== null) {
    lines.push(`    for (size_t _i = 0; _i < p->nslots; _i++) {`);
    lines.push(`      ${ops.free}(&p->slots[_i]);`);
    lines.push(`    }`);
  }
  lines.push(`    free(p->slots);`);
  lines.push(`    p->slots = NULL;`);
  lines.push(`  }`);
  lines.push(`  p->ndim = 0;`);
  lines.push(`  p->nslots = 0;`);
  lines.push(`}`);
  lines.push("");

  // _copy: alloc fresh buffer, deep-copy each slot.
  lines.push(`static ${name} ${name}_copy(${name} v) {`);
  lines.push(`  ${name} out;`);
  lines.push(`  out.ndim = v.ndim;`);
  lines.push(`  out.nslots = v.nslots;`);
  lines.push(`  for (size_t _i = 0; _i < 8; _i++) out.dims[_i] = v.dims[_i];`);
  lines.push(`  if (v.nslots == 0) {`);
  lines.push(`    out.slots = NULL;`);
  lines.push(`  } else {`);
  lines.push(
    `    out.slots = (${cTypeFor(elem)}*)malloc(sizeof(${cTypeFor(elem)}) * v.nslots);`
  );
  lines.push(`    for (size_t _i = 0; _i < v.nslots; _i++) {`);
  if (ops !== null) {
    lines.push(`      out.slots[_i] = ${ops.copy}(v.slots[_i]);`);
  } else {
    lines.push(`      out.slots[_i] = v.slots[_i];`);
  }
  lines.push(`    }`);
  lines.push(`  }`);
  lines.push(`  return out;`);
  lines.push(`}`);
  lines.push("");

  // _assign: consume-replace.
  lines.push(`static void ${name}_assign(${name} *p, ${name} rhs) {`);
  lines.push(`  ${name}_free(p);`);
  lines.push(`  *p = rhs;`);
  lines.push(`}`);
  lines.push("");

  // _disp: same numbl format `{e1, e2, ...}`.
  lines.push(`static void ${name}_disp(${name} v) {`);
  lines.push(`  if (v.nslots == 0) { printf("{}"); return; }`);
  lines.push(`  printf("{");`);
  lines.push(`  for (size_t _i = 0; _i < v.nslots; _i++) {`);
  lines.push(`    if (_i > 0) printf(", ");`);
  lines.push(...emitSlotInlineDisp(elem, `v.slots[_i]`, state, "    "));
  lines.push(`  }`);
  lines.push(`  printf("}");`);
  lines.push(`}`);
  lines.push("");

  return lines.join("\n");
}

/** Emit the C lines that print one slot value inline (no trailing
 *  newline). Matches numbl's `formatCell` rules:
 *   - Char slot → `'<value>'` (single-quoted).
 *   - String slot → `"<value>"` (double-quoted).
 *   - Otherwise → the value's own inline display (scalar, tensor, …).
 *
 *  The double / tensor / nested-cell branches reuse the existing
 *  display helpers. */
function emitSlotInlineDisp(
  ty: Type,
  expr: string,
  state: RuntimeState,
  indent: string = "  "
): string[] {
  const lines: string[] = [];
  if (ty.kind === "Char") {
    // Char value is `mtoc2_char_tensor_t { data, rows, cols, owned }` —
    // mtoc2's 1×N byte buffer. Numbl wraps with `'...'`.
    lines.push(`${indent}printf("'");`);
    lines.push(
      `${indent}printf("%.*s", (int)${expr}.cols, (const char*)${expr}.data);`
    );
    lines.push(`${indent}printf("'");`);
    return lines;
  }
  if (ty.kind === "String") {
    // String is `mtoc2_string_t { data, len, owned }`; numbl
    // wraps with `"..."`.
    lines.push(`${indent}printf("\\"");`);
    lines.push(
      `${indent}printf("%.*s", (int)${expr}.len, (const char*)${expr}.data);`
    );
    lines.push(`${indent}printf("\\"");`);
    return lines;
  }
  if (isNumeric(ty) && !isMultiElement(ty)) {
    useRuntimeByName(state, "mtoc2_disp_double");
    // mtoc2_disp_double prints a trailing newline normally; numbl's
    // inline cell-disp doesn't add one. Use the raw format directly.
    lines.push(`${indent}printf("%g", ${expr});`);
    return lines;
  }
  if (isMultiElement(ty)) {
    // Inline-display variant — same body as `mtoc2_disp_tensor` but
    // skips the trailing newline so numbl's `formatCell` interpolation
    // pattern matches byte-for-byte (`{e1, e2, ...   1   2   3}`).
    useRuntimeByName(state, "mtoc2_disp_tensor_inline");
    lines.push(`${indent}mtoc2_disp_tensor_inline(${expr});`);
    return lines;
  }
  if (ty.kind === "Struct") {
    // Nested struct in a cell slot — call its `_disp`. Format may not
    // line up byte-for-byte with numbl for the cell-of-struct case;
    // tests will surface divergences.
    const sname = cTypeFor(ty);
    lines.push(`${indent}${sname}_disp(${expr});`);
    return lines;
  }
  if (ty.kind === "Cell") {
    const cname = cTypeFor(ty);
    lines.push(`${indent}${cname}_disp(${expr});`);
    return lines;
  }
  // Class / Handle in a cell slot: no disp helper. Render a stub
  // matching numbl's `displayValue` fall-through if reachable; the
  // immediate need (chunkie cells) doesn't exercise this.
  lines.push(`${indent}printf("[unsupported cell slot disp]");`);
  return lines;
}

function activateSlotRuntime(ty: Type, state: RuntimeState): void {
  if (isMultiElement(ty)) {
    useRuntimeByName(state, "mtoc2_tensor_empty");
    useRuntimeByName(state, "mtoc2_tensor_assign");
    useRuntimeByName(state, "mtoc2_tensor_copy");
    useRuntimeByName(state, "mtoc2_tensor_free");
  } else if (ty.kind === "String") {
    useRuntimeByName(state, "mtoc2_string_empty");
    useRuntimeByName(state, "mtoc2_string_assign");
    useRuntimeByName(state, "mtoc2_string_copy");
    useRuntimeByName(state, "mtoc2_string_free");
  } else if (ty.kind === "Char") {
    useRuntimeByName(state, "mtoc2_char_tensor_empty");
    useRuntimeByName(state, "mtoc2_char_tensor_assign");
    useRuntimeByName(state, "mtoc2_char_tensor_copy");
    useRuntimeByName(state, "mtoc2_char_tensor_free");
  }
  // For nested Struct / Class / Cell / Handle slots, the typedef
  // ordering guarantees the inner type's helpers are emitted before
  // this typedef.
  // Empty caller for static type checker.
  void shapeNumel; // not used here; reserved for future shape-product math.
}

// JS sibling for struct disp. The c-aot path emits a per-typedef
// `<typedef>_disp(v)` function via emitNamedTypedef; JS doesn't have
// per-struct codegen, so we ship one generic helper that walks the
// struct's keys at runtime and dispatches on the value's shape.
//
// Format mirrors numbl's:
//   "    <field>: <value>\n"  for scalar fields
//   "    <field>: <space><cells>" for tensor fields (mtoc2_disp_tensor
//                                  emits its own 3-space indent + \n)
//   "    <field>: \n<nested>"   for nested struct fields
// References `$write` and the disp helpers as free vars, which
// resolve through the surrounding inlined snippet bodies / interpreter
// imports the same way the other io helpers do.

import { mtoc2_disp_double } from "./disp_double.js";
import { mtoc2_disp_tensor } from "./disp_tensor.js";

export function mtoc2_disp_struct(v) {
  for (const k of Object.keys(v)) {
    const val = v[k];
    $write(`    ${k}: `);
    if (typeof val === "number") {
      mtoc2_disp_double(val);
    } else if (typeof val === "boolean") {
      mtoc2_disp_double(val ? 1 : 0);
    } else if (typeof val === "string") {
      $write(val + "\n");
    } else if (val && val.mtoc2Tag === "tensor") {
      mtoc2_disp_tensor(val);
    } else if (val && val.mtoc2Tag === "char") {
      $write(val.value + "\n");
    } else if (val && typeof val === "object" && "re" in val && "im" in val) {
      // Complex scalar — print `re + imi` in a numbl-compatible form
      // (Phase 5 may want to share this with the complex disp helper).
      const re = val.re;
      const im = val.im;
      const sign = im < 0 ? " - " : " + ";
      $write(`${re}${sign}${Math.abs(im)}i\n`);
    } else if (val && typeof val === "object") {
      // Nested struct — newline then recurse. Numbl's format prints
      // the parent field's value half as just a newline, then the
      // nested fields render with their own indent.
      $write("\n");
      mtoc2_disp_struct(val);
    } else {
      $write(String(val) + "\n");
    }
  }
}

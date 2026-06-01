// JS sibling of `disp_double.h` — disp(x) for a real-scalar double.
// Mirrors numbl's `formatNumber` via `mtoc2_format_double` and adds
// the trailing newline. References `$write` as a free variable
// resolved to `globalThis.$write` at call time.
//
// The import keeps the source loadable as a real ES module (the
// interpreter resolves it that way). The build script strips this
// import when generating the inline-text version for codegen — at
// that point all snippets are inlined into one emitted module, so
// `mtoc2_format_double` resolves via module scope.

import { mtoc2_format_double } from "./format_double.js";

export function mtoc2_disp_double(x) {
  $write(mtoc2_format_double(x) + "\n");
}
